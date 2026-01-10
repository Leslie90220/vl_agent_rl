# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
VRAG-style PPO Trainer with Ray-based single controller.
This trainer supports multi-turn agent interaction for visual RAG tasks.
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import re
import torch

# Import VRAG agent components
import sys
code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from vrag_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """Role definitions for different worker types."""
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """Advantage estimation methods."""
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """Manage resource pools for different worker roles."""
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    """Apply KL penalty to token-level scores."""
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'], kl_penalty=kl_penalty)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    """Compute advantages using the specified estimator."""
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=gamma,
            lam=lam
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        # Create a simple config object with gamma for REINFORCE++
        from omegaconf import OmegaConf
        reinforce_config = OmegaConf.create({'gamma': gamma})
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, response_mask=response_mask, config=reinforce_config
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        reward_baselines = data.batch['reward_baselines']
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards,
            reward_baselines=reward_baselines,
            response_mask=response_mask
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last



class RayPPOTrainerVRAG(object):
    """
    VRAG-style PPO Trainer with multi-turn agent interaction.
    Runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # Define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS,
                AdvantageEstimator.REMAX, AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        """Validate configuration parameters."""
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")
            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size'.")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding

        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        """Create train and validation dataloaders."""
        from omegaconf import OmegaConf
        
        # Handle seed - use default if None or not set
        data_seed = self.config.data.get('seed')
        if data_seed is None:
            data_seed = 42  # Default seed
        
        # Create a config dict for RLHFDataset that matches official verl API
        train_data_config = OmegaConf.create({
            'prompt_key': self.config.data.prompt_key,
            'image_key': self.config.data.get('image_key', 'images'),
            'max_prompt_length': self.config.data.max_prompt_length,
            'filter_prompts': True,
            'return_raw_chat': self.config.data.get('return_raw_chat', False),
            'truncation': self.config.data.get('truncation', 'error'),
            'filter_overlong_prompts': self.config.data.filter_overlong_prompts,
            'shuffle': self.config.data.get('shuffle', False),
            'seed': data_seed,
        })
        
        self.train_dataset = RLHFDataset(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            config=train_data_config,
            processor=self.processor,
        )

        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(data_seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler
        )

        val_data_config = OmegaConf.create({
            'prompt_key': self.config.data.prompt_key,
            'image_key': self.config.data.get('image_key', 'images'),
            'max_prompt_length': self.config.data.max_prompt_length,
            'filter_prompts': True,
            'return_raw_chat': self.config.data.get('return_raw_chat', False),
            'truncation': self.config.data.get('truncation', 'error'),
            'filter_overlong_prompts': self.config.data.filter_overlong_prompts,
            'shuffle': False,
        })
        
        self.val_dataset = RLHFDataset(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            config=val_data_config,
            processor=self.processor,
        )

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=505,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn
        )

        assert len(self.train_dataloader) >= 1
        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def init_workers(self):
        """Initialize resource pool and worker groups."""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role='actor_rollout'
            )
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role='ref'
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

        # Create async rollout manager for vLLM async mode
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager
            
            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=None,  # VRAG uses its own reward manager
            )

    def _save_checkpoint(self):
        """Save model checkpoint."""
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir,
            f'global_step_{self.global_steps}'
        )
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps
            )

        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir,
            'latest_checkpointed_iteration.txt'
        )
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """Load model checkpoint if available."""
        if self.config.trainer.resume_mode == 'disable':
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            resume_from_path = self.config.trainer.get('resume_from_path', False)
            if not (resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str)
                assert 'global_step_' in self.config.trainer.resume_mode
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)

        print(f'Load from checkpoint folder: {global_step_folder}')
        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        print(f'Setting global step to {self.global_steps}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')

        # Use .get() to safely access optional config with default value
        del_local_after_load = self.config.trainer.get('del_local_ckpt_after_load', False)
        
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            hdfs_path=None,
            del_local_after_load=del_local_after_load
        )

        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path,
                hdfs_path=None,
                del_local_after_load=del_local_after_load
            )

        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder data for balanced token distribution across DP ranks."""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst,
            k_partitions=world_size,
            equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix
        )
        metrics.update(global_balance_stats)


    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens (mask user turns)."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        # Add response_mask to batch (required by dp_actor)
        batch.batch['response_mask'] = response_mask
        
        state_mask = torch.ones_like(response_mask)
        responses = [self.tokenizer.decode(resp, skip_special_tokens=False) for resp in batch.batch['responses']]
    
        for i, response in enumerate(responses):
            start_marker = "\n<|im_start|>user\n"
            end_marker = "<|im_end|>\n<|im_start|>assistant\n"
            
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), response)]
            end_positions = [m.start() + len(end_marker) for m in re.finditer(re.escape(end_marker), response)]
            
            for start, end in zip(start_positions, end_positions):
                prefix_to_start = response[:start]
                state_section = response[start:end]
                
                start_tokens = self.tokenizer.encode(prefix_to_start, add_special_tokens=False)
                state_tokens = self.tokenizer.encode(state_section, add_special_tokens=False)
                
                start_token_pos = len(start_tokens)
                end_token_pos = start_token_pos + len(state_tokens)
                
                state_mask[i, start_token_pos:end_token_pos] = 0

        # Mask last user turn with vision tokens
        for i, response in enumerate(responses): 
            start_marker = "\n<|im_start|>user\n"
            end_marker = "<|im_end|>"
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), response)]
            end_positions = [m.start() + len(end_marker) for m in re.finditer(re.escape(end_marker), response)]
            for start, end in zip(start_positions, end_positions):
                prefix_to_start = response[:start]
                state_section = response[start:end]
                start_tokens = self.tokenizer.encode(prefix_to_start, add_special_tokens=False)
                state_tokens = self.tokenizer.encode(state_section, add_special_tokens=False)
                start_token_pos = len(start_tokens)
                end_token_pos = start_token_pos + len(state_tokens)
                state_mask[i, start_token_pos:end_token_pos] = 0
            
        loss_mask = state_mask * response_mask
        batch.batch['loss_mask'] = loss_mask
        
        return batch, metrics

    def fit(self):
        """
        Main training loop for VRAG-style PPO.
        Supports multi-turn agent interaction with search operations.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        #设置日志追踪
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True)
        )

        self.global_steps = 0
        #加载检查点（如果有）
        self._load_checkpoint()
        
        # DEBUG: 打印训练循环关键参数
        print(f"\n{'='*60}")
        print(f"[DEBUG fit()] Training Loop Configuration:")
        print(f"  total_training_steps: {self.total_training_steps}")
        print(f"  total_epochs: {self.config.trainer.total_epochs}")
        print(f"  train_dataloader length: {len(self.train_dataloader)}")
        print(f"  train_batch_size: {self.config.data.train_batch_size}")
        print(f"  global_steps (after checkpoint load): {self.global_steps}")
        print(f"  resume_mode: {self.config.trainer.resume_mode}")
        print(f"{'='*60}\n")

        # Validation before training训练前验证（可选）
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        self.global_steps += 1
        last_val_metrics = None

        # VRAG Agent configuration配置
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_prompt_length=99999,
            num_gpus=self.config.trainer.n_gpus_per_node,
            n_agent=self.config.actor_rollout_ref.rollout.n_agent,
            search_url=self.config.retriever.url,
        )

        # Use async_rollout_manager if available, otherwise use actor_rollout_wg
        rollout_manager = self.async_rollout_manager if self.async_rollout_mode else self.actor_rollout_wg
        
        generation_manager = LLMGenerationManager(
            processor=self.processor,
            actor_rollout_wg=rollout_manager,
            config=gen_config,
        )

        # Training loop训练主循环,目前epoch=3
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # 数据准备，复制 batch，每个样本生成 n_agent=2 个候选
                gen_batch = batch.repeat_deepcopy(
                    repeat_times=self.config.actor_rollout_ref.rollout.n_agent,
                    interleave=True
                )

                # Pop keys for generation
                # Include data_source, reward_model, extra_info for reward computation
                if 'multi_modal_inputs' in gen_batch.non_tensor_batch.keys():
                    gen_batch = gen_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'raw_prompt', 'multi_modal_data', 'multi_modal_inputs', 'data_source', 'reward_model', 'extra_info'],
                    )
                else:
                    gen_batch = gen_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'raw_prompt', 'data_source', 'reward_model', 'extra_info'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                
                # DEBUG: 每个step打印进度信息
                # [DEBUG fit()] Step n/426, is_last_step=False, epoch=1/3
                print(f"\n========= 开始第 {self.global_steps} 步训练========= ")
                print(f"\n[DEBUG fit()] Step {self.global_steps}/{self.total_training_steps}, is_last_step={is_last_step}, epoch={epoch+1}/{self.config.trainer.total_epochs}")
                
                # Multi-turn agent generation loop
                with _timer('step', timing_raw):
                    first_input_ids = gen_batch.batch['input_ids'].clone().long()  
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        # 多轮Agent交互生成循环
                        # │    1. 模型生成响应 → <think>...</think><search>query</search>
                        # │    2. 解析动作：search/answer/bbox                          
                        # │    3. 执行环境交互：                                         
                        # │       - search → 调用搜索引擎获取图像                        
                        # │       - bbox → 裁剪图像                                      
                        # │       - answer → 结束                                        
                        # │    4. 将检索结果拼接到上下文                                 
                        # │    5. 更新状态，进入下一轮     
                        
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    print('========= 结束打印n个sample的run_llm_loop过程 =========')
                    for key in final_gen_batch_output.batch.keys():
                        final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()
                    
                    batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                    batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object
                    )
                    
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n_agent,
                        interleave=True
                    )
                    
                    # Remove multi_modal_data and multi_modal_inputs from batch before union
                    # since final_gen_batch_output has the updated data with retrieved images
                    if 'multi_modal_data' in batch.non_tensor_batch:
                        del batch.non_tensor_batch['multi_modal_data']
                    if 'multi_modal_inputs' in batch.non_tensor_batch:
                        del batch.non_tensor_batch['multi_modal_inputs']
                    
                    batch = batch.union(final_gen_batch_output)

                    # Process extra_info for RAG dataset
                    extra_info_list = []
                    for extra_info in batch.non_tensor_batch['extra_info']:
                        file_name = extra_info.get('file_name', '')
                        reference_page = extra_info.get('reference_page', [])
                        extra_info_list.append(dict(
                            file_name=file_name,
                            reference_page=reference_page,
                            question=extra_info['question'],
                            answer=extra_info['answer']
                        ))
                    batch.non_tensor_batch['extra_info'] = np.array(extra_info_list)
                    
                    for idx in range(len(batch.non_tensor_batch['raw_prompt_ids'])):
                        batch.non_tensor_batch['raw_prompt_ids'][idx] = [
                            int(item) for item in batch.non_tensor_batch['raw_prompt_ids'][idx]
                        ]

                    # Balance batch across DP ranks
                    if self.config.trainer.get('balance_batch', False):
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info['global_token_num'] = torch.sum(
                        batch.batch['attention_mask'], dim=-1
                    ).tolist()

                    # Recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        with torch.no_grad():
                            print('========= compute log prob =========')
                            # 当前策略的 log prob
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    # Reference policy log prob
                    if self.use_reference_policy:
                        print('===============ref log prob==================')
                        
                        # === DEBUG: 打印每个样本的检索结果是否包含 reference_page ===
                        if 'retrievaled_images' in batch.non_tensor_batch and 'extra_info' in batch.non_tensor_batch:
                            print('\n[RETRIEVAL SUMMARY] Checking if retrieved images contain reference pages:')
                            num_to_print = min(16, len(batch.non_tensor_batch['retrievaled_images']))
                            for i in range(num_to_print):
                                retrieved_images = batch.non_tensor_batch['retrievaled_images'][i]
                                extra_info = batch.non_tensor_batch['extra_info'][i]
                                
                                # 获取 reference_page 和 file_name
                                reference_page = extra_info.get('reference_page', None) if extra_info else None
                                file_name = extra_info.get('file_name', None) if extra_info else None
                                
                                # 构建期望的图片名称
                                expected_image_names = set()
                                if file_name and reference_page:
                                    file_base = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                                    if isinstance(reference_page, list):
                                        for page in reference_page:
                                            expected_image_names.add(f"{file_base}_{page}")
                                    elif hasattr(reference_page, 'tolist'):
                                        for page in reference_page.tolist():
                                            expected_image_names.add(f"{file_base}_{page}")
                                    else:
                                        expected_image_names.add(f"{file_base}_{reference_page}")
                                
                                # 检查检索到的图片是否包含期望的图片
                                retrieved_basenames = set()
                                for img_path in retrieved_images:
                                    if img_path and isinstance(img_path, str):
                                        import os
                                        basename = os.path.basename(img_path)
                                        img_name = basename.rsplit('.', 1)[0] if '.' in basename else basename
                                        retrieved_basenames.add(img_name)
                                
                                found_reference = bool(expected_image_names & retrieved_basenames)
                                status = "✓ FOUND" if found_reference else "✗ NOT FOUND"
                                print(f"  Sample {i}: {status} | retrieved={len(retrieved_images)} pages | expected={expected_image_names}")
                            print("END compute log and ref prob")
                        # === END DEBUG ===
                        
                        with _timer('ref', timing_raw):
                            # 参考策略的 log prob（用于 KL 惩罚）
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # Compute values (if using critic)
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute advantages
                    with _timer('adv', timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        # 2. 计算奖励
                        # 调用 RMManager 计算奖励
                        # 奖励 = 0.7*模型评分 + 0.1*格式评分 + 0.2*NDCG检索质量
                        print('=== 开始计算并打印已经生成了回答的item的模型格式打分和n个sample的最终表现 ===')
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            # KL 惩罚（可选）
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl,
                                kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        print('=== 完成计算并打印已经生成了回答的item的模型格式打分和n个sample的最终表现 ===')
                        print('========= compute advantage =========')
                        # GAE 优势估计
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n_agent
                        )
                        print('========= 完成compute advantage=========')

                    # Update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            # 更新 Critic（价值网络）
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # Update actor (after critic warmup)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer('update_actor', timing_raw):
                            batch, metrics = self._create_loss_mask(batch, metrics)
                            # 更新 Actor（策略网络）- 在 critic_warmup 之后
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # Save checkpoint定期保存检查点
                    if self.config.trainer.save_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                    # Validation定期验证
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    print(f"\n========= 第 {self.global_steps} 步全部完成========= ")
                # Collect and log metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    print(f"\n[DEBUG fit()] Exiting training loop: global_steps={self.global_steps}, total_training_steps={self.total_training_steps}")
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                self.global_steps += 1


    def _validate(self):
        """
        Validation loop with multi-turn agent interaction.
        Accumulates metrics across all batches before computing final statistics.
        """
        reward_tensor_lst = []
        data_source_lst = []

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_prompt_length=99999,
            num_gpus=self.config.trainer.n_gpus_per_node,
            n_agent=self.config.actor_rollout_ref.rollout.get('n_agent', 1),
            search_url=self.config.retriever.url,
        )

        # Use async_rollout_manager if available, otherwise use actor_rollout_wg
        # This is the same logic as in the training loop (fit method)
        rollout_manager = self.async_rollout_manager if self.async_rollout_mode else self.actor_rollout_wg

        generation_manager = LLMGenerationManager(
            processor=self.processor,
            actor_rollout_wg=rollout_manager,
            config=gen_config,
            is_validation=True
        )

        for test_data in self.val_dataloader:
            timing_raw = {}
            test_batch = DataProto.from_single_dict(test_data)

            test_batch = test_batch.repeat_deepcopy(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True
            )

            # Include data_source, reward_model, extra_info for reward computation
            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'raw_prompt', 'multi_modal_data', 'multi_modal_inputs', 'data_source', 'reward_model', 'extra_info'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'raw_prompt', 'data_source', 'reward_model', 'extra_info'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            with _timer('step', timing_raw):
                first_input_ids = test_gen_batch.batch['input_ids'].clone().long()
                with _timer('gen', timing_raw):
                    generation_manager.timing_raw = timing_raw
                    final_gen_batch_output = generation_manager.run_llm_loop(
                        gen_batch=test_gen_batch,
                        initial_input_ids=first_input_ids,
                    )
                
                test_batch = test_batch.union(final_gen_batch_output)
                
                for key in test_batch.batch.keys():
                    test_batch.batch[key] = test_batch.batch[key].long()
                
                reward_tensor = self.val_reward_fn(test_batch)
                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(
                    test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])
                )

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        metric_dict = {}

        # Evaluate by data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
        
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        # Overall score
        overall_rewards = [reward_tensor[i].item() for i in range(reward_tensor.shape[0])]
        metric_dict[f'val/test_score/overall'] = np.mean(overall_rewards)
        
        return metric_dict
