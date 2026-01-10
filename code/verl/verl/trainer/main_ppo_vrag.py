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
VRAG-style PPO trainer entry point.
This module provides VRAG-specific training with multi-turn agent interaction.
"""

# Apply nest_asyncio early to allow nested event loops in Ray workers
import nest_asyncio
nest_asyncio.apply()

import os
import ray
import hydra

from verl.trainer.ppo.ray_trainer_vrag import RayPPOTrainerVRAG


def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)

# yaml配置文件路径：../config/ppo_trainer.yaml
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    # 加载参数
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    # Ray进程初始化
    if not ray.is_initialized():
        # this is for local ray cluster
        # Pass VLLM_USE_V1 to Ray workers - must be set before vLLM import
        vllm_use_v1 = os.environ.get('VLLM_USE_V1', '1')
        vllm_attention_backend = os.environ.get('VLLM_ATTENTION_BACKEND', 'FLASH_ATTN')
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN',
                'VLLM_USE_V1': vllm_use_v1,
                'VLLM_ATTENTION_BACKEND': vllm_attention_backend,
            }
        })
    # 创建一个 TaskRunner 类的远程 Actor 实例
    runner = TaskRunner.remote()
    # ray.get() 阻塞主进程等待远程任务完成并获取返回值，异步调用远程 Actor 的 run 方法，传入配置参数，返回一个 ObjectRef（future 对象）
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # Reward model runs on CPU, no GPU needed
class TaskRunner:
    def run(self, config):
        import debugpy
        try:
            debugpy.listen(("localhost", 9503))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
        except Exception as e:
            pass
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        # actor and reward model的本地路径
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        tokenizer = hf_tokenizer(local_path)
        #从本地加载模型
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            # Use AsyncActorRolloutRefWorker for async rollout mode
            if config.actor_rollout_ref.rollout.mode == "async":
                from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as ActorRolloutRefWorker, CriticWorker
            else:
                from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError
        #ray_trainer_vrag！！！
        from verl.trainer.ppo.ray_trainer_vrag import ResourcePoolManager, Role
        # 角色到 Worker 类的映射
        role_worker_mapping = {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
        }

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # 角色到 GPU 资源池的映射(共用一组global_pool的GPU)
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "naive")

        # VRAG only supports 'rm' reward manager
        assert reward_manager_name in ['rm', 'naive'], f"vrag supports reward manager: rm, naive. Got {reward_manager_name}"
        # 从配置文件动态加载自定义的奖励计算函数
        compute_score = get_custom_reward_fn(config)

        # Import RMManager for VRAG-style reward computation
        from verl.workers.reward_manager import RMManager

        # Local reward model parameters - always use CPU to avoid GPU memory conflicts
        rm_params = dict(
            use_local_rm=config.reward_model.get("use_local_rm", True),
            local_rm_path=config.reward_model.get("local_rm_path", "./models/Skywork-Reward-V2-Llama-3.2-3B"),
            local_rm_device="cpu",  # Force CPU to avoid GPU memory conflicts with training
        )

        reward_fn = RMManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, eval_mode=False, **rm_params)
        val_reward_fn = RMManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, eval_mode=True, **rm_params)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayPPOTrainerVRAG(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':

    main()
