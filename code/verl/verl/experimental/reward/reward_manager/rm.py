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
RMRewardLoopManager - Reward loop manager wrapper for VRAG-style RMManager.
This bridges the experimental reward loop system with the existing RMManager.
"""

import inspect
import os
import re
import numpy as np

from verl import DataProto
from verl.experimental.reward.reward_manager import register
from verl.experimental.reward.reward_manager.base import RewardLoopManagerBase


def get_answer_from_predict_str(text):
    """Extract answer from <answer>...</answer> tags."""
    end_tag = '</answer>'
    start_tag = '<answer>'
    
    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None
    
    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None
    
    start_pos += len(start_tag)
    return text[start_pos:end_pos]


def dcg(relevance_scores):
    """Calculate Discounted Cumulative Gain (DCG)."""
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        dcg_value += (2 ** relevance - 1) / np.log2(i + 1)
    return dcg_value


def ndcg(sorted_docs, golden_answer_list):
    """Calculate Normalized Discounted Cumulative Gain (NDCG)."""
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]
    dcg_value = dcg(relevance_scores)
    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (len(sorted_docs) - len(golden_answer_list))
    idcg_value = dcg(ideal_relevance_scores)
    if idcg_value == 0:
        return 0.0
    return dcg_value / idcg_value


def vrag_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Compute score for VRAG datasets (vidoseek, mmlongdoc, etc.).
    
    For VRAG, we check if the generated answer contains the ground truth.
    This is a simple rule-based scoring that returns 1.0 if correct, 0.0 otherwise.
    """
    # Extract answer from the solution string
    generated_answer = get_answer_from_predict_str(solution_str)
    
    if generated_answer is None:
        # No answer tag found
        return 0.0
    
    # Normalize both strings for comparison
    generated_answer = generated_answer.strip().lower()
    ground_truth_normalized = ground_truth.strip().lower() if ground_truth else ""
    
    # Check if ground truth is contained in generated answer or vice versa
    if ground_truth_normalized in generated_answer or generated_answer in ground_truth_normalized:
        return 1.0
    
    # Also check for exact match after removing punctuation
    import string
    gen_clean = generated_answer.translate(str.maketrans('', '', string.punctuation))
    gt_clean = ground_truth_normalized.translate(str.maketrans('', '', string.punctuation))
    
    if gen_clean == gt_clean:
        return 1.0
    
    # Check word overlap for partial credit
    gen_words = set(gen_clean.split())
    gt_words = set(gt_clean.split())
    
    if gt_words and gen_words:
        overlap = len(gen_words & gt_words) / len(gt_words)
        if overlap >= 0.8:  # 80% word overlap
            return 1.0
    
    return 0.0


@register("rm")
class RMRewardLoopManager(RewardLoopManagerBase):
    """
    Reward loop manager for VRAG-style training.
    This wraps the RMManager functionality for use with the experimental reward loop system.
    """

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        # Use custom compute_score if provided, otherwise use vrag_compute_score
        self.compute_score = compute_score or vrag_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        
        # Get RM config from reward_model section
        rm_config = getattr(config, 'reward_model', config)
        self.eval_mode = getattr(rm_config, 'eval_mode', False)
        self.num_examine = getattr(rm_config, 'num_examine', 0)

    async def run_single(self, data: DataProto) -> dict:
        """
        Compute reward for a single data item.
        
        This method is called by RewardLoopWorker for each data item.
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        
        # Extract response
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Get metadata
        data_source = data_item.non_tensor_batch.get("data_source", "vidoseek")
        reward_model_data = data_item.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_model_data, dict):
            ground_truth = reward_model_data.get("ground_truth", "")
        else:
            ground_truth = ""
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        
        # Handle tool_extra_fields if present
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            if isinstance(extra_info, dict):
                extra_info.update(tool_extra_fields.items() if hasattr(tool_extra_fields, 'items') else {})

        # Get turn info
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        if isinstance(extra_info, dict):
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

        # Decode response
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # Compute base score using compute_score function
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result.get("score", 0.0)
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = float(result) if result is not None else 0.0
            reward_extra_info["acc"] = score

        # For VRAG, we may want to incorporate retrieval quality (NDCG)
        # This is optional and depends on whether retrieval info is available
        retrievaled_images = data_item.non_tensor_batch.get('retrievaled_images', None)
        if retrievaled_images is not None and score > 0.0 and not self.eval_mode:
            try:
                retrievaled_images_basename_list = [
                    os.path.basename(item.rstrip('/')).split(".jpg")[0] 
                    for item in retrievaled_images
                ]
                reference_page = extra_info.get("reference_page", None) if isinstance(extra_info, dict) else None
                file_name = extra_info.get("file_name", "") if isinstance(extra_info, dict) else ""
                
                if reference_page is not None:
                    reference_images_basename_list = [
                        f'{file_name.split(".pdf")[0]}_{page}' 
                        for page in (reference_page.tolist() if hasattr(reference_page, 'tolist') else reference_page)
                    ]
                    ndcg_value = ndcg(retrievaled_images_basename_list, reference_images_basename_list)
                    # Combine scores: 70% model eval + 10% base score + 20% NDCG
                    score = 0.7 * score + 0.1 * reward_extra_info.get("acc", score) + 0.2 * ndcg_value
                    reward_extra_info["ndcg"] = ndcg_value
            except Exception as e:
                # If NDCG calculation fails, just use the base score
                pass

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
