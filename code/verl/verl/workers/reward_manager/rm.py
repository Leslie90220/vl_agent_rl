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
RMManager for VRAG-style reward computation.
Uses local Skywork-Reward model for evaluation.
"""

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import numpy as np
import os
from typing import List, Dict


def dcg(relevance_scores):
    """
    计算折扣累积增益（DCG）
    :param relevance_scores: 一个列表，表示每个文档的相关性分数
    :return: DCG 值
    """
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        dcg_value += (2 ** relevance - 1) / np.log2(i + 1)
    return dcg_value


def ndcg(sorted_docs, golden_answer_list):
    """
    计算归一化折扣累积增益（NDCG）
    :param sorted_docs: 一个列表，表示已经排好序的文档
    :param golden_answer_list: 一个列表，表示所有相关文档（golden answers）
    :return: NDCG 值
    """
    # 将文档映射为相关性分数（在 golden_answer_list 中的文档为 1，否则为 0）
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]
    
    # 计算 DCG
    dcg_value = dcg(relevance_scores)
    
    # 计算 IDCG（理想情况下的 DCG，所有相关文档都排在前面）
    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (len(sorted_docs) - len(golden_answer_list))
    idcg_value = dcg(ideal_relevance_scores)
    
    # 防止分母为零
    if idcg_value == 0:
        return 0.0
    
    # 计算 NDCG
    ndcg_value = dcg_value / idcg_value
    return ndcg_value


def get_answer_from_predict_str(text):
    end_tag = '</answer>'
    start_tag = '<answer>'
    
    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None  # 如果没有找到</answer>，返回None
    
    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None  # 如果没有找到<answer>，返回None
    
    start_pos += len(start_tag)  # 跳过<answer>标签
    return text[start_pos:end_pos]


class LocalRewardModel:
    """Local reward model using Skywork-Reward-V2-Llama-3.2-3B."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the local reward model.
        
        Args:
            model_path: Path to the local model directory (e.g., ./models/Skywork-Reward-V2-Llama-3.2-3B)
            device: Device to run the model on (will fallback to CPU if CUDA unavailable)
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        print(f"[LocalRewardModel] Loading reward model from {model_path}...")
        
        # Check CUDA availability and fallback to CPU if needed
        if device == "cuda" and not torch.cuda.is_available():
            print(f"[LocalRewardModel] Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Use appropriate dtype based on device
        model_dtype = torch.bfloat16 if device != "cpu" else torch.float32
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            device_map=device if device != "cpu" else None
        )
        
        # Move to device if not using device_map
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        print(f"[LocalRewardModel] Reward model loaded successfully on {device}")
    
    def score(self, query: str, response: str, reference: str) -> float:
        """
        Score a response given a query and reference answer.
        
        Args:
            query: The input query/question
            response: The generated response to evaluate
            reference: The reference/ground truth answer
            
        Returns:
            A score between 0 and 1 indicating the quality of the response
        """
        # Format the conversation for the reward model
        # Skywork-Reward uses a conversation format
        conversation = [
            {"role": "user", "content": f"Question: {query}\n\nReference Answer: {reference}"},
            {"role": "assistant", "content": response}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Get reward score
        with torch.no_grad():
            outputs = self.model(**inputs)
            # The model outputs logits, we use sigmoid to get a probability
            score = torch.sigmoid(outputs.logits).item()
        
        return score
    
    def batch_score(self, data_items: List[Dict[str, str]]) -> List[float]:
        """
        Score multiple items in batch.
        
        Args:
            data_items: List of dicts with 'query', 'generated_answer', 'reference_answer' keys
            
        Returns:
            List of scores
        """
        scores = []
        for idx, item in enumerate(data_items):
            score = self.score(
                query=item['query'],
                response=item['generated_answer'],
                reference=item['reference_answer']
            )
            # DEBUG: 打印原始reward model分数
            print(f"[DEBUG LocalRewardModel] Item {idx}: raw_score={score:.4f}, binary={1.0 if score > 0.5 else 0.0}")
            scores.append(score)
        return scores



class RMManager:
    """The reward manager for VRAG-style training using local reward model."""

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None, 
        eval_mode=False, 
        use_local_rm=True,
        local_rm_path="./models/Skywork-Reward-V2-Llama-3.2-3B",
        local_rm_device="cuda",
        **kwargs  # Accept and ignore any extra parameters for backward compatibility
    ) -> None:
        """
        Initialize the reward manager.
        
        Args:
            tokenizer: The tokenizer to use for decoding
            num_examine: Number of batches of decoded responses to print for debugging
            compute_score: Custom score computation function
            eval_mode: Whether in evaluation mode
            use_local_rm: Whether to use local reward model (must be True)
            local_rm_path: Path to local reward model
            local_rm_device: Device for local reward model
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.eval_mode = eval_mode
        
        # Local reward model setup (required)
        self.use_local_rm = use_local_rm
        if not use_local_rm:
            raise ValueError(
                "use_local_rm must be True. Remote API reward model is no longer supported. "
                "Please set use_local_rm=True and provide local_rm_path."
            )
        
        if not os.path.exists(local_rm_path):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            raise ValueError(
                f"Local reward model path does not exist: {local_rm_path}\n"
                f"Please download Skywork-Reward-V2-Llama-3.2-3B to {local_rm_path}"
            )
        self.local_rm = LocalRewardModel(local_rm_path, device=local_rm_device)
    
    def rm_score(self, data_eval_item: Dict[str, str]) -> float:
        """
        Get reward score for a single item using local model.
        
        Args:
            data_eval_item: Dict with 'query', 'generated_answer', 'reference_answer' keys
            
        Returns:
            Score between 0 and 1
        """
        score = self.local_rm.score(
            query=data_eval_item["query"],
            response=data_eval_item["generated_answer"],
            reference=data_eval_item["reference_answer"]
        )
        # Convert to binary (threshold at 0.5)
        return 1.0 if score > 0.5 else 0.0

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores


    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        data_eval = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            generated_answer = get_answer_from_predict_str(self.tokenizer.decode(valid_response_ids))
            if generated_answer is None:
                generated_answer = 'Please Judge False'
            data_eval.append(dict(
                query=extra_info['question'],
                generated_answer=generated_answer,
                reference_answer=data_item.non_tensor_batch['reward_model']['ground_truth']
            ))

        data_to_be_eval = []
        data_to_be_eval_indices = []  # Track which indices need evaluation
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if score > 0.0:
                data_to_be_eval.append(data_eval[i])
                data_to_be_eval_indices.append(i)
        
        eval_results = []
        if len(data_to_be_eval) > 0:
            # Use local model for batch scoring
            print(f"[RMManager] Using local reward model for {len(data_to_be_eval)} items...")
            raw_scores = self.local_rm.batch_score(data_to_be_eval)
            # Convert to binary scores
            eval_results = [1.0 if s > 0.5 else 0.0 for s in raw_scores]
        
        eval_results_iter = iter(eval_results)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # 获取检索图片和参考图片信息（所有sample都需要）
            retrievaled_images_basename_list = [
                os.path.basename(item.rstrip('/')).split(".jpg")[0] 
                for item in data_item.non_tensor_batch['retrievaled_images']
            ] if 'retrievaled_images' in data_item.non_tensor_batch else []
            
            # Handle reference_page as either list or tensor
            reference_page = extra_info.get("reference_page", [])
            if hasattr(reference_page, 'tolist'):
                reference_page = reference_page.tolist()
            elif not isinstance(reference_page, list):
                reference_page = [reference_page] if reference_page is not None else []
            
            file_name = extra_info.get("file_name", "")
            reference_images_basename_list = [
                f'{file_name.split(".pdf")[0]}_{page}' 
                for page in reference_page
            ]
            ndcg_value = ndcg(retrievaled_images_basename_list, reference_images_basename_list)

            if score > 0.0:
                model_eval_score = next(eval_results_iter)
                if self.eval_mode:
                    score = model_eval_score
                else:
                    # DEBUG: 打印关键调试信息（主动回答的sample）
                    print(f"\n[DEBUG RMManager] ========== Sample {i} (answered before max_turns) ==========")
                    print(f"[DEBUG RMManager] data_source: {data_source}")
                    print(f"[DEBUG RMManager] format_score (before): {score}")
                    print(f"[DEBUG RMManager] model_eval_score: {model_eval_score}")
                    print(f"[DEBUG RMManager] ndcg_value: {ndcg_value}")
                    print(f"[DEBUG RMManager] retrievaled_images_basename_list: {retrievaled_images_basename_list[:10]}...")
                    print(f"[DEBUG RMManager] reference_images_basename_list: {reference_images_basename_list}")
                    print(f"[DEBUG RMManager] file_name: {file_name}")
                    print(f"[DEBUG RMManager] reference_page: {reference_page}")
                    print(f"[DEBUG RMManager] data_eval[{i}]: query={data_eval[i]['query'][:100]}...")
                    print(f"[DEBUG RMManager] data_eval[{i}]: generated_answer={data_eval[i]['generated_answer'][:200] if data_eval[i]['generated_answer'] else 'None'}...")
                    print(f"[DEBUG RMManager] data_eval[{i}]: reference_answer={data_eval[i]['reference_answer'][:200] if data_eval[i]['reference_answer'] else 'None'}...")

                    old_score = score
                    score = 0.7 * model_eval_score + 0.1 * score + 0.2 * ndcg_value
                    print(f"[DEBUG RMManager] final_score: 0.7*{model_eval_score} + 0.1*{old_score} + 0.2*{ndcg_value} = {score}")
                    print(f"[DEBUG RMManager] =====================================\n")
            else:
                # DEBUG: 打印被强制回答的sample信息（score=0，即在max_turns结束时被强制回答）
                print(f"\n[DEBUG RMManager] ========== Sample {i} (forced answer at max_turns) ==========")
                print(f"[DEBUG RMManager] data_source: {data_source}")
                print(f"[DEBUG RMManager] format_score: {score} (no valid answer format)")
                print(f"[DEBUG RMManager] model_eval_score: N/A (skipped due to format_score=0)")
                print(f"[DEBUG RMManager] ndcg_value: {ndcg_value}")
                print(f"[DEBUG RMManager] retrievaled_images_basename_list: {retrievaled_images_basename_list[:10]}...")
                print(f"[DEBUG RMManager] reference_images_basename_list: {reference_images_basename_list}")
                print(f"[DEBUG RMManager] file_name: {file_name}")
                print(f"[DEBUG RMManager] reference_page: {reference_page}")
                print(f"[DEBUG RMManager] data_eval[{i}]: query={data_eval[i]['query'][:100]}...")
                print(f"[DEBUG RMManager] data_eval[{i}]: generated_answer={data_eval[i]['generated_answer'][:200] if data_eval[i]['generated_answer'] else 'None'}...")
                print(f"[DEBUG RMManager] data_eval[{i}]: reference_answer={data_eval[i]['reference_answer'][:200] if data_eval[i]['reference_answer'] else 'None'}...")
                print(f"[DEBUG RMManager] final_score: {score}")
                print(f"[DEBUG RMManager] =====================================\n")

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
