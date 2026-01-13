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
import copy
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))
        # Ensure empty list is treated as None to avoid processor errors
        if image_data is not None and len(image_data) == 0:
            image_data = None

        metrics = {}
        request_id = uuid4().hex
        
        # Store computed multi_modal_inputs for postprocessing
        computed_multi_modal_inputs = None
        # Flag to indicate if we're using collapsed prompt_ids
        using_collapsed_prompt_ids = False

        # CRITICAL FIX: For multimodal inputs with vLLM async mode
        # 
        # In VRAG training, raw_prompt (message list) may not be updated across turns,
        # but multi_modal_data is accumulated. This causes a mismatch:
        # - raw_prompt may have 1 image placeholder
        # - multi_modal_data may have 2+ images
        # 
        # Solution: Use raw_prompt_ids (pre-computed token IDs with collapsed image tokens)
        # for vLLM generation. Mark that we're using collapsed prompt_ids so postprocessing
        # can handle position_ids calculation correctly.
        #
        # The flow is:
        # 1. Use raw_prompt_ids for vLLM generation (collapsed image tokens)
        # 2. Pass image_data to vLLM
        # 3. vLLM will expand tokens based on actual image data
        # 4. For postprocessing, skip get_rope_index since prompt_ids are collapsed
        
        # Check if raw_prompt_ids is available (from VRAG training loop)
        if "raw_prompt_ids" in kwargs and kwargs["raw_prompt_ids"] is not None:
            # Use pre-computed raw_prompt_ids for vLLM generation (collapsed image tokens)
            prompt_ids = list(kwargs["raw_prompt_ids"])
            using_collapsed_prompt_ids = True
        elif self.processor is not None and image_data is not None:
            # Fallback: Use processor to get prompt_ids with expanded image tokens
            # This path is used when raw_prompt_ids is not available
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # processor() will expand image tokens based on image dimensions
            # _qwen2_5_vl_dedup_image_tokens in vLLM will collapse them back
            model_inputs = await self.loop.run_in_executor(
                None,
                lambda: self.processor(text=[raw_prompt], images=image_data, return_tensors="pt"),
            )
            prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
            # Store multi_modal_inputs for postprocessing
            computed_multi_modal_inputs = {k: v for k, v in model_inputs.items() if k not in ["input_ids", "attention_mask"]}
        elif self.processor is not None:
            # Text-only with processor
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            )
        else:
            # Text-only with tokenizer
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )

        with simple_timer("generate_sequences", metrics):
            # ALWAYS pass image_data to vLLM when we have multimodal data
            # vLLM will:
            # 1. Call _qwen2_5_vl_dedup_image_tokens to collapse any remaining expanded tokens
            # 2. Process image_data to get pixel_values
            # 3. Re-expand tokens based on actual image dimensions
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
            )
        
        response_mask = [1] * len(output.token_ids)

        # Build extra_fields
        extra_fields = {}
        if computed_multi_modal_inputs:
            extra_fields["computed_multi_modal_inputs"] = computed_multi_modal_inputs
        if using_collapsed_prompt_ids:
            extra_fields["using_collapsed_prompt_ids"] = True

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output
