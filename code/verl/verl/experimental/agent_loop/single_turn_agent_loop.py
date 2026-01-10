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
        
        # Store multi_modal_inputs computed during prompt generation
        # This ensures image_grid_thw matches the expanded image tokens in prompt_ids
        computed_multi_modal_inputs = None

        # Check if we have multimodal data
        has_multimodal = image_data is not None and (len(image_data) > 0 if isinstance(image_data, list) else True)
        
        # For multimodal inputs, we need to use the processor to get prompt_ids with
        # correctly expanded image tokens that match the image_grid_thw
        if has_multimodal and self.processor is not None:
            # Use processor to get prompt_ids with expanded image tokens
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            
            # Remove attention_mask as we'll compute it later based on padding
            model_inputs.pop("attention_mask", None)
            
            # Store the multi_modal_inputs (pixel_values, image_grid_thw, etc.)
            # This ensures the image_grid_thw matches the expanded tokens in prompt_ids
            computed_multi_modal_inputs = dict(model_inputs)
        elif self.processor is not None:
            # Non-multimodal but has processor
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=None, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            # Text-only with tokenizer
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )

        with simple_timer("generate_sequences", metrics):
            # CRITICAL: When prompt_ids already has expanded image tokens (from processor),
            # we should NOT pass image_data to vLLM. vLLM's _qwen2_5_vl_dedup_image_tokens
            # will collapse the expanded tokens, and then vLLM will re-expand them based on
            # image_data. However, this can cause mismatches if the image processing differs.
            # 
            # When computed_multi_modal_inputs is set, it means we used the processor to
            # expand image tokens in prompt_ids, so we should NOT pass image_data to vLLM.
            # The multi_modal_inputs (pixel_values, image_grid_thw) will be used later in
            # postprocess for position_ids calculation.
            image_data_for_vllm = None if computed_multi_modal_inputs is not None else image_data
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data_for_vllm
            )
        response_mask = [1] * len(output.token_ids)

        # Build extra_fields to pass computed_multi_modal_inputs to postprocess
        extra_fields = {}
        if computed_multi_modal_inputs is not None:
            extra_fields["computed_multi_modal_inputs"] = computed_multi_modal_inputs

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
