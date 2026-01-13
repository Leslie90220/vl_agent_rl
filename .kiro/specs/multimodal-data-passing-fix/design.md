# Design Document: Multimodal Data Passing Fix

## Overview

本设计文档描述了如何修复 VRAG 训练中多模态数据（pixel_values）无法正确传递给模型的问题。核心问题是：当前的 `generate_sequences` 方法只使用 `input_ids`, `attention_mask`, `position_ids`，而忽略了存储在 `non_tensor_batch` 中的 `multi_modal_inputs`（包含 pixel_values 和 image_grid_thw）。

## Architecture

### 当前问题分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Current Data Flow (BROKEN)                            │
└─────────────────────────────────────────────────────────────────────────┘

LLMGenerationManager.run_llm_loop()
        │
        ▼
_generate_with_gpu_padding(active_batch)
        │
        │  active_batch.batch = {input_ids, attention_mask, position_ids}
        │  active_batch.non_tensor_batch = {multi_modal_inputs: {pixel_values, image_grid_thw}}
        │
        ▼
actor_rollout_wg.generate_sequences(active_batch)
        │
        │  ❌ Only uses: input_ids, attention_mask, position_ids
        │  ❌ IGNORES: non_tensor_batch['multi_modal_inputs']
        │
        ▼
model.generate(input_ids, attention_mask, position_ids)
        │
        │  ❌ No pixel_values passed!
        │  ❌ Model cannot "see" images!
        │
        ▼
Model outputs: "No evidence image provided; search for..."
```

### 解决方案架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Fixed Data Flow                                       │
└─────────────────────────────────────────────────────────────────────────┘

LLMGenerationManager.run_llm_loop()
        │
        ▼
_generate_with_gpu_padding(active_batch)
        │
        │  active_batch.batch = {input_ids, attention_mask, position_ids}
        │  active_batch.non_tensor_batch = {multi_modal_inputs: {pixel_values, image_grid_thw}}
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Option A: Modify generate_sequences to handle multi_modal_inputs        │
│  - Extract pixel_values from non_tensor_batch                            │
│  - Pass to model.generate() via **kwargs                                 │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
model.generate(input_ids, attention_mask, position_ids, 
               pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        │
        │  ✅ pixel_values passed!
        │  ✅ Model can "see" images!
        │
        ▼
Model outputs: {"think": "Based on image_01...", "action": "answer", ...}
```

## Components and Interfaces

### 1. 修改 _generate_with_gpu_padding 方法

当前方法只是简单地调用 `self.actor_rollout_wg.generate_sequences(active_batch)`，需要修改为：

```python
def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
    """
    Wrapper for generation that handles multi-GPU padding and multimodal data.
    
    Key changes:
    1. Extract multi_modal_inputs from non_tensor_batch
    2. Batch pixel_values across samples
    3. Pass to generate_sequences with multimodal support
    """
    # ... existing padding logic ...
    
    # NEW: Prepare multimodal inputs for generation
    if 'multi_modal_inputs' in active_batch.non_tensor_batch:
        # Batch pixel_values and image_grid_thw
        batched_pixel_values, batched_image_grid_thw = self._batch_multimodal_inputs(
            active_batch.non_tensor_batch['multi_modal_inputs']
        )
        # Store in batch for rollout worker to use
        active_batch.meta_info['pixel_values'] = batched_pixel_values
        active_batch.meta_info['image_grid_thw'] = batched_image_grid_thw
    
    return self.actor_rollout_wg.generate_sequences(active_batch)
```

### 2. 新增 _batch_multimodal_inputs 方法

```python
def _batch_multimodal_inputs(self, multi_modal_inputs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch multimodal inputs from per-sample format to batched tensors.
    
    Args:
        multi_modal_inputs: Array of per-sample multimodal inputs
        
    Returns:
        Tuple of (batched_pixel_values, batched_image_grid_thw)
    """
    pixel_values_list = []
    image_grid_thw_list = []
    
    for mm_input in multi_modal_inputs:
        if mm_input is not None and 'pixel_values' in mm_input:
            pixel_values_list.append(mm_input['pixel_values'])
            image_grid_thw_list.append(mm_input['image_grid_thw'])
    
    if not pixel_values_list:
        return None, None
    
    # Concatenate along batch dimension
    batched_pixel_values = torch.cat(pixel_values_list, dim=0)
    batched_image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
    
    return batched_pixel_values, batched_image_grid_thw
```

### 3. 修改 vLLM Rollout Worker

需要修改 `code/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py` 中的 `generate_sequences` 方法，使其能够处理 `meta_info` 中的多模态数据：

```python
def generate_sequences(self, batch: DataProto) -> DataProto:
    """Generate sequences with multimodal support."""
    # Extract multimodal data from meta_info
    pixel_values = batch.meta_info.get('pixel_values', None)
    image_grid_thw = batch.meta_info.get('image_grid_thw', None)
    
    # Prepare multi_modal_data for vLLM
    multi_modal_data = None
    if pixel_values is not None:
        multi_modal_data = {
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
        }
    
    # Call vLLM generate with multimodal data
    outputs = self.llm.generate(
        prompts=None,
        prompt_token_ids=input_ids,
        sampling_params=sampling_params,
        multi_modal_data=multi_modal_data,  # NEW: Pass multimodal data
    )
```

### 4. 参考 single_turn_agent_loop.py 的模式

`single_turn_agent_loop.py` 已经有正确的多模态处理模式：

```python
# From single_turn_agent_loop.py lines 50-75
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
    
    # Store the multi_modal_inputs (pixel_values, image_grid_thw, etc.)
    computed_multi_modal_inputs = dict(model_inputs)
```

## Data Models

### MultiModalInputs 数据结构

```python
@dataclass
class MultiModalInputs:
    """Per-sample multimodal inputs."""
    pixel_values: torch.Tensor  # Shape: [num_images, channels, height, width]
    image_grid_thw: torch.Tensor  # Shape: [num_images, 3] for Qwen2-VL
```

### BatchedMultiModalInputs 数据结构

```python
@dataclass
class BatchedMultiModalInputs:
    """Batched multimodal inputs for generation."""
    pixel_values: torch.Tensor  # Shape: [total_images, channels, height, width]
    image_grid_thw: torch.Tensor  # Shape: [total_images, 3]
    # Mapping from sample index to image indices
    sample_to_image_indices: List[List[int]]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Pixel Values Data Flow

*For any* batch containing multi_modal_inputs with pixel_values, when passed through _generate_with_gpu_padding and generate_sequences, the pixel_values SHALL be accessible to the model's generate method.

**Validates: Requirements 1.1, 1.2**

### Property 2: Position Encoding with Image Grid

*For any* multi_modal_inputs containing image_grid_thw, the system SHALL compute position_ids using get_rope_index that correctly accounts for the image token positions.

**Validates: Requirements 1.3**

### Property 3: Multi-Modal Inputs Format Consistency

*For any* multi_modal_inputs created by the processor, the format SHALL be consumable by both LLMGenerationManager and the rollout workers without transformation errors.

**Validates: Requirements 3.1**

### Property 4: Tensor Concatenation for Multi-Turn

*For any* sequence of multi_modal_inputs across multiple turns, concatenating pixel_values and image_grid_thw SHALL produce tensors with correct shapes where the first dimension equals the total number of images.

**Validates: Requirements 3.2, 3.3**

### Property 5: Variable Image Count Handling

*For any* batch where different samples have different numbers of images (including zero), the system SHALL correctly batch and unbatch the multimodal data without index errors.

**Validates: Requirements 3.4**

## Error Handling

1. **Missing pixel_values**: 如果 multi_modal_inputs 存在但 pixel_values 为空，记录警告并继续文本生成
2. **Shape 不匹配**: 如果 pixel_values 和 image_grid_thw 的第一维不匹配，抛出明确的错误
3. **设备不匹配**: 确保 pixel_values 在正确的设备上（与 input_ids 相同）
4. **内存不足**: 对于大量图像，考虑分批处理

## Testing Strategy

### 单元测试

1. 测试 `_batch_multimodal_inputs()` 方法
   - 正常情况：多个样本各有不同数量的图像
   - 边界情况：空 multi_modal_inputs
   - 边界情况：部分样本没有图像

2. 测试 `_generate_with_gpu_padding()` 方法
   - 验证 pixel_values 被正确传递到 meta_info
   - 验证 padding 不影响多模态数据

### 属性测试

使用 Python 的 `hypothesis` 库进行属性测试：

1. **Property 1**: 生成随机 pixel_values，验证数据流
2. **Property 4**: 生成多轮 multi_modal_inputs，验证拼接正确性
3. **Property 5**: 生成不同图像数量的批次，验证处理正确性

### 集成测试

1. 端到端测试：运行完整的 VRAG 训练循环，验证模型能够"看到"图像
2. 回归测试：验证纯文本生成仍然正常工作

### 测试配置

- 每个属性测试运行至少 100 次迭代
- 使用 pytest 作为测试框架
- 使用 hypothesis 进行属性测试
- 标签格式: **Feature: multimodal-data-passing-fix, Property {number}: {property_text}**

