# Requirements Document

## Introduction

本文档定义了修复 VRAG 训练中多模态数据（图像 pixel_values）无法正确传递给模型的问题。当前问题是：图像 token 存在于上下文窗口中，但实际的 pixel_values 数据没有被传递给模型的 generate 方法，导致模型输出 "No evidence image provided"。

## Glossary

- **VRAG_Agent**: Visual RAG Agent，视觉检索增强生成代理
- **multi_modal_inputs**: 包含 pixel_values 和 image_grid_thw 的多模态输入数据
- **pixel_values**: 图像的像素值张量，模型需要这些数据来"看到"图像
- **image_grid_thw**: 图像网格的时间/高度/宽度信息，用于 Qwen2-VL 的 mrope 位置编码
- **non_tensor_batch**: DataProto 中存储非张量数据的字典
- **actor_rollout_wg**: Actor Rollout Worker Group，负责模型生成
- **async_rollout_manager**: 异步 rollout 管理器，支持多模态处理
- **LLMGenerationManager**: 管理 VRAG 多轮生成循环的类
- **vLLM**: 高性能 LLM 推理引擎

## Requirements

### Requirement 1: 多模态数据传递

**User Story:** As a developer, I want the pixel_values to be passed to the model during generation, so that the model can actually "see" the retrieved images.

#### Acceptance Criteria

1. WHEN generating model responses, THE LLMGenerationManager SHALL pass pixel_values from multi_modal_inputs to the rollout worker
2. WHEN multi_modal_inputs contains pixel_values, THE rollout worker SHALL include them in the model.generate() call
3. WHEN multi_modal_inputs contains image_grid_thw, THE rollout worker SHALL use it for proper position encoding
4. IF multi_modal_inputs is empty or None, THEN THE rollout worker SHALL proceed with text-only generation

### Requirement 2: 与 async_rollout_manager 集成

**User Story:** As a developer, I want to leverage the existing async_rollout_manager's multimodal handling, so that I can reuse proven code.

#### Acceptance Criteria

1. WHEN async_rollout_mode is enabled, THE LLMGenerationManager SHALL use async_rollout_manager for generation
2. THE async_rollout_manager SHALL properly handle multi_modal_data passed in the batch
3. WHEN using async_rollout_manager, THE system SHALL follow the pattern in single_turn_agent_loop.py for multimodal processing
4. THE integration SHALL maintain compatibility with the existing VRAG training loop

### Requirement 3: 数据格式兼容性

**User Story:** As a developer, I want the multimodal data format to be compatible between generation.py and the rollout workers, so that data flows correctly.

#### Acceptance Criteria

1. THE multi_modal_inputs format SHALL be consistent between LLMGenerationManager and rollout workers
2. WHEN concatenating multi_modal_data across turns, THE system SHALL properly merge pixel_values tensors
3. WHEN concatenating multi_modal_data across turns, THE system SHALL properly merge image_grid_thw tensors
4. THE system SHALL handle the case where different samples have different numbers of images

### Requirement 4: 调试和验证

**User Story:** As a developer, I want to verify that pixel_values are being passed correctly, so that I can debug multimodal issues.

#### Acceptance Criteria

1. WHEN debug mode is enabled, THE system SHALL log pixel_values shape before generation
2. WHEN debug mode is enabled, THE system SHALL log image_grid_thw values before generation
3. THE system SHALL provide clear error messages when multimodal data is missing or malformed
4. THE system SHALL log whether the model received multimodal inputs for each generation step

### Requirement 5: 向后兼容性

**User Story:** As a developer, I want the fix to maintain backward compatibility, so that existing text-only workflows continue to work.

#### Acceptance Criteria

1. WHEN no multi_modal_inputs are present, THE system SHALL fall back to text-only generation
2. THE fix SHALL not break existing VRAG training configurations
3. THE fix SHALL work with both async and non-async rollout modes
4. THE fix SHALL maintain compatibility with the existing reward computation pipeline

</content>
</invoke>