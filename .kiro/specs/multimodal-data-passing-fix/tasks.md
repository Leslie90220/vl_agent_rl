# Implementation Plan: Multimodal Data Passing Fix

## Overview

本实现计划修复 VRAG 训练中多模态数据（pixel_values）无法正确传递给模型的问题。

## 问题根本原因

在 vLLM async 模式下，`single_turn_agent_loop.py` 和 `tool_agent_loop.py` 中存在以下问题：

1. 使用 `processor` 预处理图片，生成包含展开图片 token 的 `prompt_ids`
2. 同时保存 `computed_multi_modal_inputs`（包含 `pixel_values`, `image_grid_thw`）
3. **关键问题**：当 `computed_multi_modal_inputs` 被设置后，`image_data_for_vllm = None`
4. vLLM 的 `generate` 方法收到 `image_data=None`，所以模型无法"看到"图片内容
5. 模型输出 "No evidence image provided"

## 解决方案

使用 `processor()` 生成带展开图片 token 的 `prompt_ids`，然后让 vLLM 处理：

1. 使用 `processor()` 生成 `prompt_ids`（这会展开图片 token）
2. 将原始 `image_data` 传递给 vLLM
3. vLLM 的 `_qwen2_5_vl_dedup_image_tokens` 会将连续的 `<|image_pad|>` token 折叠为单个
4. vLLM 会根据 `image_data` 重新展开 token 并处理图片

**为什么之前的 tokenizer.encode() 方案不行**：
- `tokenizer.encode()` 可能不会正确编码 `<|image_pad|>` 等特殊 token
- vLLM 需要在 `prompt_ids` 中找到 `image_token_id` 才能知道在哪里插入图片特征
- 如果找不到，vLLM 会抛出：`"Failed to apply prompt replacement for mm_items['image'][0]"`

## Tasks

- [x] 1. 分析问题根本原因
  - [x] 1.1 分析 single_turn_agent_loop.py 的多模态处理逻辑
  - [x] 1.2 分析 vllm_async_server.py 的 generate 方法
  - [x] 1.3 理解 _qwen2_5_vl_dedup_image_tokens 的作用

- [x] 2. 修复 single_turn_agent_loop.py
  - [x] 2.1 使用 processor() 生成 prompt_ids（带展开的图片 token）
  - [x] 2.2 始终将 image_data 传递给 vLLM
  - [x] 2.3 vLLM 的 _qwen2_5_vl_dedup_image_tokens 会折叠 token

- [x] 3. 修复 tool_agent_loop.py
  - [x] 3.1 修改 _handle_pending_state 方法
  - [x] 3.2 修改 _handle_generating_state 方法
  - [x] 3.3 移除 _image_tokens_expanded 标志

- [x] 4. 修复 partial_tool_agent_loop.py
  - [x] 4.1 修改 _handle_generating_state_partial 方法
  - [x] 4.2 在 _init_agent_data() 中传递 raw_prompt_ids 到 agent_data.extra_fields

- [ ] 5. 验证修复
  - [ ] 5.1 运行 VRAG 训练
  - [ ] 5.2 验证模型能够"看到"图像
  - [ ] 5.3 验证模型不再输出 "No evidence image provided"

## 修改的文件

- `code/verl/verl/experimental/agent_loop/single_turn_agent_loop.py`
- `code/verl/verl/experimental/agent_loop/tool_agent_loop.py`
- `code/verl/recipe/fully_async_policy/agent_loop/partial_tool_agent_loop.py`

## 最新修复 (2026-01-10)

**问题 1**：`tool_agent_loop.py` 和 `partial_tool_agent_loop.py` 中的 `_handle_pending_state` 方法检查 `agent_data.extra_fields["raw_prompt_ids"]`，但 `raw_prompt_ids` 从未被存储到 `agent_data.extra_fields` 中。

**修复 1**：
1. `tool_agent_loop.py` - 在 `run()` 方法中，创建 `AgentData` 后，添加：
   ```python
   if "raw_prompt_ids" in kwargs and kwargs["raw_prompt_ids"] is not None:
       agent_data.extra_fields["raw_prompt_ids"] = kwargs["raw_prompt_ids"]
   ```

2. `partial_tool_agent_loop.py` - 在 `_init_agent_data()` 方法中做同样的修复

**问题 2**：`_agent_loop_postprocess` 中的 `get_rope_index` 函数期望 `input_ids` 中的图片 token 数量与 `image_grid_thw` 匹配，但当使用折叠的 `raw_prompt_ids` 时，图片 token 是折叠的（每张图片 1 个 token），导致 shape mismatch 错误。

**修复 2**：
1. `single_turn_agent_loop.py` - 添加 `using_collapsed_prompt_ids` 标志到 `extra_fields`
2. `agent_loop.py` - 在 `_agent_loop_postprocess` 中检测 `using_collapsed_prompt_ids` 标志，当使用折叠的 `prompt_ids` 时，跳过 `get_rope_index` 计算，使用简单的顺序 `position_ids`

## Notes

- 之前的修改（HF rollout 相关）仍然保留，但不是 vLLM async 模式的解决方案
- vLLM 的 `_qwen2_5_vl_dedup_image_tokens` 函数会将展开的图片 token 折叠回单个 token
- vLLM 内部会根据 `image_data` 重新处理图片并展开 token
- **关键修复**：使用 `raw_prompt_ids`（预计算的 token IDs，已折叠图片 token）而不是重新 tokenize `raw_prompt`
  - `raw_prompt_ids` 由 `generation.py` 中的 `_raw_prompt_ids()` 方法计算
  - 使用 `replace_consecutive_elements()` 折叠连续的图片 token
  - 确保图片占位符数量与 `multi_modal_data` 中的图片数量匹配
