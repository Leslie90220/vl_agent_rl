# Implementation Plan: JSON Output Format Adaptation

## Overview

本实现计划将 VRAG Agent 的模型输出解析从 XML 标签格式适配到新的 JSON 格式，同时保持向后兼容性。

## Tasks

- [x] 1. 修改 generation.py 中的 JSON 解析逻辑
  - [x] 1.1 更新 extract_json_response() 函数以支持新的 JSON 格式
    - 修改 JSON 字段验证逻辑，检查 think, action, arguments, answer 字段
    - 添加对 action 字段值的验证（search, crop, ocr, answer）
    - _Requirements: 1.1, 2.1, 2.2_

  - [x] 1.2 更新 _postprocess_responses() 函数
    - 确保 JSON 响应被正确检测和保留
    - 保持对 XML 标签格式的回退支持
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 2. 修改 postprocess_predictions() 函数
  - [x] 2.1 添加新的 JSON 格式解析逻辑
    - 解析 action 字段确定动作类型
    - 根据 action 类型提取对应的 arguments 或 answer
    - 支持 search, crop, ocr, answer 四种动作
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 2.2 保持 XML 标签格式的向后兼容
    - 当 JSON 解析失败时回退到 XML 解析
    - 保持对 `<tools_call>`, `<search>`, `<answer>` 标签的支持
    - _Requirements: 1.6, 4.1, 4.2, 4.3_

- [x] 3. 更新错误消息为 JSON 格式
  - [x] 3.1 更新 execute_predictions() 中的无效动作错误消息
    - 提供 JSON 格式示例
    - 包含所有动作类型的示例
    - _Requirements: 3.1, 3.4_

  - [x] 3.2 更新 crop 动作失败的错误消息
    - 提供 crop 动作的 JSON 格式示例
    - _Requirements: 3.2_

  - [x] 3.3 更新 OCR 动作失败的错误消息
    - 提供 OCR 动作的 JSON 格式示例
    - _Requirements: 3.3_

- [x] 4. 更新强制搜索响应格式
  - [x] 4.1 修改第一轮强制搜索的响应格式为 JSON
    - 使用 JSON 格式构造强制搜索响应
    - 包含 think, action, arguments, answer 字段
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 5. Checkpoint - 验证基本功能
  - 确保所有修改编译通过
  - 手动测试 JSON 格式解析
  - 验证向后兼容性

- [x] 6. 编写测试
  - [x] 6.1 编写 JSON 解析的单元测试
    - 测试有效 JSON 输入
    - 测试无效 JSON 输入
    - 测试缺少字段的 JSON
    - _Requirements: 1.1, 1.6_

  - [x] 6.2 编写属性测试
    - **Property 1: JSON 解析正确性**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

- [x] 7. Final checkpoint - 确保所有测试通过
  - 运行所有测试 ✅ (21 unit tests + 7 property tests all passing)
  - 验证功能完整性 ✅

## Notes

- 主要修改集中在 `code/vrag_agent/generation.py` 文件
- 保持向后兼容性是关键，确保旧格式仍然可用
- JSON 格式优先级高于 XML 格式
