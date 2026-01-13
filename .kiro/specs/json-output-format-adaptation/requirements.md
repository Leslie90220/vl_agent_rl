# Requirements Document

## Introduction

本文档定义了将 VRAG Agent 的模型输出格式从旧的 XML 标签格式适配到新的 JSON 格式的需求。新的 JSON 格式更加结构化，便于解析和验证。

## Glossary

- **VRAG_Agent**: Visual RAG Agent，视觉检索增强生成代理
- **JSON_Output**: 模型输出的 JSON 格式响应
- **Action**: 模型执行的动作类型（search, crop, ocr, answer）
- **Generation_Manager**: LLMGenerationManager 类，负责管理模型生成和解析
- **Response_Parser**: 解析模型输出的组件

## Requirements

### Requirement 1: JSON 输出格式解析

**User Story:** As a developer, I want the system to parse the new JSON output format, so that the model can use a more structured response format.

#### Acceptance Criteria

1. WHEN the model outputs a valid JSON response, THE Response_Parser SHALL extract the action, arguments, and answer fields correctly
2. WHEN the JSON contains `"action": "search"`, THE Response_Parser SHALL extract the query from `arguments.query`
3. WHEN the JSON contains `"action": "crop"`, THE Response_Parser SHALL extract image_id and region from arguments
4. WHEN the JSON contains `"action": "ocr"`, THE Response_Parser SHALL extract image_id and region from arguments
5. WHEN the JSON contains `"action": "answer"`, THE Response_Parser SHALL extract the answer from the `answer` field
6. IF the JSON is malformed or missing required fields, THEN THE Response_Parser SHALL fall back to legacy XML tag parsing

### Requirement 2: 响应后处理适配

**User Story:** As a developer, I want the response postprocessing to handle JSON format, so that valid JSON responses are preserved correctly.

#### Acceptance Criteria

1. WHEN processing model responses, THE Generation_Manager SHALL detect and extract valid JSON objects
2. WHEN a valid JSON response is found, THE Generation_Manager SHALL preserve the complete JSON structure
3. WHEN no valid JSON is found, THE Generation_Manager SHALL fall back to extracting XML tags
4. THE Generation_Manager SHALL append EOS token to processed responses

### Requirement 3: 错误提示更新

**User Story:** As a developer, I want error messages to guide the model to use the new JSON format, so that invalid actions can be corrected.

#### Acceptance Criteria

1. WHEN an invalid action is detected, THE VRAG_Agent SHALL return an error message with JSON format examples
2. WHEN a crop action fails, THE VRAG_Agent SHALL provide JSON format guidance for crop actions
3. WHEN an OCR action fails, THE VRAG_Agent SHALL provide JSON format guidance for OCR actions
4. THE error messages SHALL include the exact JSON schema expected

### Requirement 4: 向后兼容性

**User Story:** As a developer, I want the system to maintain backward compatibility with XML tag format, so that existing models can still work.

#### Acceptance Criteria

1. WHEN the model outputs XML tag format, THE Response_Parser SHALL still parse it correctly
2. THE Response_Parser SHALL first attempt JSON parsing, then fall back to XML parsing
3. WHILE supporting both formats, THE VRAG_Agent SHALL prioritize JSON format detection

### Requirement 5: 第一轮强制搜索适配

**User Story:** As a developer, I want the first-turn forced search to use JSON format, so that it matches the new output format.

#### Acceptance Criteria

1. WHEN forcing a search on the first turn, THE Generation_Manager SHALL generate a JSON format response
2. THE forced search response SHALL include think, action, arguments, and answer fields
3. THE forced search response SHALL set action to "search" and answer to null
