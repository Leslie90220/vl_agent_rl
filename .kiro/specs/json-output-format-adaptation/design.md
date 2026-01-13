# Design Document: JSON Output Format Adaptation

## Overview

本设计文档描述了如何将 VRAG Agent 的模型输出解析从旧的 XML 标签格式适配到新的 JSON 格式。新格式更加结构化，包含 `think`, `action`, `arguments`, `answer` 四个必需字段。

## Architecture

系统采用分层解析策略：
1. 首先尝试解析 JSON 格式
2. 如果 JSON 解析失败，回退到 XML 标签格式
3. 保持向后兼容性

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Output                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              _postprocess_responses()                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. Try extract_json_response()                      │    │
│  │  2. If failed, try extract_tags() (legacy XML)       │    │
│  │  3. Append EOS token                                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              postprocess_predictions()                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. Try parse JSON format                            │    │
│  │  2. Extract action and arguments                     │    │
│  │  3. If failed, try legacy XML tags                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              execute_predictions()                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Execute action based on parsed type                 │    │
│  │  Return JSON format error messages if invalid        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. JSON 格式定义

新的 JSON 输出格式：

```json
{
  "think": "string, brief plan (1-3 sentences)",
  "action": "search" | "crop" | "ocr" | "answer",
  "arguments": {
    // For search: {"query": "search query"}
    // For crop: {"image_id": "image_01", "region": [x1, y1, x2, y2]}
    // For ocr: {"image_id": "image_01", "region": [x1, y1, x2, y2]}
    // For answer: {}
  },
  "answer": null | "string"
}
```

### 2. extract_json_response() 函数

```python
def extract_json_response(text: str) -> Optional[str]:
    """
    从文本中提取有效的 JSON 响应。
    
    Args:
        text: 模型输出的原始文本
        
    Returns:
        提取的 JSON 字符串，如果无效则返回 None
    """
    # 1. 找到第一个 { 字符
    # 2. 使用括号匹配找到对应的 }
    # 3. 尝试解析 JSON
    # 4. 验证必需字段: think, action, arguments, answer
    # 5. 返回有效的 JSON 字符串或 None
```

### 3. postprocess_predictions() 修改

```python
def postprocess_predictions(self, predictions: List[str]) -> Tuple[List[str], List[Any]]:
    """
    解析模型预测，提取动作和内容。
    
    解析顺序：
    1. 尝试解析 JSON 格式
    2. 如果失败，回退到 XML 标签格式
    
    JSON 格式动作映射：
    - action="search" -> 返回 ('search', arguments.query)
    - action="crop" -> 返回 ('crop', arguments)
    - action="ocr" -> 返回 ('ocr', arguments)
    - action="answer" -> 返回 ('answer', answer)
    """
```

### 4. 错误消息模板

```python
INVALID_ACTION_ERROR = '''
<|im_start|>user
Your previous action is invalid. You MUST output a valid JSON object with this exact format:

{
  "think": "your brief reasoning (1-3 sentences)",
  "action": "search" | "crop" | "ocr" | "answer",
  "arguments": {},
  "answer": null
}

Examples:
- For search: {"think": "I need to find...", "action": "search", "arguments": {"query": "your search query"}, "answer": null}
- For answer: {"think": "Based on the information...", "action": "answer", "arguments": {}, "answer": "your final answer"}

Please try again.
<|im_end|>
<|im_start|>assistant
'''

CROP_ERROR = '''
<|im_start|>user
Your crop action failed. Please use the correct JSON format:

{
  "think": "I need to zoom in on...",
  "action": "crop",
  "arguments": {"image_id": "image_01", "region": [x1, y1, x2, y2]},
  "answer": null
}

Please try again.
<|im_end|>
<|im_start|>assistant
'''

OCR_ERROR = '''
<|im_start|>user
Your OCR action failed. Please use the correct JSON format:

{
  "think": "I need to extract text from...",
  "action": "ocr",
  "arguments": {"image_id": "image_01", "region": [x1, y1, x2, y2]},
  "answer": null
}

Please try again.
<|im_end|>
<|im_start|>assistant
'''
```

### 5. 强制搜索响应格式

```python
FORCED_SEARCH_TEMPLATE = '''{{
  "think": "I need to search for relevant information to answer this question.",
  "action": "search",
  "arguments": {{"query": "{question}"}},
  "answer": null
}}'''
```

## Data Models

### ParsedAction 数据结构

```python
@dataclass
class ParsedAction:
    action_type: str  # 'search', 'crop', 'ocr', 'answer', None
    content: Any      # query string, arguments dict, or answer string
    raw_json: Optional[str]  # 原始 JSON 字符串（如果是 JSON 格式）
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: JSON 解析正确性

*For any* valid JSON response containing all required fields (think, action, arguments, answer), the parser SHALL correctly extract the action type and corresponding content.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

### Property 2: JSON 检测和保留

*For any* model output containing a valid JSON object, the postprocessor SHALL detect and preserve the complete JSON structure with EOS token appended.

**Validates: Requirements 2.1, 2.2, 2.4**

### Property 3: 向后兼容性

*For any* model output in legacy XML tag format (containing `<search>`, `<answer>`, or `<tools_call>` tags), the parser SHALL correctly extract the action and content when JSON parsing fails.

**Validates: Requirements 1.6, 2.3, 4.1, 4.2, 4.3**

### Property 4: JSON 解析优先级

*For any* model output, the parser SHALL first attempt JSON parsing before falling back to XML tag parsing.

**Validates: Requirements 4.2, 4.3**

## Error Handling

1. **JSON 解析失败**: 回退到 XML 标签解析
2. **两种格式都失败**: 返回 None 动作，触发错误提示
3. **动作执行失败**: 返回包含 JSON 格式示例的错误消息
4. **缺少必需字段**: 视为无效 JSON，回退到 XML 解析

## Testing Strategy

### 单元测试

1. 测试 `extract_json_response()` 函数
   - 有效 JSON 输入
   - 无效 JSON 输入
   - 缺少必需字段的 JSON
   - 嵌入在其他文本中的 JSON

2. 测试 `postprocess_predictions()` 函数
   - 各种 action 类型的 JSON 输入
   - XML 标签格式输入
   - 混合格式输入

### 属性测试

使用 Python 的 `hypothesis` 库进行属性测试：

1. **Property 1**: 生成随机有效 JSON，验证解析正确性
2. **Property 2**: 生成包含 JSON 的文本，验证检测和保留
3. **Property 3**: 生成 XML 格式响应，验证向后兼容
4. **Property 4**: 验证解析顺序

### 测试配置

- 每个属性测试运行至少 100 次迭代
- 使用 pytest 作为测试框架
- 使用 hypothesis 进行属性测试
