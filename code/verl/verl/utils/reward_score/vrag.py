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
VRAG reward scoring functions for vidoseek, mmlongdoc, and other RAG datasets.
"""

import re
import json


def get_answer_from_predict_str(text):
    """Extract answer from JSON format.
    
    Expected format:
    {"think": "...", "action": "answer", "arguments": {}, "answer": "..."}
    """
    return extract_answer_from_json(text)


def extract_answer_from_json(text):
    """Extract answer from JSON format response.
    
    Expected format:
    {"think": "...", "action": "answer", "arguments": {}, "answer": "the answer"}
    """
    try:
        # 找到所有 JSON 对象
        json_objects = extract_all_json_objects(text)
        
        # 从后往前找，找到最后一个 action="answer" 的 JSON
        for json_obj in reversed(json_objects):
            if json_obj.get('action') == 'answer' and 'answer' in json_obj:
                answer = json_obj.get('answer')
                if answer is not None:
                    return str(answer)
        
        return None
    except Exception:
        return None


def extract_all_json_objects(text):
    """Extract all valid JSON objects from text."""
    json_objects = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # 使用括号匹配找到对应的 }
            brace_count = 0
            end_idx = -1
            for j in range(i, len(text)):
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = j
                        break
            
            if end_idx != -1:
                json_str = text[i:end_idx + 1]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        json_objects.append(parsed)
                except json.JSONDecodeError:
                    pass
                i = end_idx + 1
            else:
                i += 1
        else:
            i += 1
    
    return json_objects


def extract_json_action(text):
    """Extract action from JSON format response.
    
    Expected format:
    {"think": "...", "action": "search|crop|ocr|answer", "arguments": {...}, "answer": null|"..."}
    
    Returns:
        Tuple of (action, content) or (None, None) if not valid
    """
    VALID_ACTIONS = {'search', 'crop', 'ocr', 'answer'}
    
    try:
        # 找到第一个 { 字符
        start_idx = text.find('{')
        if start_idx == -1:
            return None, None
        
        # 使用括号匹配找到对应的 }
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            return None, None
        
        json_str = text[start_idx:end_idx + 1]
        parsed = json.loads(json_str)
        
        # 验证必需字段
        required_fields = ['think', 'action', 'arguments', 'answer']
        if not all(key in parsed for key in required_fields):
            return None, None
        
        action = parsed.get('action', '')
        if action not in VALID_ACTIONS:
            return None, None
        
        # 根据 action 类型提取 content
        if action == 'search':
            content = parsed.get('arguments', {}).get('query', '')
        elif action == 'crop':
            content = parsed.get('arguments', {})
        elif action == 'ocr':
            content = parsed.get('arguments', {})
        elif action == 'answer':
            content = parsed.get('answer', '')
        else:
            return None, None
        
        return action, content
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, None


def calculate_anls(gold_labels, prediction, threshold=0.7):
    """Calculate Average Normalized Levenshtein Similarity (ANLS)."""
    try:
        from Levenshtein import distance as levenshtein_distance
    except ImportError:
        # Fallback to simple exact match if Levenshtein not available
        return 1.0 if prediction in gold_labels else 0.0
    
    max_scores = []
    for gold_label in gold_labels:
        ld = levenshtein_distance(prediction, gold_label)
        max_len = max(len(prediction), len(gold_label))
        if max_len == 0:
            nld = 0.0
        else:
            nld = ld / max_len
        if nld < threshold:
            score = 1 - nld
        else:
            score = 0.0
        max_scores.append(score)
    return max(max_scores) if max_scores else 0.0


def remove_text_between_tags(text):
    """Remove text between <|im_start|> and <|im_end|> tags."""
    pattern = r'<\|im_start\|>user.*?<\|im_end\|>'
    result = re.sub(pattern, '', text)
    return result


def compute_format_reward_only(predict_str: str, ground_truth: str = None, extra_info=None) -> float:
    """
    Compute format reward - checks if the response has proper JSON action format.
    
    Expected JSON format:
    - {"think": "...", "action": "search", "arguments": {"query": "..."}, "answer": null}
    - {"think": "...", "action": "crop", "arguments": {"image_id": "...", "region": [...]}, "answer": null}
    - {"think": "...", "action": "ocr", "arguments": {"image_id": "...", "region": [...]}, "answer": null}
    - {"think": "...", "action": "answer", "arguments": {}, "answer": "..."}
    
    Returns 1.0 if the response has both:
    1. At least one tool call (search, crop, ocr)
    2. A final answer
    """
    predict_str = remove_text_between_tags(predict_str)
    
    json_objects = extract_all_json_objects(predict_str)
    
    has_tool_call = False
    has_answer = False
    
    for json_obj in json_objects:
        action = json_obj.get('action', '')
        # 检查必需字段
        if all(key in json_obj for key in ['think', 'action', 'arguments', 'answer']):
            if action in ('search', 'crop', 'ocr'):
                has_tool_call = True
            elif action == 'answer' and json_obj.get('answer') is not None:
                has_answer = True
    
    return 1.0 if has_answer and has_tool_call else 0.0


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute score for VRAG datasets (vidoseek, mmlongdoc, etc.).
    
    First checks format (must have tool calls and answer),
    then computes ANLS score for the answer.
    """
    predict_str = remove_text_between_tags(predict_str)
    format_reward_value = compute_format_reward_only(predict_str, ground_truth, extra_info)

    if format_reward_value == 1.0:
        answer = get_answer_from_predict_str(predict_str)
        if answer is None:
            return 0.0
        anls_score = calculate_anls([ground_truth], answer, 0.5)
        return anls_score
    else:
        return 0.0
