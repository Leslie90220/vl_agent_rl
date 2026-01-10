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


def get_answer_from_predict_str(text):
    """Extract answer from <answer>...</answer> tags."""
    end_tag = '</answer>'
    start_tag = '<answer>'
    
    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None
    
    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None
    
    start_pos += len(start_tag)
    return text[start_pos:end_pos]


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
    Compute format reward - checks if the response has proper action tags.
    
    Valid tool call formats:
    - <search>query</search> - legacy search format
    - <tools_call>{"type":"search",...}</tools_call> - new search format
    - <tools_call>{"type":"crop",...}</tools_call> - crop tool
    - <tools_call>{"type":"ocr",...}</tools_call> - ocr tool
    - <answer>...</answer> - final answer
    
    Returns 1.0 if the response has both:
    1. At least one tool call (search, crop, ocr via either format)
    2. A final answer
    """
    predict_str = remove_text_between_tags(predict_str)
    
    # Check for answer tag
    answer_pattern = re.compile(r'<answer>.*</answer>', re.DOTALL)
    answer_match = re.search(answer_pattern, predict_str)
    
    # Check for tool calls - either legacy <search> or new <tools_call> format
    search_pattern = re.compile(r'<search>.*</search>', re.DOTALL)
    tools_call_pattern = re.compile(r'<tools_call>.*</tools_call>', re.DOTALL)
    
    search_match = re.search(search_pattern, predict_str)
    tools_call_match = re.search(tools_call_pattern, predict_str)
    
    # Valid if has answer AND at least one tool call (either format)
    has_tool_call = search_match or tools_call_match
    
    return 1.0 if answer_match and has_tool_call else 0.0


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute score for VRAG datasets (vidoseek, mmlongdoc, etc.).
    
    First checks format (must have <answer> and <search> tags),
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
