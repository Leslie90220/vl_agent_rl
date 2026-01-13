"""
Property-based tests for JSON output format parsing.

**Feature: json-output-format-adaptation, Property 1: JSON 解析正确性**
**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

For any valid JSON response containing all required fields (think, action, arguments, answer),
the parser SHALL correctly extract the action type and corresponding content.
"""

import pytest
import json
import re
from typing import Tuple, Any, Optional
from hypothesis import given, strategies as st, settings, assume


# Standalone parsing functions (same as in test_json_parsing.py)
VALID_ACTIONS = {'search', 'crop', 'ocr', 'answer'}


def extract_json_response(text: str) -> Optional[str]:
    """Extract and validate JSON response format."""
    try:
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
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
            return None
        
        json_str = text[start_idx:end_idx + 1]
        parsed = json.loads(json_str)
        
        required_fields = ['think', 'action', 'arguments', 'answer']
        if not all(key in parsed for key in required_fields):
            return None
        
        action = parsed.get('action', '')
        if action not in VALID_ACTIONS:
            return None
        
        if not isinstance(parsed.get('arguments'), dict):
            return None
        
        return json_str
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    return None


def parse_new_json_format(text: str) -> Tuple[Optional[str], Any]:
    """Parse the new JSON format and extract action and content."""
    try:
        start_idx = text.find('{')
        if start_idx == -1:
            return None, None
        
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
        
        required_fields = ['think', 'action', 'arguments', 'answer']
        if not all(key in parsed for key in required_fields):
            return None, None
        
        action = parsed.get('action', '')
        if action not in VALID_ACTIONS:
            return None, None
        
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


def extract_tags(text: str) -> str:
    """Legacy: Extract XML tag format responses."""
    pattern = r"<(answer|search|think|bbox|tools_call)>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    result = "\n".join([f"<{tag}>{content}</{tag}>" for tag, content in matches])
    return result


# Strategies for generating valid JSON components
@st.composite
def valid_think_strategy(draw):
    """Generate valid think strings."""
    return draw(st.text(min_size=1, max_size=200).filter(lambda x: '"' not in x and '\\' not in x and '{' not in x and '}' not in x))


@st.composite
def valid_query_strategy(draw):
    """Generate valid search queries."""
    return draw(st.text(min_size=1, max_size=100).filter(lambda x: '"' not in x and '\\' not in x and '{' not in x and '}' not in x))


@st.composite
def valid_image_id_strategy(draw):
    """Generate valid image IDs."""
    num = draw(st.integers(min_value=1, max_value=99))
    return f"image_{num:02d}"


@st.composite
def valid_region_strategy(draw):
    """Generate valid region coordinates."""
    x1 = draw(st.integers(min_value=0, max_value=500))
    y1 = draw(st.integers(min_value=0, max_value=500))
    x2 = draw(st.integers(min_value=x1 + 1, max_value=1000))
    y2 = draw(st.integers(min_value=y1 + 1, max_value=1000))
    return [x1, y1, x2, y2]


@st.composite
def valid_answer_strategy(draw):
    """Generate valid answer strings."""
    return draw(st.text(min_size=1, max_size=200).filter(lambda x: '"' not in x and '\\' not in x and '{' not in x and '}' not in x))


@st.composite
def valid_search_json_strategy(draw):
    """Generate valid search action JSON."""
    think = draw(valid_think_strategy())
    query = draw(valid_query_strategy())
    return {
        "think": think,
        "action": "search",
        "arguments": {"query": query},
        "answer": None
    }


@st.composite
def valid_crop_json_strategy(draw):
    """Generate valid crop action JSON."""
    think = draw(valid_think_strategy())
    image_id = draw(valid_image_id_strategy())
    region = draw(valid_region_strategy())
    return {
        "think": think,
        "action": "crop",
        "arguments": {"image_id": image_id, "region": region},
        "answer": None
    }


@st.composite
def valid_ocr_json_strategy(draw):
    """Generate valid OCR action JSON."""
    think = draw(valid_think_strategy())
    image_id = draw(valid_image_id_strategy())
    region = draw(valid_region_strategy())
    return {
        "think": think,
        "action": "ocr",
        "arguments": {"image_id": image_id, "region": region},
        "answer": None
    }


@st.composite
def valid_answer_json_strategy(draw):
    """Generate valid answer action JSON."""
    think = draw(valid_think_strategy())
    answer = draw(valid_answer_strategy())
    return {
        "think": think,
        "action": "answer",
        "arguments": {},
        "answer": answer
    }


@st.composite
def any_valid_json_strategy(draw):
    """Generate any valid JSON action."""
    strategy = draw(st.sampled_from([
        valid_search_json_strategy(),
        valid_crop_json_strategy(),
        valid_ocr_json_strategy(),
        valid_answer_json_strategy()
    ]))
    return draw(strategy)


class TestJsonParsingProperty:
    """
    Property-based tests for JSON parsing correctness.
    
    **Feature: json-output-format-adaptation, Property 1: JSON 解析正确性**
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
    """
    
    @given(json_obj=valid_search_json_strategy())
    @settings(max_examples=100)
    def test_search_json_parsing_property(self, json_obj):
        """
        Property: For any valid search JSON, parser extracts action='search' and query correctly.
        **Validates: Requirements 1.1, 1.2**
        """
        json_str = json.dumps(json_obj)
        
        # Test extract_json_response
        result = extract_json_response(json_str)
        assert result is not None, f"Failed to extract JSON: {json_str}"
        
        # Test parse_new_json_format
        action, content = parse_new_json_format(json_str)
        assert action == 'search', f"Expected action='search', got {action}"
        assert content == json_obj['arguments']['query'], f"Query mismatch: {content} != {json_obj['arguments']['query']}"
    
    @given(json_obj=valid_crop_json_strategy())
    @settings(max_examples=100)
    def test_crop_json_parsing_property(self, json_obj):
        """
        Property: For any valid crop JSON, parser extracts action='crop' and arguments correctly.
        **Validates: Requirements 1.1, 1.3**
        """
        json_str = json.dumps(json_obj)
        
        result = extract_json_response(json_str)
        assert result is not None
        
        action, content = parse_new_json_format(json_str)
        assert action == 'crop'
        assert content['image_id'] == json_obj['arguments']['image_id']
        assert content['region'] == json_obj['arguments']['region']
    
    @given(json_obj=valid_ocr_json_strategy())
    @settings(max_examples=100)
    def test_ocr_json_parsing_property(self, json_obj):
        """
        Property: For any valid OCR JSON, parser extracts action='ocr' and arguments correctly.
        **Validates: Requirements 1.1, 1.4**
        """
        json_str = json.dumps(json_obj)
        
        result = extract_json_response(json_str)
        assert result is not None
        
        action, content = parse_new_json_format(json_str)
        assert action == 'ocr'
        assert content['image_id'] == json_obj['arguments']['image_id']
        assert content['region'] == json_obj['arguments']['region']
    
    @given(json_obj=valid_answer_json_strategy())
    @settings(max_examples=100)
    def test_answer_json_parsing_property(self, json_obj):
        """
        Property: For any valid answer JSON, parser extracts action='answer' and answer correctly.
        **Validates: Requirements 1.1, 1.5**
        """
        json_str = json.dumps(json_obj)
        
        result = extract_json_response(json_str)
        assert result is not None
        
        action, content = parse_new_json_format(json_str)
        assert action == 'answer'
        assert content == json_obj['answer']
    
    @given(json_obj=any_valid_json_strategy(), prefix=st.text(max_size=50), suffix=st.text(max_size=50))
    @settings(max_examples=100)
    def test_json_embedded_in_text_property(self, json_obj, prefix, suffix):
        """
        Property: For any valid JSON embedded in text, parser extracts it correctly.
        **Validates: Requirements 2.1, 2.2**
        """
        # Filter out text that contains JSON-like characters that could interfere
        assume('{' not in prefix and '}' not in prefix)
        assume('{' not in suffix and '}' not in suffix)
        
        json_str = json.dumps(json_obj)
        text = prefix + json_str + suffix
        
        result = extract_json_response(text)
        assert result is not None, f"Failed to extract JSON from: {text[:100]}..."
        
        # Verify the extracted JSON is valid
        parsed = json.loads(result)
        assert parsed['action'] == json_obj['action']


class TestBackwardCompatibilityProperty:
    """
    Property-based tests for backward compatibility.
    
    **Feature: json-output-format-adaptation, Property 3: 向后兼容性**
    **Validates: Requirements 1.6, 2.3, 4.1, 4.2, 4.3**
    """
    
    @given(query=valid_query_strategy())
    @settings(max_examples=100)
    def test_xml_search_tag_parsing_property(self, query):
        """
        Property: For any XML search tag, parser extracts it correctly when JSON fails.
        **Validates: Requirements 1.6, 4.1**
        """
        xml_text = f'<think>reasoning</think><search>{query}</search>'
        
        # JSON should fail
        json_result = extract_json_response(xml_text)
        assert json_result is None
        
        # XML should succeed
        xml_result = extract_tags(xml_text)
        assert f'<search>{query}</search>' in xml_result
    
    @given(answer=valid_answer_strategy())
    @settings(max_examples=100)
    def test_xml_answer_tag_parsing_property(self, answer):
        """
        Property: For any XML answer tag, parser extracts it correctly when JSON fails.
        **Validates: Requirements 1.6, 4.1**
        """
        xml_text = f'<think>done</think><answer>{answer}</answer>'
        
        json_result = extract_json_response(xml_text)
        assert json_result is None
        
        xml_result = extract_tags(xml_text)
        assert f'<answer>{answer}</answer>' in xml_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
