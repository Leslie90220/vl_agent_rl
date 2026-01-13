"""
Unit tests for JSON output format parsing in generation.py

Tests the following requirements:
- 1.1: Valid JSON response parsing
- 1.2: Search action parsing
- 1.3: Crop action parsing
- 1.4: OCR action parsing
- 1.5: Answer action parsing
- 1.6: Fallback to XML tag parsing
"""

import pytest
import json
import re
from typing import Tuple, List, Any, Optional


# Standalone parsing functions for testing (extracted from generation.py)
VALID_ACTIONS = {'search', 'crop', 'ocr', 'answer'}


def extract_json_response(text: str) -> Optional[str]:
    """Extract and validate JSON response format.
    
    Expected format:
    {
      "think": "string, brief plan (1-3 sentences)",
      "action": "search" | "crop" | "ocr" | "answer",
      "arguments": { },
      "answer": null | "string"
    }
    """
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


class TestExtractJsonResponse:
    """Tests for extract_json_response function."""
    
    def test_valid_search_json(self):
        """Test parsing valid search JSON."""
        text = '{"think": "I need to search", "action": "search", "arguments": {"query": "test query"}, "answer": null}'
        result = extract_json_response(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed['action'] == 'search'
        assert parsed['arguments']['query'] == 'test query'
    
    def test_valid_crop_json(self):
        """Test parsing valid crop JSON."""
        text = '{"think": "I need to crop", "action": "crop", "arguments": {"image_id": "image_01", "region": [0, 0, 100, 100]}, "answer": null}'
        result = extract_json_response(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed['action'] == 'crop'
        assert parsed['arguments']['image_id'] == 'image_01'
    
    def test_valid_ocr_json(self):
        """Test parsing valid OCR JSON."""
        text = '{"think": "I need OCR", "action": "ocr", "arguments": {"image_id": "image_02", "region": [10, 20, 30, 40]}, "answer": null}'
        result = extract_json_response(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed['action'] == 'ocr'
    
    def test_valid_answer_json(self):
        """Test parsing valid answer JSON."""
        text = '{"think": "I can answer now", "action": "answer", "arguments": {}, "answer": "Beijing"}'
        result = extract_json_response(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed['action'] == 'answer'
        assert parsed['answer'] == 'Beijing'
    
    def test_json_embedded_in_text(self):
        """Test extracting JSON embedded in other text."""
        text = 'Some prefix text {"think": "test", "action": "search", "arguments": {"query": "q"}, "answer": null} some suffix'
        result = extract_json_response(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed['action'] == 'search'
    
    def test_invalid_action(self):
        """Test that invalid action values are rejected."""
        text = '{"think": "test", "action": "invalid_action", "arguments": {}, "answer": null}'
        result = extract_json_response(text)
        assert result is None
    
    def test_missing_required_field(self):
        """Test that missing required fields are rejected."""
        # Missing 'think' field
        text = '{"action": "search", "arguments": {"query": "q"}, "answer": null}'
        result = extract_json_response(text)
        assert result is None
    
    def test_invalid_json(self):
        """Test that malformed JSON is rejected."""
        text = '{"think": "test", "action": "search", "arguments": {"query": "q"}, "answer": null'  # Missing closing brace
        result = extract_json_response(text)
        assert result is None
    
    def test_no_json(self):
        """Test that text without JSON returns None."""
        text = 'This is just plain text without any JSON'
        result = extract_json_response(text)
        assert result is None
    
    def test_arguments_not_dict(self):
        """Test that non-dict arguments are rejected."""
        text = '{"think": "test", "action": "search", "arguments": "not a dict", "answer": null}'
        result = extract_json_response(text)
        assert result is None


class TestParseNewJsonFormat:
    """Tests for parse_new_json_format function."""
    
    def test_search_action_extracts_query(self):
        """Test that search action extracts query from arguments."""
        text = '{"think": "searching", "action": "search", "arguments": {"query": "my search query"}, "answer": null}'
        action, content = parse_new_json_format(text)
        assert action == 'search'
        assert content == 'my search query'
    
    def test_crop_action_extracts_arguments(self):
        """Test that crop action extracts full arguments dict."""
        text = '{"think": "cropping", "action": "crop", "arguments": {"image_id": "image_01", "region": [0, 0, 50, 50]}, "answer": null}'
        action, content = parse_new_json_format(text)
        assert action == 'crop'
        assert isinstance(content, dict)
        assert content['image_id'] == 'image_01'
        assert content['region'] == [0, 0, 50, 50]
    
    def test_ocr_action_extracts_arguments(self):
        """Test that OCR action extracts full arguments dict."""
        text = '{"think": "ocr", "action": "ocr", "arguments": {"image_id": "image_03", "region": [10, 10, 100, 100]}, "answer": null}'
        action, content = parse_new_json_format(text)
        assert action == 'ocr'
        assert isinstance(content, dict)
        assert content['image_id'] == 'image_03'
    
    def test_answer_action_extracts_answer(self):
        """Test that answer action extracts answer field."""
        text = '{"think": "answering", "action": "answer", "arguments": {}, "answer": "The answer is 42"}'
        action, content = parse_new_json_format(text)
        assert action == 'answer'
        assert content == 'The answer is 42'
    
    def test_invalid_returns_none(self):
        """Test that invalid input returns None, None."""
        text = 'not valid json'
        action, content = parse_new_json_format(text)
        assert action is None
        assert content is None


class TestExtractTags:
    """Tests for legacy XML tag extraction."""
    
    def test_extract_search_tag(self):
        """Test extracting search tag."""
        text = '<think>reasoning</think><search>query text</search>'
        result = extract_tags(text)
        assert '<think>reasoning</think>' in result
        assert '<search>query text</search>' in result
    
    def test_extract_answer_tag(self):
        """Test extracting answer tag."""
        text = '<think>done</think><answer>Beijing</answer>'
        result = extract_tags(text)
        assert '<answer>Beijing</answer>' in result
    
    def test_extract_tools_call_tag(self):
        """Test extracting tools_call tag."""
        text = '<think>need crop</think><tools_call>{"type": "crop"}</tools_call>'
        result = extract_tags(text)
        assert '<tools_call>{"type": "crop"}</tools_call>' in result
    
    def test_no_tags_returns_empty(self):
        """Test that text without tags returns empty string."""
        text = 'plain text without tags'
        result = extract_tags(text)
        assert result == ''


class TestBackwardCompatibility:
    """Tests for backward compatibility with XML format."""
    
    def test_json_takes_priority_over_xml(self):
        """Test that JSON format is tried before XML."""
        # Text with both JSON and XML - JSON should be extracted
        json_text = '{"think": "test", "action": "search", "arguments": {"query": "json query"}, "answer": null}'
        xml_text = '<search>xml query</search>'
        combined = json_text + xml_text
        
        # JSON should be found
        json_result = extract_json_response(combined)
        assert json_result is not None
        parsed = json.loads(json_result)
        assert parsed['arguments']['query'] == 'json query'
    
    def test_xml_fallback_when_no_json(self):
        """Test that XML is used when JSON is not present."""
        text = '<think>reasoning</think><search>xml query</search>'
        
        # JSON should not be found
        json_result = extract_json_response(text)
        assert json_result is None
        
        # XML should be found
        xml_result = extract_tags(text)
        assert '<search>xml query</search>' in xml_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
