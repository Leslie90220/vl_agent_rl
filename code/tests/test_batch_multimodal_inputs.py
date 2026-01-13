"""
Unit tests for _batch_multimodal_inputs method in LLMGenerationManager.

Tests cover:
- Normal case: multiple samples with images
- Edge case: empty multi_modal_inputs
- Edge case: partial samples without images
- Edge case: mixed formats (dict and BatchFeature)
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from transformers.image_processing_base import BatchFeature


class MockTensorHelper:
    """Mock TensorHelper for testing."""
    def __init__(self, config):
        self.config = config
    
    def create_attention_mask(self, tensor):
        return (tensor != 0).long()
    
    def create_position_ids(self, attention_mask):
        return torch.cumsum(attention_mask, dim=1) - 1


class MockTensorConfig:
    """Mock TensorConfig for testing."""
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id


class MockGenerationConfig:
    """Mock GenerationConfig for testing."""
    def __init__(self):
        self.max_turns = 5
        self.max_prompt_length = 1024
        self.num_gpus = 1
        self.n_agent = 1
        self.search_url = None
        self.ocr_api_url = "http://localhost:8080"
        self.max_model_len = 32768
        self.image_pad_id = 151655
        self.endoftext_id = 151643


class MockProcessor:
    """Mock processor for testing."""
    def __init__(self):
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0


def create_mock_generation_manager():
    """Create a mock LLMGenerationManager for testing."""
    # Import the actual class
    import sys
    sys.path.insert(0, 'code')
    from vrag_agent.generation import LLMGenerationManager, GenerationConfig
    from vrag_agent.tensor_helper import TensorHelper, TensorConfig
    
    processor = MockProcessor()
    config = GenerationConfig(
        max_turns=5,
        max_prompt_length=1024,
        num_gpus=1,
        n_agent=1,
    )
    
    # Create manager with mocked dependencies
    manager = LLMGenerationManager(
        processor=processor,
        actor_rollout_wg=MagicMock(),
        config=config,
    )
    
    return manager


class TestBatchMultimodalInputs:
    """Test suite for _batch_multimodal_inputs method."""
    
    @pytest.fixture
    def manager(self):
        """Create a mock LLMGenerationManager."""
        return create_mock_generation_manager()
    
    def test_normal_case_multiple_samples_with_images(self, manager):
        """Test normal case: multiple samples each with images."""
        # Create sample multi_modal_inputs with pixel_values and image_grid_thw
        sample1 = {
            'pixel_values': torch.randn(2, 3, 224, 224),  # 2 images
            'image_grid_thw': torch.tensor([[1, 14, 14], [1, 14, 14]]),
        }
        sample2 = {
            'pixel_values': torch.randn(1, 3, 224, 224),  # 1 image
            'image_grid_thw': torch.tensor([[1, 14, 14]]),
        }
        
        multi_modal_inputs = np.array([sample1, sample2], dtype=object)
        
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        # Should concatenate: 2 + 1 = 3 images total
        assert batched_pv is not None
        assert batched_pv.shape[0] == 3
        assert batched_pv.shape[1:] == (3, 224, 224)
        
        assert batched_igt is not None
        assert batched_igt.shape[0] == 3
        assert batched_igt.shape[1] == 3
    
    def test_empty_multi_modal_inputs(self, manager):
        """Test edge case: empty multi_modal_inputs array."""
        multi_modal_inputs = np.array([], dtype=object)
        
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        assert batched_pv is None
        assert batched_igt is None
    
    def test_all_none_inputs(self, manager):
        """Test edge case: all samples have None multi_modal_inputs."""
        multi_modal_inputs = np.array([None, None, None], dtype=object)
        
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        assert batched_pv is None
        assert batched_igt is None
    
    def test_partial_samples_without_images(self, manager):
        """Test edge case: some samples have images, some don't."""
        sample1 = {
            'pixel_values': torch.randn(2, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 14, 14], [1, 14, 14]]),
        }
        sample2 = None  # No images
        sample3 = {}  # Empty dict
        sample4 = {
            'pixel_values': torch.randn(1, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 14, 14]]),
        }
        
        multi_modal_inputs = np.array([sample1, sample2, sample3, sample4], dtype=object)
        
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        # Should only include samples with images: 2 + 1 = 3 images
        assert batched_pv is not None
        assert batched_pv.shape[0] == 3
        
        assert batched_igt is not None
        assert batched_igt.shape[0] == 3
    
    def test_batch_feature_format(self, manager):
        """Test handling of BatchFeature format inputs."""
        # Create a proper BatchFeature-like object
        sample1_data = {
            'pixel_values': torch.randn(1, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 14, 14]]),
        }
        
        # Create a simple class that mimics BatchFeature behavior
        class MockBatchFeature:
            def __init__(self, data):
                self.data = data
            
            def get(self, key, default=None):
                return self.data.get(key, default)
        
        sample1 = MockBatchFeature(sample1_data)
        
        multi_modal_inputs = np.array([sample1], dtype=object)
        
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        assert batched_pv is not None
        assert batched_pv.shape[0] == 1
    
    def test_mixed_image_counts(self, manager):
        """Test samples with different numbers of images."""
        sample1 = {
            'pixel_values': torch.randn(5, 3, 224, 224),  # 5 images
            'image_grid_thw': torch.tensor([[1, 14, 14]] * 5),
        }
        sample2 = {
            'pixel_values': torch.randn(1, 3, 224, 224),  # 1 image
            'image_grid_thw': torch.tensor([[1, 14, 14]]),
        }
        sample3 = {
            'pixel_values': torch.randn(3, 3, 224, 224),  # 3 images
            'image_grid_thw': torch.tensor([[1, 14, 14]] * 3),
        }
        
        multi_modal_inputs = np.array([sample1, sample2, sample3], dtype=object)
        
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        # Total: 5 + 1 + 3 = 9 images
        assert batched_pv.shape[0] == 9
        assert batched_igt.shape[0] == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
