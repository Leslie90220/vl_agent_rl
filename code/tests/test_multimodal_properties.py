"""
Property-based tests for multimodal data passing fix.

Feature: multimodal-data-passing-fix

Tests cover:
- Property 1: Pixel Values Data Flow
- Property 4: Tensor Concatenation for Multi-Turn
- Property 5: Variable Image Count Handling
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import MagicMock


# Import the actual class
import sys
sys.path.insert(0, 'code')
from vrag_agent.generation import LLMGenerationManager, GenerationConfig


class MockProcessor:
    """Mock processor for testing."""
    def __init__(self):
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0


def create_mock_generation_manager():
    """Create a mock LLMGenerationManager for testing."""
    processor = MockProcessor()
    config = GenerationConfig(
        max_turns=5,
        max_prompt_length=1024,
        num_gpus=1,
        n_agent=1,
    )
    
    manager = LLMGenerationManager(
        processor=processor,
        actor_rollout_wg=MagicMock(),
        config=config,
    )
    
    return manager


# Strategy for generating valid pixel_values tensors
@st.composite
def pixel_values_strategy(draw):
    """Generate valid pixel_values tensors."""
    num_images = draw(st.integers(min_value=1, max_value=5))
    channels = 3
    height = draw(st.sampled_from([224, 336, 448]))
    width = draw(st.sampled_from([224, 336, 448]))
    return torch.randn(num_images, channels, height, width)


# Strategy for generating valid image_grid_thw tensors
@st.composite
def image_grid_thw_strategy(draw, num_images):
    """Generate valid image_grid_thw tensors."""
    # Each row is [temporal, height_grid, width_grid]
    grids = []
    for _ in range(num_images):
        t = draw(st.integers(min_value=1, max_value=2))
        h = draw(st.integers(min_value=7, max_value=28))
        w = draw(st.integers(min_value=7, max_value=28))
        grids.append([t, h, w])
    return torch.tensor(grids)


# Strategy for generating multi_modal_inputs samples
@st.composite
def multi_modal_input_strategy(draw):
    """Generate a single multi_modal_input sample."""
    has_images = draw(st.booleans())
    if not has_images:
        return draw(st.sampled_from([None, {}]))
    
    num_images = draw(st.integers(min_value=1, max_value=5))
    channels = 3
    # Use fixed size to ensure concatenation works
    # In real usage, the processor ensures consistent sizes
    height = 224
    width = 224
    
    pixel_values = torch.randn(num_images, channels, height, width)
    image_grid_thw = torch.tensor([[1, 14, 14]] * num_images)
    
    return {
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
    }


class TestProperty1PixelValuesDataFlow:
    """
    Property 1: Pixel Values Data Flow
    
    For any batch containing multi_modal_inputs with pixel_values, when passed 
    through _batch_multimodal_inputs, the pixel_values SHALL be correctly 
    batched and accessible.
    
    Feature: multimodal-data-passing-fix, Property 1: Pixel Values Data Flow
    Validates: Requirements 1.1, 1.2
    """
    
    @pytest.fixture
    def manager(self):
        return create_mock_generation_manager()
    
    @given(st.lists(multi_modal_input_strategy(), min_size=1, max_size=8))
    @settings(max_examples=100, deadline=None)
    def test_pixel_values_preserved_after_batching(self, inputs):
        """Pixel values should be preserved after batching."""
        manager = create_mock_generation_manager()
        multi_modal_inputs = np.array(inputs, dtype=object)
        
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        # Count expected total images
        expected_images = 0
        for inp in inputs:
            if inp is not None and isinstance(inp, dict) and 'pixel_values' in inp:
                expected_images += inp['pixel_values'].shape[0]
        
        if expected_images == 0:
            assert batched_pv is None
            assert batched_igt is None
        else:
            assert batched_pv is not None
            assert batched_pv.shape[0] == expected_images
            assert batched_igt is not None
            assert batched_igt.shape[0] == expected_images
    
    @given(st.lists(multi_modal_input_strategy(), min_size=1, max_size=8))
    @settings(max_examples=100, deadline=None)
    def test_pixel_values_tensor_type_preserved(self, inputs):
        """Batched pixel_values should be a torch.Tensor."""
        manager = create_mock_generation_manager()
        multi_modal_inputs = np.array(inputs, dtype=object)
        
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        if batched_pv is not None:
            assert isinstance(batched_pv, torch.Tensor)
        if batched_igt is not None:
            assert isinstance(batched_igt, torch.Tensor)


class TestProperty4TensorConcatenation:
    """
    Property 4: Tensor Concatenation for Multi-Turn
    
    For any sequence of multi_modal_inputs across multiple turns, concatenating 
    pixel_values and image_grid_thw SHALL produce tensors with correct shapes 
    where the first dimension equals the total number of images.
    
    Feature: multimodal-data-passing-fix, Property 4: Tensor Concatenation
    Validates: Requirements 3.2, 3.3
    """
    
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=5),  # num_images
                st.just(224),  # height - fixed to ensure concatenation works
                st.just(224),  # width - fixed to ensure concatenation works
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_concatenation_preserves_total_image_count(self, image_specs):
        """Total image count should be preserved after concatenation."""
        manager = create_mock_generation_manager()
        
        inputs = []
        total_images = 0
        for num_images, height, width in image_specs:
            total_images += num_images
            inputs.append({
                'pixel_values': torch.randn(num_images, 3, height, width),
                'image_grid_thw': torch.tensor([[1, 14, 14]] * num_images),
            })
        
        multi_modal_inputs = np.array(inputs, dtype=object)
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        assert batched_pv.shape[0] == total_images
        assert batched_igt.shape[0] == total_images
    
    @given(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_concatenation_order_preserved(self, image_counts):
        """Image order should be preserved after concatenation."""
        manager = create_mock_generation_manager()
        
        inputs = []
        expected_values = []
        for i, num_images in enumerate(image_counts):
            # Use unique values to track order
            pv = torch.full((num_images, 3, 224, 224), float(i))
            inputs.append({
                'pixel_values': pv,
                'image_grid_thw': torch.tensor([[1, 14, 14]] * num_images),
            })
            expected_values.extend([float(i)] * num_images)
        
        multi_modal_inputs = np.array(inputs, dtype=object)
        batched_pv, _ = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        # Check that values are in expected order
        actual_values = batched_pv[:, 0, 0, 0].tolist()
        assert actual_values == expected_values


class TestProperty5VariableImageCount:
    """
    Property 5: Variable Image Count Handling
    
    For any batch where different samples have different numbers of images 
    (including zero), the system SHALL correctly batch and unbatch the 
    multimodal data without index errors.
    
    Feature: multimodal-data-passing-fix, Property 5: Variable Image Count
    Validates: Requirements 3.4
    """
    
    @given(st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=10))
    @settings(max_examples=100, deadline=None)
    def test_variable_image_counts_handled(self, image_counts):
        """Variable image counts should be handled without errors."""
        manager = create_mock_generation_manager()
        
        inputs = []
        total_images = 0
        for count in image_counts:
            if count == 0:
                inputs.append(None)
            else:
                total_images += count
                inputs.append({
                    'pixel_values': torch.randn(count, 3, 224, 224),
                    'image_grid_thw': torch.tensor([[1, 14, 14]] * count),
                })
        
        multi_modal_inputs = np.array(inputs, dtype=object)
        
        # Should not raise any errors
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        if total_images == 0:
            assert batched_pv is None
        else:
            assert batched_pv.shape[0] == total_images
    
    @given(st.lists(st.booleans(), min_size=1, max_size=10))
    @settings(max_examples=100, deadline=None)
    def test_mixed_none_and_valid_inputs(self, has_images_list):
        """Mixed None and valid inputs should be handled correctly."""
        manager = create_mock_generation_manager()
        
        inputs = []
        total_images = 0
        for has_images in has_images_list:
            if has_images:
                num_images = 2
                total_images += num_images
                inputs.append({
                    'pixel_values': torch.randn(num_images, 3, 224, 224),
                    'image_grid_thw': torch.tensor([[1, 14, 14]] * num_images),
                })
            else:
                inputs.append(None)
        
        multi_modal_inputs = np.array(inputs, dtype=object)
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        if total_images == 0:
            assert batched_pv is None
        else:
            assert batched_pv.shape[0] == total_images


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
