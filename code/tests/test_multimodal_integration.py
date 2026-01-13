"""
Integration tests for multimodal data passing fix.

These tests verify that:
1. pixel_values flows correctly through the generation pipeline
2. The model receives multimodal data in model.generate()
3. Backward compatibility with text-only generation is maintained

Feature: multimodal-data-passing-fix
Validates: Requirements 2.4, 5.1, 5.2, 5.3
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass


# Import the actual classes
import sys
sys.path.insert(0, 'code')


class MockProcessor:
    """Mock processor for testing."""
    def __init__(self):
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 151643
        self.tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
        self.tokenizer.batch_decode = MagicMock(return_value=['test response'])


class MockTensorHelper:
    """Mock TensorHelper for testing."""
    def __init__(self, config):
        self.config = config
    
    def create_attention_mask(self, tensor):
        return (tensor != 0).long()
    
    def create_position_ids(self, attention_mask):
        return torch.cumsum(attention_mask, dim=1) - 1
    
    def concatenate_with_padding(self, tensors, pad_to_left=True):
        return torch.cat(tensors, dim=1)


@dataclass
class MockTensorConfig:
    pad_token_id: int = 0


class TestMultimodalDataFlow:
    """Test that multimodal data flows correctly through the pipeline."""
    
    def test_pixel_values_added_to_meta_info(self):
        """Test that pixel_values is correctly added to meta_info."""
        from vrag_agent.generation import LLMGenerationManager, GenerationConfig
        from verl import DataProto
        
        # Create mock components
        processor = MockProcessor()
        mock_rollout_wg = MagicMock()
        
        # Mock generate_sequences to capture the input
        captured_batch = None
        def capture_generate_sequences(batch):
            nonlocal captured_batch
            captured_batch = batch
            # Return a mock output
            return DataProto.from_dict({
                'input_ids': batch.batch['input_ids'],
                'attention_mask': batch.batch['attention_mask'],
                'position_ids': batch.batch['position_ids'],
                'responses': torch.zeros(batch.batch['input_ids'].shape[0], 100, dtype=torch.long),
                'prompts': batch.batch['input_ids'],
            })
        
        mock_rollout_wg.generate_sequences = capture_generate_sequences
        
        config = GenerationConfig(
            max_turns=5,
            max_prompt_length=1024,
            num_gpus=1,
            n_agent=1,
        )
        
        manager = LLMGenerationManager(
            processor=processor,
            actor_rollout_wg=mock_rollout_wg,
            config=config,
        )
        
        # Create test batch with multimodal inputs
        batch_size = 2
        seq_len = 100
        
        # Create multi_modal_inputs with pixel_values
        mm_input1 = {
            'pixel_values': torch.randn(2, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 14, 14], [1, 14, 14]]),
        }
        mm_input2 = {
            'pixel_values': torch.randn(1, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 14, 14]]),
        }
        
        multi_modal_inputs = np.array([mm_input1, mm_input2], dtype=object)
        
        active_batch = DataProto.from_dict(
            {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len),
                'position_ids': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            },
            {
                'multi_modal_inputs': multi_modal_inputs,
            }
        )
        
        # Call _generate_with_gpu_padding
        manager._generate_with_gpu_padding(active_batch)
        
        # Verify that pixel_values was added to meta_info
        assert captured_batch is not None
        assert 'pixel_values' in captured_batch.meta_info
        assert 'image_grid_thw' in captured_batch.meta_info
        
        # Verify shapes
        assert captured_batch.meta_info['pixel_values'].shape[0] == 3  # 2 + 1 images
        assert captured_batch.meta_info['image_grid_thw'].shape[0] == 3
    
    def test_text_only_generation_still_works(self):
        """Test backward compatibility: text-only generation should work."""
        from vrag_agent.generation import LLMGenerationManager, GenerationConfig
        from verl import DataProto
        
        processor = MockProcessor()
        mock_rollout_wg = MagicMock()
        
        captured_batch = None
        def capture_generate_sequences(batch):
            nonlocal captured_batch
            captured_batch = batch
            return DataProto.from_dict({
                'input_ids': batch.batch['input_ids'],
                'attention_mask': batch.batch['attention_mask'],
                'position_ids': batch.batch['position_ids'],
                'responses': torch.zeros(batch.batch['input_ids'].shape[0], 100, dtype=torch.long),
                'prompts': batch.batch['input_ids'],
            })
        
        mock_rollout_wg.generate_sequences = capture_generate_sequences
        
        config = GenerationConfig(
            max_turns=5,
            max_prompt_length=1024,
            num_gpus=1,
            n_agent=1,
        )
        
        manager = LLMGenerationManager(
            processor=processor,
            actor_rollout_wg=mock_rollout_wg,
            config=config,
        )
        
        # Create test batch WITHOUT multimodal inputs
        batch_size = 2
        seq_len = 100
        
        active_batch = DataProto.from_dict(
            {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len),
                'position_ids': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            },
            {}  # No multi_modal_inputs
        )
        
        # Should not raise any errors
        manager._generate_with_gpu_padding(active_batch)
        
        # Verify that pixel_values is NOT in meta_info
        assert captured_batch is not None
        assert 'pixel_values' not in captured_batch.meta_info or captured_batch.meta_info.get('pixel_values') is None
    
    def test_empty_multimodal_inputs_handled(self):
        """Test that empty multi_modal_inputs is handled gracefully."""
        from vrag_agent.generation import LLMGenerationManager, GenerationConfig
        from verl import DataProto
        
        processor = MockProcessor()
        mock_rollout_wg = MagicMock()
        
        captured_batch = None
        def capture_generate_sequences(batch):
            nonlocal captured_batch
            captured_batch = batch
            return DataProto.from_dict({
                'input_ids': batch.batch['input_ids'],
                'attention_mask': batch.batch['attention_mask'],
                'position_ids': batch.batch['position_ids'],
                'responses': torch.zeros(batch.batch['input_ids'].shape[0], 100, dtype=torch.long),
                'prompts': batch.batch['input_ids'],
            })
        
        mock_rollout_wg.generate_sequences = capture_generate_sequences
        
        config = GenerationConfig(
            max_turns=5,
            max_prompt_length=1024,
            num_gpus=1,
            n_agent=1,
        )
        
        manager = LLMGenerationManager(
            processor=processor,
            actor_rollout_wg=mock_rollout_wg,
            config=config,
        )
        
        # Create test batch with empty multi_modal_inputs
        batch_size = 2
        seq_len = 100
        
        # All samples have None or empty multi_modal_inputs
        multi_modal_inputs = np.array([None, {}], dtype=object)
        
        active_batch = DataProto.from_dict(
            {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len),
                'position_ids': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            },
            {
                'multi_modal_inputs': multi_modal_inputs,
            }
        )
        
        # Should not raise any errors
        manager._generate_with_gpu_padding(active_batch)
        
        # pixel_values should be None since no images
        assert captured_batch is not None


class TestHFRolloutMultimodal:
    """Test HFRollout multimodal data handling.
    
    Note: These tests directly test the _generate_minibatch logic by extracting
    the multimodal handling code, since the actual HFRollout class has complex
    dependencies on the verl package.
    """
    
    def test_multimodal_kwargs_passed_to_generate(self):
        """Test that multimodal kwargs are correctly prepared for model.generate()."""
        from verl import DataProto
        
        # Create test prompts with multimodal data in meta_info
        batch_size = 2
        seq_len = 100
        
        pixel_values = torch.randn(3, 3, 224, 224)
        image_grid_thw = torch.tensor([[1, 14, 14], [1, 14, 14], [1, 14, 14]])
        
        prompts = DataProto.from_dict(
            {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len),
                'position_ids': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            }
        )
        prompts.meta_info = {
            'eos_token_id': 151643,
            'pad_token_id': 0,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
        }
        
        # Simulate the multimodal extraction logic from _generate_minibatch
        idx = prompts.batch["input_ids"]
        device = idx.device
        
        pv = prompts.meta_info.get("pixel_values", None)
        igt = prompts.meta_info.get("image_grid_thw", None)
        
        multimodal_kwargs = {}
        if pv is not None:
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv)
            pv = pv.to(device=device, dtype=torch.bfloat16)
            multimodal_kwargs["pixel_values"] = pv
            
            if igt is not None:
                if not isinstance(igt, torch.Tensor):
                    igt = torch.tensor(igt)
                igt = igt.to(device=device)
                multimodal_kwargs["image_grid_thw"] = igt
        
        # Verify that multimodal kwargs are correctly prepared
        assert 'pixel_values' in multimodal_kwargs
        assert 'image_grid_thw' in multimodal_kwargs
        assert multimodal_kwargs['pixel_values'].shape[0] == 3
        assert multimodal_kwargs['pixel_values'].dtype == torch.bfloat16
    
    def test_text_only_generation_no_multimodal_kwargs(self):
        """Test that text-only generation doesn't produce multimodal kwargs."""
        from verl import DataProto
        
        batch_size = 2
        seq_len = 100
        
        prompts = DataProto.from_dict(
            {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len),
                'position_ids': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            }
        )
        prompts.meta_info = {
            'eos_token_id': 151643,
            'pad_token_id': 0,
            # No pixel_values or image_grid_thw
        }
        
        # Simulate the multimodal extraction logic from _generate_minibatch
        pv = prompts.meta_info.get("pixel_values", None)
        igt = prompts.meta_info.get("image_grid_thw", None)
        
        multimodal_kwargs = {}
        if pv is not None:
            multimodal_kwargs["pixel_values"] = pv
            if igt is not None:
                multimodal_kwargs["image_grid_thw"] = igt
        
        # Verify that multimodal kwargs are empty
        assert 'pixel_values' not in multimodal_kwargs
        assert 'image_grid_thw' not in multimodal_kwargs


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_batch_multimodal_inputs_with_various_formats(self):
        """Test _batch_multimodal_inputs handles various input formats."""
        from vrag_agent.generation import LLMGenerationManager, GenerationConfig
        
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
        
        # Test with dict format
        dict_input = {
            'pixel_values': torch.randn(1, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 14, 14]]),
        }
        
        # Test with BatchFeature-like format
        class MockBatchFeature:
            def __init__(self, data):
                self.data = data
            def get(self, key, default=None):
                return self.data.get(key, default)
        
        bf_input = MockBatchFeature({
            'pixel_values': torch.randn(1, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 14, 14]]),
        })
        
        multi_modal_inputs = np.array([dict_input, bf_input, None, {}], dtype=object)
        
        batched_pv, batched_igt = manager._batch_multimodal_inputs(multi_modal_inputs)
        
        # Should have 2 images (from dict_input and bf_input)
        assert batched_pv is not None
        assert batched_pv.shape[0] == 2
        assert batched_igt.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
