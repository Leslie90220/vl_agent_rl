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
"""Utils for tokenization."""

import contextlib
import warnings

__all__ = ["hf_tokenizer", "hf_processor", "_is_local_path", "_patch_hf_hub_validation"]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def _is_local_path(path: str) -> bool:
    """Check if a path looks like a local filesystem path.

    This function checks if the path appears to be a local path by examining its format,
    rather than checking if it exists. This is important for distributed environments
    where the path may exist on some nodes but not others.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path looks like a local filesystem path.
    """
    import os

    # Check if it's an absolute path (starts with / on Unix or drive letter on Windows)
    if os.path.isabs(path):
        return True
    # Check if it starts with common local path indicators
    if path.startswith(("./", "../", "~")):
        return True
    # Check if it contains path separators (likely a relative path)
    if os.sep in path or (os.altsep and os.altsep in path):
        # But exclude paths that look like HuggingFace repo IDs (org/model)
        # HF repo IDs have exactly one "/" and no other path-like characters
        parts = path.split("/")
        if len(parts) == 2 and all(part and not part.startswith(".") for part in parts):
            # Could be a HF repo ID like "meta-llama/Llama-2-7b"
            # Check if it actually exists locally
            return os.path.exists(path)
        return True
    return False


@contextlib.contextmanager
def _patch_hf_hub_validation():
    """Context manager to temporarily disable huggingface_hub repo_id validation.

    This is needed because newer versions of huggingface_hub validate repo_id format
    even for local paths, which causes errors when using absolute paths like
    '/root/models/Qwen2.5-3B-Instruct'.
    """
    try:
        from huggingface_hub.utils import _validators
        original_validate = _validators.validate_repo_id

        def patched_validate(repo_id: str) -> None:
            # Skip validation for local paths
            if _is_local_path(repo_id):
                return
            return original_validate(repo_id)

        _validators.validate_repo_id = patched_validate
        yield
    except (ImportError, AttributeError):
        # If we can't patch, just proceed without patching
        yield
    finally:
        try:
            _validators.validate_repo_id = original_validate
        except (NameError, UnboundLocalError):
            pass


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    import os

    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107

    # If path looks like a local path, set local_files_only=True only if it exists
    # This handles distributed environments where the path may not exist on all nodes
    if isinstance(name_or_path, str) and _is_local_path(name_or_path):
        if os.path.exists(name_or_path):
            kwargs.setdefault("local_files_only", True)
        else:
            # Path looks local but doesn't exist - this is likely an error in distributed setup
            raise FileNotFoundError(
                f"Local model path '{name_or_path}' does not exist on this node. "
                f"In distributed training, ensure the model path is accessible from all nodes "
                f"(e.g., use a shared filesystem like NFS or HDFS, or copy the model to all nodes)."
            )

    # Use context manager to patch huggingface_hub validation for local paths
    with _patch_hf_hub_validation():
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)

    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    import os

    from transformers import AutoProcessor

    # If path looks like a local path, set local_files_only=True only if it exists
    # This handles distributed environments where the path may not exist on all nodes
    if isinstance(name_or_path, str) and _is_local_path(name_or_path):
        if os.path.exists(name_or_path):
            kwargs.setdefault("local_files_only", True)
        else:
            # Path looks local but doesn't exist - this is likely an error in distributed setup
            # For processor, we just warn and return None since it's optional
            warnings.warn(
                f"Local model path '{name_or_path}' does not exist on this node. "
                f"Processor loading skipped.", stacklevel=1
            )
            return None

    try:
        # Use context manager to patch huggingface_hub validation for local paths
        with _patch_hf_hub_validation():
            processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
