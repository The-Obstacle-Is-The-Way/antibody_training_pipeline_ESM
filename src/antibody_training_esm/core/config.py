"""
Core configuration defaults.

Centralizes the magic numbers used across the training pipeline so they can be
updated in one place (or overridden by CLI/config files later).
"""

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_SEQ_LENGTH = 1024
GPU_CACHE_CLEAR_INTERVAL = 10  # Clear GPU cache every N batches to prevent OOM
ERROR_PREVIEW_LIMIT = 10  # Show first N errors in validation messages
