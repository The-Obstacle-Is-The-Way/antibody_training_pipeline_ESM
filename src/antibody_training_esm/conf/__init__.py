"""
Hydra configuration package

Contains YAML configs and structured config schemas.
"""

# Import config_schema to execute ConfigStore registrations
# This MUST run at import time for structured configs to work
from . import config_schema  # noqa: F401
