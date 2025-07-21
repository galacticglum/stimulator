"""Helper functions for the stimulator package."""

import os
import re
from getpass import getpass
from typing import Any


def get_config_value(
    key: str, ask_user: bool = False, secret: bool = False, allow_empty: bool = True
) -> Any:
    """Get a configuration value from environment variables or user input.

    Args:
        key: The configuration key to retrieve.
        ask_user: If True, prompt the user for the value if not found in environment variables.
        secret: If True, use getpass to hide user input.
        allow_empty: If True, allow the value to be empty (None) in the environment.
            In this case, the user will not be prompted for input if the value is not found.

    Returns:
        The configuration value.
    """
    value = os.getenv(key)
    if allow_empty and not value:
        return None
    if not value and ask_user:
        if secret:
            value = getpass(f"Please enter the value for '{key}': ")
        else:
            value = input(f"Please enter the value for '{key}': ")
    return value


def remove_urls(text: str) -> str:
    """Remove URLs from the given text."""
    URL_MATCH_PATTERN = re.compile(
        r"(?i)(https?:\/\/(?:www\.|(?!www))"
        r"[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|"
        r"www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|"
        r"https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|"
        r"www\.[a-zA-Z0-9]+\.[^\s]{2,})"
    )
    return re.sub(URL_MATCH_PATTERN, "", text)


def get_device() -> str:
    """Get the device to use for model training or inference.

    Prefer GPU or MPS if available, otherwise use CPU.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
