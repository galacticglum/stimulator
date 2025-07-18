"""Helper functions for the stimulator package."""

import os
import re
from getpass import getpass
from typing import Any


def get_config_value(key: str, ask_user: bool = False, secret: bool = False) -> Any:
    """Get a configuration value from environment variables or user input.

    Args:
        key: The configuration key to retrieve.
        ask_user: If True, prompt the user for the value if not found in environment variables.
        secret: If True, use getpass to hide user input.

    Returns:
        The configuration value.
    """
    value = os.getenv(key)
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
