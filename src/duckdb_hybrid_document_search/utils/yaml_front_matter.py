"""YAML front matter parsing utilities."""

import re
from typing import Tuple


def strip_yaml_front_matter(content: str) -> str:
    """Strip YAML front matter from Markdown content.

    Args:
        content: Markdown content with possible YAML front matter

    Returns:
        Content with YAML front matter removed
    """
    # Pattern for YAML front matter: starts and ends with ---
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, content, re.DOTALL)

    if match:
        # Return content after the front matter
        return content[match.end() :]

    # No front matter found, return original content
    return content


def extract_yaml_front_matter(content: str) -> Tuple[str, str]:
    """Extract YAML front matter from Markdown content.

    Args:
        content: Markdown content with possible YAML front matter

    Returns:
        Tuple of (yaml_content, remaining_content)
    """
    # Pattern for YAML front matter: starts and ends with ---
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, content, re.DOTALL)

    if match:
        yaml_content = match.group(1)
        remaining_content = content[match.end() :]
        return yaml_content, remaining_content

    # No front matter found
    return "", content
