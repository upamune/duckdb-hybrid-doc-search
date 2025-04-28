"""Tests for the Markdown splitter."""

import os
import tempfile
from pathlib import Path

import pytest

from duckdb_hybrid_document_search.splitter import Chunk, MarkdownSplitter
from duckdb_hybrid_document_search.utils.yaml_front_matter import strip_yaml_front_matter


def test_strip_yaml_front_matter():
    """Test stripping YAML front matter."""
    content = """---
title: Test Document
author: Test Author
---

# Header 1

Content here.
"""
    result = strip_yaml_front_matter(content)
    assert result.strip().startswith("# Header 1")
    assert "title: Test Document" not in result


def test_get_line_numbers():
    """Test getting line numbers for a substring."""
    content = """Line 1
Line 2
Line 3
Line 4
Line 5"""

    splitter = MarkdownSplitter()
    start, end = splitter._get_line_numbers(content, "Line 3")
    assert start == 3
    assert end == 3

    # Multi-line substring
    start, end = splitter._get_line_numbers(content, "Line 2\nLine 3")
    assert start == 2
    assert end == 3


def test_tokenize():
    """Test tokenization with Lindera."""
    splitter = MarkdownSplitter()
    tokens = splitter._tokenize("これはテストです。")
    assert isinstance(tokens, str)
    assert " " in tokens  # Should contain spaces between tokens


def test_split_file():
    """Test splitting a Markdown file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test Markdown file
        test_md = os.path.join(tmpdir, "test.md")
        with open(test_md, "w", encoding="utf-8") as f:
            f.write(
                """---
title: Test Document
---

# Section 1

This is content for section 1.

## Subsection 1.1

More content here.

# Section 2

Final content.
"""
            )

        splitter = MarkdownSplitter()
        chunks = splitter.split_file(test_md)

        # Should have at least 2 chunks (one per main section)
        assert len(chunks) >= 2

        # Check chunk properties
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.file_path == test_md
            assert isinstance(chunk.header_path, str)
            assert chunk.line_start > 0
            assert chunk.line_end >= chunk.line_start
            assert chunk.content
            assert chunk.tokens


def test_split_directory():
    """Test splitting multiple Markdown files in a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test directory structure
        docs_dir = os.path.join(tmpdir, "docs")
        os.makedirs(docs_dir)

        # Create a few test files
        for i in range(3):
            with open(os.path.join(docs_dir, f"doc{i}.md"), "w", encoding="utf-8") as f:
                f.write(
                    f"""# Document {i}

Content for document {i}.

## Section {i}.1

More content.
"""
                )

        # Create a non-markdown file that should be ignored
        with open(os.path.join(docs_dir, "ignore.txt"), "w") as f:
            f.write("This should be ignored")

        splitter = MarkdownSplitter()
        chunks = splitter.split_directory([docs_dir], workers=1)

        # Should have chunks from 3 files
        assert len(chunks) > 0

        # Check that all chunks are from .md files
        for chunk in chunks:
            assert chunk.file_path.endswith(".md")
            assert "ignore.txt" not in chunk.file_path
