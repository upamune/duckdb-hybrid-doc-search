"""Document splitter implementations for Markdown files."""

import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple, Callable

from chonkie import RecursiveChunker, RecursiveRules, RecursiveLevel
from lindera_py import Segmenter, Tokenizer, load_dictionary
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document
from llama_index.core.text_splitter import TokenTextSplitter

from duckdb_hybrid_document_search.utils.yaml_front_matter import strip_yaml_front_matter


class SplitterType(Enum):
    """Type of splitter to use."""
    LLAMA_INDEX = auto()  # llama-index based splitter
    CHONKIE = auto()      # chonkie based splitter


# Helper function for parallel processing with LlamaIndexSplitter
def _process_file(args):
    """Process a file in a separate process using llama-index.

    Args:
        args: Tuple of (file_path, dict_type, chunk_size, chunk_overlap)

    Returns:
        List of Chunk objects
    """
    file_path, dict_type, chunk_size, chunk_overlap = args

    # Create a new tokenizer for this process
    dictionary = load_dictionary(dict_type)
    segmenter = Segmenter("normal", dictionary)
    tokenizer = Tokenizer(segmenter)

    # Helper function to tokenize text
    def tokenize_text(text):
        tokens = tokenizer.tokenize(text)
        return " ".join(token.text for token in tokens)

    # Create text splitter
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n",
    )

    # Create node parser
    md_parser = MarkdownNodeParser()

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip YAML front matter
    content_no_yaml = strip_yaml_front_matter(content)

    # Create a Document object
    doc = Document(text=content_no_yaml)

    # Parse into nodes based on Markdown headers
    nodes = md_parser.get_nodes_from_documents([doc])

    chunks = []
    for node in nodes:
        # Get header path
        header_path = "/".join(node.metadata.get("section_headers", []))

        # Split node text if it's too long
        if len(node.text) > chunk_size * 4:  # Rough character estimate
            sub_texts = text_splitter.split_text(node.text)
            for sub_text in sub_texts:
                # Get line numbers
                start_idx = content.find(sub_text)
                if start_idx == -1:
                    line_start, line_end = 1, 1
                else:
                    end_idx = start_idx + len(sub_text)
                    line_start = content[:start_idx].count("\n") + 1
                    line_end = content[:end_idx].count("\n") + 1

                tokens = tokenize_text(sub_text)
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        header_path=header_path,
                        line_start=line_start,
                        line_end=line_end,
                        content=sub_text,
                        tokens=tokens,
                    )
                )
        else:
            # Get line numbers
            start_idx = content.find(node.text)
            if start_idx == -1:
                line_start, line_end = 1, 1
            else:
                end_idx = start_idx + len(node.text)
                line_start = content[:start_idx].count("\n") + 1
                line_end = content[:end_idx].count("\n") + 1

            tokens = tokenize_text(node.text)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    header_path=header_path,
                    line_start=line_start,
                    line_end=line_end,
                    content=node.text,
                    tokens=tokens,
                )
            )

    return chunks


# Helper function for parallel processing with ChonkieSplitter
def _process_file_chonkie(args):
    """Process a file in a separate process using chonkie.

    Args:
        args: Tuple of (file_path, chunk_size, chunk_overlap, dict_type)

    Returns:
        List of Chunk objects
    """
    file_path, chunk_size, chunk_overlap, dict_type = args

    # Create rules for Markdown structure
    rules = RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=['######', '#####', '####', '###', '##', '#']),
            RecursiveLevel(delimiters=['\n\n', '\n', '\r\n', '\r']),
            RecursiveLevel(delimiters=['.?!;:']),
            RecursiveLevel()
        ]
    )

    # Initialize chunker
    chunker = RecursiveChunker(rules=rules, chunk_size=chunk_size)

    # Initialize tokenizer
    dictionary = load_dictionary(dict_type)
    segmenter = Segmenter("normal", dictionary)
    tokenizer = Tokenizer(segmenter)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip YAML front matter
    content_no_yaml = strip_yaml_front_matter(content)

    # Chunk using Chonkie
    chonkie_chunks = chunker(content_no_yaml)

    chunks = []
    for chunk in chonkie_chunks:
        # Get line numbers
        start_idx = content.find(chunk.text)
        if start_idx == -1:
            line_start, line_end = 1, 1
        else:
            end_idx = start_idx + len(chunk.text)
            line_start = content[:start_idx].count("\n") + 1
            line_end = content[:end_idx].count("\n") + 1

        # Extract header path
        header_path = ""
        if start_idx != -1:
            text_before = content[:start_idx]
            lines = text_before.split("\n")

            # Find the last headers
            headers = []
            current_level = 100  # Large initial value

            for line in reversed(lines):
                if line.startswith("#"):
                    # Get header level
                    level = 0
                    for char in line:
                        if char == '#':
                            level += 1
                        else:
                            break

                    # Only add higher-level headers
                    if level < current_level:
                        current_level = level
                        header_text = line.lstrip('#').strip()
                        headers.insert(0, header_text)

            header_path = "/".join(headers)

        # Tokenize
        tokens = tokenizer.tokenize(chunk.text)
        token_str = " ".join(token.text for token in tokens)

        chunks.append(
            Chunk(
                file_path=file_path,
                header_path=header_path,
                line_start=line_start,
                line_end=line_end,
                content=chunk.text,
                tokens=token_str,
            )
        )

    return chunks


@dataclass
class Chunk:
    """A chunk of text from a document."""

    file_path: str
    header_path: str
    line_start: int
    line_end: int
    content: str
    tokens: str


class LlamaIndexSplitter:
    """Split Markdown documents into chunks using llama-index.

    Uses MarkdownNodeParser and TokenTextSplitter from llama-index.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer_dict_type: str = "ipadic",
    ) -> None:
        """Initialize the splitter.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            tokenizer_dict_type: Dictionary type for Lindera tokenizer
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.md_parser = MarkdownNodeParser()
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n",
        )
        dictionary = load_dictionary(tokenizer_dict_type)
        segmenter = Segmenter("normal", dictionary)
        self.tokenizer = Tokenizer(segmenter)

    def _get_line_numbers(self, text: str, substring: str) -> Tuple[int, int]:
        """Get the line numbers for a substring in text.

        Args:
            text: The full text
            substring: The substring to find

        Returns:
            Tuple of (start_line, end_line) (1-indexed)
        """
        lines = text.splitlines(keepends=True)
        start_idx = text.find(substring)
        if start_idx == -1:
            return (1, 1)  # Default if not found

        end_idx = start_idx + len(substring)

        # Count newlines before start_idx
        start_line = text[:start_idx].count("\n") + 1

        # Count newlines before end_idx
        end_line = text[:end_idx].count("\n") + 1

        return (start_line, end_line)

    def _tokenize(self, text: str) -> str:
        """Tokenize text using Lindera.

        Args:
            text: Text to tokenize

        Returns:
            Space-joined tokens
        """
        tokens = self.tokenizer.tokenize(text)
        return " ".join(token.text for token in tokens)

    def split_file(self, args):
        """Split a Markdown file into chunks.

        Args:
            args: Either a file path string or a tuple (file_path, dict_type)

        Returns:
            List of Chunk objects
        """
        # Handle both string and tuple arguments for backward compatibility
        if isinstance(args, tuple):
            file_path, dict_type = args
        else:
            file_path = args
            dict_type = "ipadic"  # Default dictionary type

        # Create a new tokenizer for this process
        dictionary = load_dictionary(dict_type)
        segmenter = Segmenter("normal", dictionary)
        local_tokenizer = Tokenizer(segmenter)

        # Helper function to tokenize text
        def tokenize_text(text):
            tokens = local_tokenizer.tokenize(text)
            return " ".join(token.text for token in tokens)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Strip YAML front matter
        content_no_yaml = strip_yaml_front_matter(content)

        # Create a Document object
        doc = Document(text=content_no_yaml)

        # Parse into nodes based on Markdown headers
        nodes = self.md_parser.get_nodes_from_documents([doc])

        chunks = []
        for node in nodes:
            # Get header path
            header_path = "/".join(node.metadata.get("section_headers", []))

            # Split node text if it's too long
            if len(node.text) > self.chunk_size * 4:  # Rough character estimate
                sub_texts = self.text_splitter.split_text(node.text)
                for sub_text in sub_texts:
                    line_start, line_end = self._get_line_numbers(content, sub_text)
                    tokens = tokenize_text(sub_text)
                    chunks.append(
                        Chunk(
                            file_path=file_path,
                            header_path=header_path,
                            line_start=line_start,
                            line_end=line_end,
                            content=sub_text,
                            tokens=tokens,
                        )
                    )
            else:
                line_start, line_end = self._get_line_numbers(content, node.text)
                tokens = tokenize_text(node.text)
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        header_path=header_path,
                        line_start=line_start,
                        line_end=line_end,
                        content=node.text,
                        tokens=tokens,
                    )
                )

        return chunks

    def split_directory(
        self,
        directory_paths: List[str],
        workers: int = 4,
        file_extension: str = ".md",
        progress_callback=None,
    ) -> List[Chunk]:
        """Split all Markdown files in directories.

        Args:
            directory_paths: List of directory paths to process
            workers: Number of worker processes
            file_extension: File extension to filter
            progress_callback: Optional callback function to report progress

        Returns:
            List of Chunk objects from all files
        """
        md_files = []
        for directory in directory_paths:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(file_extension):
                        md_files.append(os.path.join(root, file))

        total_files = len(md_files)
        if progress_callback:
            progress_callback(0, total_files)

        # Create a list of tuples with file path, dict type, chunk size, and chunk overlap
        args_list = [
            (file_path, "ipadic", self.chunk_size, self.chunk_overlap) for file_path in md_files
        ]

        all_chunks = []
        if workers <= 1:
            # Process sequentially if workers is 1 or less
            for i, args in enumerate(args_list):
                all_chunks.extend(_process_file(args))
                if progress_callback:
                    progress_callback(i + 1, total_files)
        else:
            # Process in parallel using the standalone function
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Use as_completed to track progress
                futures = {executor.submit(_process_file, args): i
                          for i, args in enumerate(args_list)}

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    chunk_list = future.result()
                    all_chunks.extend(chunk_list)
                    if progress_callback:
                        progress_callback(i + 1, total_files)

        return all_chunks


class ChonkieSplitter:
    """Split Markdown documents into chunks using Chonkie."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer_dict_type: str = "ipadic",
    ) -> None:
        """Initialize the splitter.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            tokenizer_dict_type: Dictionary type for Lindera tokenizer
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Define rules optimized for Markdown structure
        rules = RecursiveRules(
            levels=[
                RecursiveLevel(delimiters=['######', '#####', '####', '###', '##', '#']),
                RecursiveLevel(delimiters=['\n\n', '\n', '\r\n', '\r']),
                RecursiveLevel(delimiters=['.?!;:']),
                RecursiveLevel()
            ]
        )

        # Initialize Chonkie chunker
        self.chunker = RecursiveChunker(rules=rules, chunk_size=chunk_size)

        # Initialize Lindera tokenizer
        dictionary = load_dictionary(tokenizer_dict_type)
        segmenter = Segmenter("normal", dictionary)
        self.tokenizer = Tokenizer(segmenter)

    def split_file(self, file_path: str) -> List[Chunk]:
        """Split a Markdown file into chunks.

        Args:
            file_path: Path to Markdown file

        Returns:
            List of Chunk objects
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Strip YAML front matter
        content_no_yaml = strip_yaml_front_matter(content)

        # Chunk using Chonkie
        chonkie_chunks = self.chunker(content_no_yaml)

        chunks = []
        for chunk in chonkie_chunks:
            # Get line numbers
            line_start, line_end = self._get_line_numbers(content, chunk.text)

            # Extract header path
            header_path = self._extract_header_path(content, chunk.text)

            # Tokenize
            tokens = self._tokenize(chunk.text)

            chunks.append(
                Chunk(
                    file_path=file_path,
                    header_path=header_path,
                    line_start=line_start,
                    line_end=line_end,
                    content=chunk.text,
                    tokens=tokens,
                )
            )

        return chunks

    def split_directory(
        self,
        directory_paths: List[str],
        workers: int = 4,
        file_extension: str = ".md",
        progress_callback=None,
    ) -> List[Chunk]:
        """Split all Markdown files in directories.

        Args:
            directory_paths: List of directory paths to process
            workers: Number of worker processes
            file_extension: File extension to filter
            progress_callback: Optional callback function to report progress

        Returns:
            List of Chunk objects from all files
        """
        md_files = []
        for directory in directory_paths:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(file_extension):
                        md_files.append(os.path.join(root, file))

        total_files = len(md_files)
        if progress_callback:
            progress_callback(0, total_files)

        all_chunks = []
        if workers <= 1:
            # Process sequentially
            for i, file_path in enumerate(md_files):
                all_chunks.extend(self.split_file(file_path))
                if progress_callback:
                    progress_callback(i + 1, total_files)
        else:
            # Process in parallel using a standalone function
            # Create a list of tuples with file path, chunk size, chunk overlap, and tokenizer dict type
            args_list = [
                (file_path, self.chunk_size, self.chunk_overlap, "ipadic")
                for file_path in md_files
            ]

            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Use as_completed to track progress
                futures = {executor.submit(_process_file_chonkie, args): i
                          for i, args in enumerate(args_list)}

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    chunk_list = future.result()
                    all_chunks.extend(chunk_list)
                    if progress_callback:
                        progress_callback(i + 1, total_files)

        return all_chunks

    def _get_line_numbers(self, text: str, substring: str) -> Tuple[int, int]:
        """Get the line numbers for a substring in text."""
        start_idx = text.find(substring)
        if start_idx == -1:
            return (1, 1)  # Default if not found

        end_idx = start_idx + len(substring)
        start_line = text[:start_idx].count("\n") + 1
        end_line = text[:end_idx].count("\n") + 1

        return (start_line, end_line)

    def _tokenize(self, text: str) -> str:
        """Tokenize text using Lindera."""
        tokens = self.tokenizer.tokenize(text)
        return " ".join(token.text for token in tokens)

    def _extract_header_path(self, full_text: str, chunk_text: str) -> str:
        """Extract header path for a chunk.

        This is a simplified implementation that looks for headers before the chunk.
        """
        start_idx = full_text.find(chunk_text)
        if start_idx == -1:
            return ""

        text_before = full_text[:start_idx]
        lines = text_before.split("\n")

        # Find the last headers
        headers = []
        current_level = 100  # Large initial value

        for line in reversed(lines):
            if line.startswith("#"):
                # Get header level
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break

                # Only add higher-level headers
                if level < current_level:
                    current_level = level
                    header_text = line.lstrip('#').strip()
                    headers.insert(0, header_text)

        return "/".join(headers)


def create_splitter(
    splitter_type: SplitterType = SplitterType.CHONKIE,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    tokenizer_dict_type: str = "ipadic",
):
    """Create a splitter instance.

    Args:
        splitter_type: Type of splitter to use (CHONKIE or LLAMA_INDEX)
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        tokenizer_dict_type: Dictionary type for Lindera tokenizer

    Returns:
        A splitter instance (ChonkieSplitter or LlamaIndexSplitter)
    """
    if splitter_type == SplitterType.CHONKIE:
        return ChonkieSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer_dict_type=tokenizer_dict_type,
        )
    elif splitter_type == SplitterType.LLAMA_INDEX:
        return LlamaIndexSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer_dict_type=tokenizer_dict_type,
        )
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")