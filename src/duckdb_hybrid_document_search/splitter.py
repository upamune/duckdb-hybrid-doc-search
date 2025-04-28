"""Markdown document splitter implementation."""

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from lindera_py import Segmenter, Tokenizer, load_dictionary
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document
from llama_index.core.text_splitter import TokenTextSplitter

from duckdb_hybrid_document_search.utils.yaml_front_matter import strip_yaml_front_matter


# Helper function for parallel processing
def _process_file(args):
    """Process a file in a separate process.

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


@dataclass
class Chunk:
    """A chunk of text from a document."""

    file_path: str
    header_path: str
    line_start: int
    line_end: int
    content: str
    tokens: str


class MarkdownSplitter:
    """Split Markdown documents into chunks for indexing."""

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
    ) -> List[Chunk]:
        """Split all Markdown files in directories.

        Args:
            directory_paths: List of directory paths to process
            workers: Number of worker processes
            file_extension: File extension to filter

        Returns:
            List of Chunk objects from all files
        """
        md_files = []
        for directory in directory_paths:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(file_extension):
                        md_files.append(os.path.join(root, file))

        # Create a list of tuples with file path, dict type, chunk size, and chunk overlap
        args_list = [
            (file_path, "ipadic", self.chunk_size, self.chunk_overlap) for file_path in md_files
        ]

        all_chunks = []
        if workers <= 1:
            # Process sequentially if workers is 1 or less
            for args in args_list:
                all_chunks.extend(_process_file(args))
        else:
            # Process in parallel using the standalone function
            with ProcessPoolExecutor(max_workers=workers) as executor:
                chunk_lists = list(executor.map(_process_file, args_list))
                for chunk_list in chunk_lists:
                    all_chunks.extend(chunk_list)

        return all_chunks
