# duckdb-hybrid-doc-search

A tool for hybrid indexing of internal documents managed in Markdown using DuckDB 1.2.2 with full-text search (FTS) + vector search (VSS), and making them callable from AI coding agents as an MCP stdio server.

## Features

- Advanced splitting and indexing of Markdown documents
- Hybrid search combining full-text search (FTS) and vector search (VSS)
- High-precision search using Japanese morphological analysis
- Search result re-ranking using CrossEncoder
- Integration with AI agents via MCP stdio server

## Usage

### Creating an Index

```bash
duckdb-hybrid-doc-search index docs/ handbook/ \
    --embedding-model cl-nagoya/ruri-v3-310m
```

### Starting the Server

```bash
duckdb-hybrid-doc-search serve --db index.duckdb
```

### Changing the Model

To use a different model, re-run the index command with a different model specified using the --embedding-model parameter.

## Docker

You can also use the Docker image:

```bash
docker pull ghcr.io/upamune/duckdb-hybrid-doc-search:latest
```

### Creating an Index with Docker

```bash
# Mount document directories and create an index
docker run -v /path/to/docs:/docs -v /path/to/output:/output \
    ghcr.io/upamune/duckdb-hybrid-doc-search:latest \
    index /docs --db /output/index.duckdb --embedding-model cl-nagoya/ruri-v3-310m
```

### Starting the MCP Server with Docker

```bash
# Mount the directory containing the index.duckdb file and start the server
docker run -v /path/to/output:/data -p 8000:8000 \
    ghcr.io/upamune/duckdb-hybrid-doc-search:latest \
    serve --db /data/index.duckdb --rerank-model cl-nagoya/ruri-v3-310m
```

### Searching Documents with Docker

```bash
# Direct search with a specific query
docker run -v /path/to/output:/data -it \
    ghcr.io/upamune/duckdb-hybrid-doc-search:latest \
    search --db /data/index.duckdb --query "your search query" --rerank-model cl-nagoya/ruri-v3-310m

# Interactive search mode (when --query is omitted)
docker run -v /path/to/output:/data -it \
    ghcr.io/upamune/duckdb-hybrid-doc-search:latest \
    search --db /data/index.duckdb --rerank-model cl-nagoya/ruri-v3-310m
```

## Development

This project uses [Task](https://taskfile.dev/) to manage build and development tasks.

### Setting Up Development Environment

```bash
task dev:setup
source .venv/bin/activate
```

### Running Tests

```bash
task test
```

### Running Linters

```bash
task lint
```

### Formatting Code

```bash
task format
```

### Creating Document Index

```bash
task run:index DIRS="docs/ handbook/"
```

### Starting the Server

```bash
task run:serve
```

### Running Search

```bash
task run:search
```

### Building and Running Docker Image

```bash
task docker:build
task docker:run CLI_ARGS="serve --db /app/index.duckdb"
```

### Listing Available Tasks

```bash
task
```

### Migration from Makefile

Previously, we used Makefile, but we've migrated to Task for more flexibility and features.
You can replace any `make` command with the corresponding `task` command.

## License

MIT
