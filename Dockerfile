FROM python:3.12-slim-bookworm AS build
RUN pip install --no-cache-dir uv
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --locked
RUN pip install --no-cache-dir .

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends tini && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/duckdb-hybrid-doc-search /usr/local/bin/

ENTRYPOINT ["/usr/bin/tini", "--", "duckdb-hybrid-doc-search"]
