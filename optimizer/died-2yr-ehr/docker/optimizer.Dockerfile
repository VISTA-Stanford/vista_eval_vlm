# The optimizer's image: the engine's isolation image (isolation/Dockerfile) plus python3, so the
# optimizer can run the program on a few examples before it finishes. Give the optimizer the same
# packages the worker image has, or it will write code it cannot test. This image has no pip: add
# python3-pip to the apt line, COPY the worker's requirements.txt (one lock for both images) and
# `pip3 install --no-cache-dir --break-system-packages -r` it. Debian bookworm's python3 is 3.11, like the worker.
FROM node:22-bookworm-slim@sha256:43ac6c60b8f89723f746e8a92ce91abd5017e627ce1ddfe4238355d3a30b772c

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates python3 python3-pip ripgrep \
    && rm -rf /var/lib/apt/lists/*
# The patients' sessions run on this image too: pandas, pyarrow and duckdb to read the record's parquet files;
# grep (in the base) and ripgrep for notes.txt. No CT viewer (died-2yr-ct has one).
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt && rm /tmp/requirements.txt

# The image's tag must name this version (name:<version>); the engine checks the CLI inside matches.
ARG CLAUDE_CODE_VERSION=2.1.295
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

WORKDIR /workspace/program
RUN chown 1000:1000 /workspace/program
USER 1000:1000
