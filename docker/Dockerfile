FROM python:3.10-slim

ENV TZ=Asia/Ho_Chi_Minh

# basic system deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /workspace

# copy source
COPY libs ./libs
COPY pynguin ./pynguin
COPY uv.lock pyproject.toml .python-version ./

# install dependencies
RUN --mount=type=cache,target=/tmp/uv-cache \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv pip install --system -e .

# add entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
