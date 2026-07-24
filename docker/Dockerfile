FROM python:3.10-slim

ENV TZ=Asia/Ho_Chi_Minh
ENV PYTHONUNBUFFERED=1

# basic system deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /workspace

# scratch dirs for runs: the only /workspace paths where Landlock permits
# deletes (.coverage-data is needed by coverage.py's sqlite journal files)
RUN mkdir -m 0777 /workspace/run /workspace/.coverage-data

# copy source
COPY libs ./libs
COPY pynguin ./pynguin
COPY uv.lock pyproject.toml .python-version ./

# install dependencies
RUN --mount=type=cache,target=/tmp/uv-cache \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv pip install --group dev --group langchain --system -e .

# add entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/landlock_guard.py /landlock_guard.py
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
