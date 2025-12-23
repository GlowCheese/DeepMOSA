# Deepmosa

## 1. Usage

### 1.1. Using uv venv

```
uv sync
uv add -r experiments/projects/minbpe/requirements.txt --group minbpe
uv run pynguin --project-path experiments/projects/minbpe --module-name minbpe.regex --maximum-search-time 10 --algorithm DYNAMOSA --output-path generated_tests/minbpe --report-dir pynguin-report/minbpe/minbpe.regex
```

### 1.2. Using docker

#### Create output folders:
```
mkdir generated_tests
mkdir pynguin-report
```

#### Build deepmosa image:
```
DOCKER_BUILDKIT=1 docker build -t deepmosa-runner -f docker/Dockerfile .
```

#### Run test generation:
```
docker run --rm \
  --user $(id -u):$(id -g) \
  --env-file $(pwd)/.env \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -v $(pwd)/experiments/projects/minbpe:/workspace/project:ro \
  -v $(pwd)/generated_tests:/workspace/generated_tests \
  -v $(pwd)/pynguin-report:/workspace/pynguin-report \
  deepmosa-runner \
  --project-path /workspace/project \
  --module-name minbpe.regex \
  --maximum-search-time 60 \
  --algorithm DYNAMOSA \
  --output-path /workspace/generated_tests/minbpe \
  --report-dir /workspace/pynguin-report/minbpe/minbpe.regex
```