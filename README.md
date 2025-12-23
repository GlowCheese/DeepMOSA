# Deepmosa

## 1. Usage

### 1.1. Using uv venv

```
uv sync
uv add -r experiments/projects/minbpe/requirements.txt --group minbpe
uv run pynguin --project-path experiments/projects/minbpe --module-name minbpe.regex --maximum-search-time 20 --algorithm DYNAMOSA --output-path generated_tests/minbpe --report-dir pynguin_report/minbpe/minbpe.regex
```

### 1.2. Using docker

#### Create output folders:
```
mkdir -p pynguin_report
mkdir -p generated_tests
mkdir -p .cache/project-deps
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
  -e PROJECT_NAME={{project_name}} \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -v $(pwd)/experiments/projects/minbpe:/workspace/project:ro \
  -v $(pwd)/generated_tests:/workspace/generated_tests \
  -v $(pwd)/pynguin_report:/workspace/pynguin_report \
  -v $(pwd)/.cache/project-deps:/workspace/.project-deps \
  deepmosa-runner \
  --project-name minbpe \
  --project-path /workspace/project \
  --module-name minbpe.regex \
  --maximum-search-time 20 \
  --algorithm DYNAMOSA \
  --output-path /workspace/generated_tests/minbpe \
  --report-dir /workspace/pynguin_report/minbpe/minbpe.regex
```