# DeepMOSA

DeepMOSA is an automated unit testing method that leverages Large Language Model (LLM) as a search assistant to improve test generation. Built upon CodaMOSA and DynaMOSA, our approach introduces several key enhancements to address many limitations and further boost branch coverage.

Combining evolutionary search with LLM-augmented guidance, DeepMOSA provides a more adaptive and effective method for automated unit test generation.

## Setup

Clone the DeepMOSA repository and cd into it:
```
git clone ...
cd deepmosa
```

Then, create a `.env` file with the following content:

```
PYNGUIN_DANGER_AWARE=1
OPENAI_API_KEY=sk-...
```

`PYNGUIN_DANGER_AWARE=1` is required -- pynguin refuses to run without it. This flag is your way of acknowledging that you understand the risks of executing code with random inputs.

`OPENAI_API_KEY` is the API key for an OpenAI-compatible endpoint, which is used by LLM-assisted algorithms (e.g. CodaMOSA and
DeepMOSA) during test generation.

## Usage

Assuming we need to generate unit tests for module `src.report` under project at `./sample_project`.

There are two methods to do this.

### 1. Using `uv` (runs on host)

This is a simpler method which only requires installing [uv package manager](https://docs.astral.sh/uv). Pynguin executes the SUT during search, so the code runs **directly on your
machine**. It is faster to set up at the cost of no isolation, so only use this for code you trust.

```bash
# create venv + install pynguin
uv venv
uv pip install -e .

# install target project's dependencies
uv pip install -r sample_project/requirements.txt

# generate tests
uv run pynguin \
  --project-path ./sample_project \
  --project-name sample_project --module-name src.report \
  --algorithm DEEPMOSA \
  --model deepseek-chat --base-url https://api.deepseek.com \
  --maximum-search-time 60 \
  --output-path generated_tests/sample_project \
  --report-dir pynguin_report/sample_project/src.report
```

### 2. Using docker (sandboxed)

This requires [docker engine](https://docs.docker.com/engine) to be installed in order to run Pynguin inside a container.

```bash
# build the image (one-time)
docker compose build runner

# define test target
PROJECT_NAME=sample_project
MODULE_NAME=src.report

# create output folders on host
mkdir -p .cache/project-deps
mkdir -p generated_tests/$PROJECT_NAME
mkdir -p pynguin_report/$PROJECT_NAME/$MODULE_NAME

# generate tests
HOST_UID=$(id -u) HOST_GID=$(id -g) \
PROJECT_PATH=./sample_project \
PROJECT_NAME=$PROJECT_NAME \
docker compose run --rm runner pynguin \
  --project-path /workspace/project \
  --project-name $PROJECT_NAME --module-name $MODULE_NAME \
  --algorithm DEEPMOSA \
  --model deepseek-chat --base-url https://api.deepseek.com \
  --maximum-search-time 60 \
  --output-path /workspace/generated_tests/$PROJECT_NAME \
  --report-dir /workspace/pynguin_report/$PROJECT_NAME/$MODULE_NAME
```

## What this does

Both methods run the test generation with the **DeepMOSA** algorithm (`--algorithm DEEPMOSA`) using the **deepseek-chat** model. The generation runs for at most 60 seconds (`--maximum-search-time 60`), or stops as soon as 100% branch coverage is achieved.

To use a different LLM, change `--model` and `--base-url` arguments. For non-LLM algorithms like DynaMOSA, you can drop those flags entirely. Make sure `OPENAI_API_KEY` in `.env` matches the provider at the chosen base URL. Examples:

| Provider  | `--base-url`                   | Example `--model`     |
|-----------|------------------------------- |-----------------------|
| DeepSeek  | `https://api.deepseek.com`     | `deepseek-chat`       |
| Anthropic | `https://api.anthropic.com/v1` | `claude-haiku-4-5`    |
| Mistral   | `mistralai` *                  | `devstral-2512`       |

* mistralai uses the [mistralai SDK](https://pypi.org/project/mistralai) directly instead of an HTTP URL.

## Output

As configured above, the generated test suite for the target module is created in `generated_tests/sample_project/`.

Additionally, the `pynguin_report/sample_project/report/` folder contains several reports about the run, including:

- cov_report.html: Coverage summary of the run (branch/line coverage, with annotations of covered and uncovered lines/branches).
- pynguin-config.txt: The Pynguin configuration arguments used for this run.
- statistics.csv: Detailed per-run metrics, such as TargetModule, LineCoverage, BranchCoverage, LLMCalls, LLMInputTokens, and more.
