import os
import dotenv
from pathlib import Path

import sys
import subprocess

from libs.custom_logger import getLogger
from . import utils
# import RunEntry, RunConfig, find_all_modules, LanguageModel, read_module_statistics


dotenv.load_dotenv()

_logger = getLogger("experiments")


NUM_RUNS_PER_MODULE = 3
RANDOM_SEEDS = [42, 1302, 1337, 2004, 2412]

# fmt: off
RUN_CONFIGS = [
    utils.RunConfig(
        config_id="dynamosa-30s", argv=[
            "--configuration-id", "dynamosa-30s",
            "--algorithm", "DYNAMOSA",
            "--maximum-search-time", "30",
        ]
    ),
    utils.RunConfig(
        config_id="codamosa-30s-deepseek",
        argv=[
            "--configuration-id", "codamosa-30s-deepseek",
            "--algorithm", "CODAMOSA",
            "--maximum-search-time", "30",
            "--model", "deepseek-chat",
            "--base-url", "https://api.deepseek.com/beta",
        ],
    ),
    utils.RunConfig(
        config_id="deepmosa-30s-deepseek",
        argv=[
            "--configuration-id", "deepmosa-30s-deepseek",
            "--algorithm", "DEEPMOSA",
            "--maximum-search-time", "30",
            "--model", "deepseek-chat",
            "--base-url", "https://api.deepseek.com",
        ],
    ),
]
# fmt: on


### Recognize project
# -------------------
# Find the project path from experiments/projects
# corresponds to the provided project name.
#

BASE_PROJECT_PATH = Path("experiments/projects").resolve(True)

try:
    project_name = sys.argv[1]
except IndexError:
    _logger.error("Please provide a project name when running experiments!")
    _logger.error("Example:\tpython -m experiments.run minbpe")
    exit(1)

project_path = BASE_PROJECT_PATH / project_name

if not project_path.exists():
    _logger.error("No project named '%s' can be found", project_name)
    exit(1)

if not project_path.is_dir():
    _logger.error("Path '' is not a directory!", project_path)
    exit(1)

_logger.info("Using project path: %s", project_path)


### Analyzing project
# -------------------
# Find the test report for each module to determine
# which and how many times they should be run on.
#

all_modules = utils.find_all_modules(project_path)
_logger.info("Found %d modules", len(all_modules))

queue: list[utils.RunEntry] = []

for module_name in all_modules:
    rows = utils.read_module_statistics("minbpe", module_name) or []

    for config in RUN_CONFIGS:
        num_runs = NUM_RUNS_PER_MODULE - sum(
            [1 for r in rows if r.configuration_id == config.config_id]
        )
        for _, random_seed in zip(range(num_runs), RANDOM_SEEDS):
            config_copy = config.model_copy()
            config_copy.argv.extend(
                [
                    "--module-name",
                    module_name,
                    "--seed",
                    str(random_seed),
                    "--report-dir",
                    f"/workspace/pynguin_report/{project_name}/{module_name}",
                ]
            )
            queue.append(utils.RunEntry(module_name=module_name, run_config=config_copy))

_logger.info("Number of run entries in queue: %d", len(queue))


### Test generation
# -----------------
# Build docker image and run the
# test generation for each queue entry.
#

_logger.info("Building docker image...")
__env = os.environ.copy()
__env["DOCKER_BUILDKIT"] = "1"
subprocess.run(
    ["docker", "build", "-t", "deepmosa-runner", "-f", "docker/Dockerfile", "."],
    cwd=Path.cwd(),
    env=__env,
    check=True,
)

# Create output dir
(Path("generated_tests") / project_name).mkdir(parents=True, exist_ok=True)
(Path("pynguin_report") / project_name).mkdir(parents=True, exist_ok=True)

for run_entry in queue:
    _logger.info("Running test generation for %s", run_entry.module_name)
    _logger.info("Configuration used: %s", run_entry.run_config.config_id)
    utils.run_deepmosa_runner(project_name, *run_entry.run_config.argv)
