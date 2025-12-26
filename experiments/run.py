import os
import subprocess
import sys
from pathlib import Path

import dotenv

from libs.custom_logger import getLogger

from . import utils

# import RunEntry, RunConfig, find_all_modules, LanguageModel, read_module_statistics


dotenv.load_dotenv()

_logger = getLogger("experiments")


NUM_RUNS_PER_MODULE = 2
RANDOM_SEEDS = [42, 1302, 1337, 2004, 2412]

# fmt: off
RUN_CONFIGS = [
    utils.RunConfig(
        config_id="dynamosa-10m", argv=[
            "--configuration-id", "dynamosa-10m",
            "--algorithm", "DYNAMOSA",
            "--maximum-search-time", "600",
        ]
    ),
    utils.RunConfig(
        config_id="codamosa-10m-deepseek",
        argv=[
            "--configuration-id", "codamosa-10m-deepseek",
            "--algorithm", "CODAMOSA",
            "--maximum-search-time", "600",
            "--model", "deepseek-chat",
            "--base-url", "https://api.deepseek.com/beta",
        ],
    ),
    utils.RunConfig(
        config_id="deepmosa-10m-deepseek",
        argv=[
            "--configuration-id", "deepmosa-10m-deepseek",
            "--algorithm", "DEEPMOSA",
            "--maximum-search-time", "600",
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
if len(sys.argv) >= 3:
    all_modules = [m for m in all_modules if m in sys.argv[2]]

fixed_algorithm = None
if len(sys.argv) >= 4:
    fixed_algorithm = sys.argv[3]
    _logger.info("Using fixed algorithm: %s", fixed_algorithm)

_logger.info("Found %d modules", len(all_modules))

queue: list[utils.RunEntry] = []

for module_name in all_modules:
    for config in RUN_CONFIGS:
        report_dir = Path("pynguin_report") / project_name / module_name / config.config_id
        rows = utils.read_module_statistics(report_dir) or []
        for run_id in range(len(rows), NUM_RUNS_PER_MODULE):
            if fixed_algorithm is not None:
                if fixed_algorithm.lower() not in config.config_id.lower():
                    continue
            config_copy = utils.RunConfig(
                config_id=config.config_id,
                argv=config.argv.copy(),
            )
            # fmt: off
            config_copy.argv.extend(
                [
                    "--run-id", str(run_id),
                    "--module-name", module_name,
                    "--seed", str(RANDOM_SEEDS[run_id]),
                    "--report-dir",
                    f"/workspace/{report_dir}",
                    "--output-path",
                    f"/workspace/generated_tests/{project_name}/{config.config_id}",
                ]
            )
            # fmt: on
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
(Path(".cache") / "project-deps").mkdir(parents=True, exist_ok=True)
(Path("pynguin_report") / project_name).mkdir(parents=True, exist_ok=True)
(Path("generated_tests") / project_name).mkdir(parents=True, exist_ok=True)

for run_entry in queue:
    _logger.info("Running test generation for %s", run_entry.module_name)
    _logger.info("Configuration used: %s", run_entry.run_config.config_id)
    utils.run_deepmosa_runner(project_name, *run_entry.run_config.argv)
