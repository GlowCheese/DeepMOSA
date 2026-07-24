import argparse
import os
import subprocess
from pathlib import Path

import dotenv

import experiments.models
from libs.custom_logger import getLogger

from . import utils

dotenv.load_dotenv()

_logger = getLogger("experiments")


### Parsing experiment argument
# -----------------------------
# Collect experiment project or run config id
# Leave None will run all projects using each config
#

parser = argparse.ArgumentParser()

parser.add_argument("--project", type=str, default=None)
parser.add_argument("--config-id", type=str, default=None)
parser.add_argument("--env-file", type=str, default=".env")
parser.add_argument("--build-image", action="store_true")

args = parser.parse_args()
all_projects = utils.find_all_projects() if args.project is None else [args.project]
all_configs = (
    utils.RUN_CONFIGS
    if args.config_id is None
    else [c for c in utils.RUN_CONFIGS if c.config_id == args.config_id]
)
assert all_configs, f"Run config {args.config_id} does not exist!"
assert all_projects, f"Experiment project {args.project} does not exist!"

if args.build_image:
    subprocess.run(
        ["docker", "compose", "build", "runner"],
        cwd=Path.cwd(),
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
        check=True,
    )


num_completed_runs = 0
skipped_modules: list[str] = []


def run_experiment_on_project(
    project_name: str, config: experiments.models.RunConfig, env_file: str
):
    ### Recognize project
    # -------------------
    # Find the project path from BASE_PROJECT_PATH
    # corresponds to the provided project name.
    #

    project_path = utils.BASE_PROJECT_PATH / project_name

    if not project_path.exists():
        _logger.error("No project named '%s' can be found", project_name)
        exit(1)

    if not project_path.is_dir():
        _logger.error("Path '' is not a directory!", project_path)
        exit(1)

    _logger.info("Using project path: %s", project_path)

    project_report_path = utils.BASE_REPORT_PATH / project_name
    project_ignores_path = project_report_path / "ignore.list"

    ### Analyzing project
    # -------------------
    # Find the test report for each module to determine
    # which and how many times they should be run on.
    #

    all_modules = utils.find_all_modules(project_name)
    _logger.info("Found %d modules", len(all_modules))

    queue: list[experiments.models.RunEntry] = []

    for module_name in all_modules:
        report_dir = project_report_path / module_name / config.config_id
        rows = utils.read_module_statistics(project_name, module_name, config.config_id) or []
        for run_id in range(len(rows), utils.NUM_RUNS_PER_MODULE):
            # fmt: off
            docker_report_dir = Path("/workspace", *report_dir.parts[-4:])
            docker_output_path = Path("/workspace", "generated_tests", project_name, config.config_id)
            config_copy = experiments.models.RunConfig(
                config_id=config.config_id,
                argv=[
                    *config.argv,
                    "--run-id",         str(run_id),
                    "--module-name",    module_name,
                    "--seed",           str(utils.RANDOM_SEEDS[run_id]),
                    "--report-dir",     str(docker_report_dir),
                    "--output-path",    str(docker_output_path),
                ],
            )
            # fmt: on
            queue.append(
                experiments.models.RunEntry(
                    module_name=module_name,
                    run_config=config_copy,
                    log_path=report_dir / f"run-{run_id}.log",
                )
            )

    _logger.info("Number of run entries in queue: %d", len(queue))

    ### Test generation
    # -----------------
    # Run the test generation for each queue entry
    # with the prebuilt docker image
    #

    # Create output dir
    project_report_path.mkdir(parents=True, exist_ok=True)
    (Path(".cache") / "project-deps").mkdir(parents=True, exist_ok=True)
    (Path("generated_tests") / project_name).mkdir(parents=True, exist_ok=True)

    global num_completed_runs

    for run_entry in queue:
        if run_entry.module_name in skipped_modules:
            continue

        _logger.info("Running test generation for %s", run_entry.module_name)
        _logger.info("Configuration used: %s", run_entry.run_config.config_id)

        try:
            utils.run_deepmosa_runner(
                project_name, env_file, *run_entry.run_config.argv, log_path=run_entry.log_path
            )
            num_completed_runs += 1

        except Exception as e:
            if str(e) == "SUT contains nothing we can test.":
                with project_ignores_path.open("a+", encoding="UTF-8") as file:
                    file.write(run_entry.module_name + "\n")
                _logger.info(
                    "Added %s to %s", run_entry.module_name, Path(*project_ignores_path.parts[-3:])
                )
            skipped_modules.append(run_entry.module_name)
            _logger.exception("Added %s to skipped list.", run_entry.module_name)


try:
    for project_name in all_projects:
        for config in all_configs:
            run_experiment_on_project(project_name, config, args.env_file)
except KeyboardInterrupt:
    print()
finally:
    _logger.info("======================================")
    _logger.info("%s runs have been completed.", num_completed_runs)
    if skipped_modules:
        _logger.info("Skipped modules: %s", ", ".join(skipped_modules))
