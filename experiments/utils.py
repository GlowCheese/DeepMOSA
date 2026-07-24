import os
import pkgutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import setuptools

from experiments.models import RunConfig, StatisticsRow
from libs.custom_logger import getLogger

_logger = getLogger("utils")


# Some constants
NUM_RUNS_PER_MODULE = 2
RANDOM_SEEDS = [42, 1302, 1337, 2004, 2412]
BASE_REPORT_PATH = Path("pynguin_report").resolve(True)
BASE_PROJECT_PATH = Path("experiments", "projects").resolve(True)


def find_all_modules(project_name: str):
    "Find all modules in a project using `setuptools.find_packages`."

    project_path = BASE_PROJECT_PATH / project_name
    assert project_path.exists()

    packages = setuptools.find_packages(project_path)
    all_modules: set[str] = set()
    for package in packages:
        if package.startswith("tests"):
            continue
        package_path = os.path.join(project_path, package.replace(".", "/"))
        modules = [module for module in pkgutil.iter_modules([package_path])]
        all_modules.update([f"{package}.{module.name}" for module in modules if not module.ispkg])

    ignore_path = BASE_REPORT_PATH / project_name / "ignore.list"
    if ignore_path.exists() and ignore_path.is_file():
        ignored_modules = ignore_path.read_text().split()
        all_modules.difference_update(ignored_modules)
    return all_modules


def find_all_projects():
    all_projects: set[str] = set()
    for base_report_path in BASE_PROJECT_PATH.iterdir():
        project_name = base_report_path.name
        all_projects.add(project_name)
    return all_projects


def read_module_statistics(project_name: str, module_name: str, config_id: str):
    base_report_path = Path("pynguin_report") / project_name
    report_path = base_report_path / module_name / config_id / "statistics.csv"

    if not report_path.exists():
        _logger.debug("statistics path '%s' does not exist!", report_path)
        return None

    df = pd.read_csv(report_path, keep_default_na=False)

    result: list[StatisticsRow] = []

    for _, row in df.iterrows():
        d: dict[str, Any] = row.to_dict()  # type: ignore
        cvrg_timeline = []

        for k, v in list(d.items()):
            assert isinstance(k, str)
            if k.startswith("CoverageTimeline_T"):
                assert len(cvrg_timeline) + 1 == int(k[18:])
                cvrg_timeline.append(v)
                d.pop(k)

        d["CoverageTimeline"] = cvrg_timeline
        result.append(StatisticsRow(**d))

    return result


def run_deepmosa_runner(project_name: str, env_file: str, *argv: str, log_path: Path | None = None):
    # fmt: off
    cmd = [
        "docker", "compose", "run", "--rm",
        "runner", "pynguin",
        "--project-path",    "/workspace/project",
        "--project-name",    project_name,
        *argv,
    ]
    # fmt: on

    env = {
        **os.environ,
        "HOST_UID": str(os.getuid()),
        "HOST_GID": str(os.getgid()),
        "PROJECT_NAME": project_name,
        "ENV_FILE": env_file,
    }

    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="UTF-8", buffering=1)  # line-buffered

    # stderr is merged into stdout so a single reader drains both:
    # reading them sequentially deadlocks once the unread pipe fills up.
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )

    err_msg: str | None = None

    try:
        assert proc.stdout
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log_file:
                log_file.write(line)
            if line.startswith("Exception:"):
                err_msg = line[11:].strip()

        ret = proc.wait()
        if ret != 0:
            if err_msg:
                raise RuntimeError(err_msg)
            else:
                raise subprocess.CalledProcessError(ret, cmd)

    except KeyboardInterrupt:
        proc.kill()
        raise
    finally:
        if log_file:
            log_file.close()


# fmt: off
RUN_CONFIGS = [
    RunConfig(
        config_id="dynamosa-10m", argv=[
            "--configuration-id", "dynamosa-10m",
            "--algorithm", "DYNAMOSA",
            "--maximum-search-time", "600",
        ]
    ),
]

for baseline in [
    ("codamosa", "CODAMOSA", []),
    ("deepmosa", "DEEPMOSA", []),
    # ("deepmosa-sync", "DEEPMOSA", ["--async-enabled", "False"]),
    # ("deepmosa-dese-v1", "DEEPMOSA", ["--deserializer-version", "1"]),
    # ("deepmosa-unaware", "DEEPMOSA", ["--use-codamosa-seeding", "True"])
]:
    for llm_config in ["deepseek", "devstral", "gemma"]:
        config_id = f"{baseline[0]}-10m-{llm_config}"
        RUN_CONFIGS.append(
            RunConfig(
                config_id=config_id,
                argv=[
                    "--configuration-id", config_id,
                    "--algorithm", baseline[1],
                    "--maximum-search-time", "600",
                    "--llm-config-id", llm_config,
                    *baseline[2]
                ]
            )
        )
# fmt: on


def print_table(a: list[list[str]], alg: str | None = None):
    alg = alg or "<" * len(a[0])
    mx_len = [max(len(a[i][j]) for i in range(len(a))) for j in range(len(a[0]))]
    for r in a:
        msg = ""
        for i in range(len(r)):
            msg += f"{r[i]:{alg[i]}{mx_len[i]}}  "
        print(msg)


def find_all_filtered_modules(
    project_name: str, *, fully_run: bool = False, configs: list[RunConfig] = RUN_CONFIGS
):
    """Similar to utils.find_all_modules, except that it doesn't
    include modules where all baselines achieved 100% coverage."""

    all_modules = find_all_modules(project_name)

    fully_covered_modules: set[str] = set()
    for module_name in all_modules.copy():
        should_exclude = True
        for config in configs:
            rows = read_module_statistics(project_name, module_name, config.config_id)
            if rows is not None:
                should_exclude &= all(r.branch_coverage > 0.99 for r in rows)
                if fully_run:
                    should_exclude &= len(rows) != NUM_RUNS_PER_MODULE
        if should_exclude:
            fully_covered_modules.add(module_name)

    return all_modules.difference(fully_covered_modules)
