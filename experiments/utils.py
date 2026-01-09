import ast
import logging
import os
import pkgutil
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import setuptools
from pydantic import BaseModel, Field, field_validator

from libs.custom_logger import getLogger

_logger = getLogger(__name__, logging.ERROR)


class LanguageModel(BaseModel):
    id: str
    base_url: str


def find_all_modules(project_name: str, project_path: str | Path):
    "Find all modules in a project using `setuptools.find_packages`."
    packages = setuptools.find_packages(project_path)
    all_modules: set[str] = set()
    for package in packages:
        package_path = os.path.join(project_path, package.replace(".", "/"))
        modules = [module for module in pkgutil.iter_modules([package_path])]
        all_modules.update([f"{package}.{module.name}" for module in modules if not module.ispkg])
    ignore_path = Path("pynguin_report") / project_name / "ignore.list"
    if ignore_path.exists() and ignore_path.is_file():
        ignore_list = ignore_path.read_text().split()
        all_modules.difference_update(ignore_list)
    return all_modules


class RunConfig(BaseModel):
    config_id: str
    argv: list[str]


class RunEntry(BaseModel):
    module_name: str
    run_config: RunConfig


class StatisticsRow(BaseModel):
    run_id: int = Field(alias="RunId")
    project_name: str = Field(alias="ProjectName")
    target_module: str = Field(alias="TargetModule")
    configuration_id: str = Field(alias="ConfigurationId")
    line_nos: int = Field(alias="LineNos")
    random_seed: int = Field(alias="RandomSeed")
    lines: int = Field(alias="Lines")
    predicates: int = Field(alias="Predicates")
    goals: int = Field(alias="Goals")
    mccabe_ast: list[int] = Field(alias="McCabeAST")
    mccabe_code_object: list[int] = Field(alias="McCabeCodeObject")
    code_objects: int = Field(alias="CodeObjects")
    accessible_objects_under_test: int = Field(alias="AccessibleObjectsUnderTest")
    llm_calls: int = Field(alias="LLMCalls")
    llm_query_time: float = Field(alias="LLMQueryTime")
    llm_stage_saved_tests: int = Field(alias="LLMStageSavedTests")
    llm_input_tokens: int = Field(alias="LLMInputTokens")
    llm_output_tokens: int = Field(alias="LLMOutputTokens")
    parsed_statements: int = Field(alias="ParsedStatements")
    parsable_statements: int = Field(alias="ParsableStatements")
    final_size: int = Field(alias="FinalSize")
    final_length: int = Field(alias="FinalLength")
    branch_coverage: float = Field(alias="BranchCoverage")
    coverage_timeline: list[float] = Field(alias="CoverageTimeline", exclude=True)

    @field_validator("mccabe_ast", "mccabe_code_object", mode="before")
    @classmethod
    def validate_mccabe(cls, v):
        return ast.literal_eval(v)

    @field_validator(
        "llm_calls",
        "llm_query_time",
        "llm_stage_saved_tests",
        "llm_input_tokens",
        "llm_output_tokens",
        "parsed_statements",
        "parsable_statements",
        mode="before",
    )
    @classmethod
    def empty_llm_stats(cls, v):
        if v == "":
            return 0
        return v


@lru_cache(maxsize=None)
def read_module_statistics(report_dir: Path):
    report_path = report_dir / "statistics.csv"

    if not report_path.exists():
        _logger.warning("statistics path '%s' does not exist!", report_path)
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


def run_deepmosa_runner(project_name: str, *argv: str):
    pwd = Path.cwd()

    # fmt: off
    cmd = [
        "docker", "run", "--rm",
        "-t",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--env-file", str(pwd / ".env"),
        "-e", f"PROJECT_NAME={project_name}",
        "-e", "UV_CACHE_DIR=/tmp/uv-cache",
        "-v", f"{pwd}/experiments/projects/{project_name}:/workspace/project:ro",
        "-v", f"{pwd}/generated_tests:/workspace/generated_tests",
        "-v", f"{pwd}/pynguin_report:/workspace/pynguin_report",
        "-v", f"{pwd}/.cache/project-deps:/workspace/.project-deps",
        "deepmosa-runner",
        "--project-path", "/workspace/project",
        "--project-name", project_name,
        *argv,
    ]
    # fmt: on

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    stderr_lines = []

    try:
        # stream stdout
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="")

        # stream + collect stderr
        if proc.stderr:
            for line in proc.stderr:
                print(line, end="", file=sys.stderr)
                stderr_lines.append(line)

        ret = proc.wait()
        if ret != 0:
            stderr = "".join(stderr_lines)
            m = re.search(r"Exception:\s*(.*)", stderr)
            if m:
                raise RuntimeError(m.group(1))
            else:
                raise subprocess.CalledProcessError(ret, cmd, stderr=stderr)

    except KeyboardInterrupt:
        proc.kill()
        raise


# fmt: off
RUN_CONFIGS = [
    RunConfig(
        config_id="dynamosa-10m", argv=[
            "--configuration-id", "dynamosa-10m",
            "--algorithm", "DYNAMOSA",
            "--maximum-search-time", "600",
        ]
    ),
    RunConfig(
        config_id="codamosa-10m-deepseek",
        argv=[
            "--configuration-id", "codamosa-10m-deepseek",
            "--algorithm", "CODAMOSA",
            "--maximum-search-time", "600",
            "--model", "deepseek-chat",
            "--base-url", "https://api.deepseek.com/beta",
        ],
    ),
    RunConfig(
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
