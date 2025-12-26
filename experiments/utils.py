import ast
import os
import pkgutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import setuptools
from pydantic import BaseModel, Field, field_validator

from libs.custom_logger import getLogger

_logger = getLogger(__name__)


class LanguageModel(BaseModel):
    id: str
    base_url: str


def find_all_modules(project_path: str | Path):
    "Find all modules in a project using `setuptools.find_packages`."
    packages = setuptools.find_packages(project_path)
    all_modules: list[str] = []
    for package in packages:
        package_path = os.path.join(project_path, package.replace(".", "/"))
        modules = [module for module in pkgutil.iter_modules([package_path])]
        all_modules.extend([f"{package}.{module.name}" for module in modules if (not module.ispkg)])
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

    try:
        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        _logger.warning("\nInterrupted by user.")
        exit(1)
