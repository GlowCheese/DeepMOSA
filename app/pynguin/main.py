import os
import time
import asyncio

from typing import List
from pathlib import Path

from app.temp import dese
from custom_logger import getLogger

from pynguin.config.main import ExportStrategy
from pynguin.globl import Globl
from pynguin.config import (
    Algorithm,
    Configuration,
    SeedingConfiguration,
    StoppingConfiguration,
    TestCaseOutputConfiguration,
    StatisticsOutputConfiguration
)
from pynguin.runtimevar import RuntimeVariable
from . import generator


logger = getLogger(__name__)


async def gen_tests(
    *,
    project_path: str,
    module_name: str,
    maximum_search_time: int,
    algorithm: Algorithm,
    export_strategies: List[ExportStrategy],
    seed: int = None,
    run_id: str = "",
    project_name: str = None,
    report_dir: str = "pynguin-report",
    initial_population_seeding: bool = False
):
    if project_name is None:
        project_name = project_path.split('/')[-1]

    try:
        project_path = Path(project_path)
        module_path = Path(module_name.replace('.', '/') + '.py')
        module_path = project_path / module_path
        module_path = module_path.resolve(True)
        assert module_path.is_file()
    except (FileNotFoundError, AssertionError):
        raise Exception(f"Invalid module_path: `{module_path}`")

    if seed is None:
        seed = time.time_ns()
    project_path = str(project_path)  # convert PosixPath to str
    tests_output = os.path.join(project_path, "generated_tests")

    if not initial_population_seeding:
        initial_population_data = ""
    else:
        tmp = module_name.replace('.', '_')
        initial_population_data = os.path.join(tests_output, f"test_{tmp}.py")

    allow_expandable_cluster = algorithm in [Algorithm.CODAMOSA, Algorithm.DEEPMOSA]

    conf = Configuration(
        module_name=module_name,
        module_path=module_path,
        project_path=project_path,
        test_case_output=TestCaseOutputConfiguration(
            output_path=tests_output,
            export_strategies=export_strategies
        ),
        algorithm=algorithm,
        seeding=SeedingConfiguration(
            seed=seed,
            initial_population_seeding=initial_population_seeding,
            initial_population_data=initial_population_data,
            allow_expandable_cluster=allow_expandable_cluster
        ),
        statistics_output=StatisticsOutputConfiguration(
            report_dir=report_dir,
            run_id=run_id,
            output_variables=[
                RuntimeVariable.RunId,
                RuntimeVariable.ProjectName,
                RuntimeVariable.TargetModule,
                RuntimeVariable.LineNos,
                RuntimeVariable.CodeObjects,
                RuntimeVariable.Lines,
                RuntimeVariable.Predicates,
                RuntimeVariable.LineCoverage,
                RuntimeVariable.BranchCoverage,
                RuntimeVariable.LLMCalls,
                RuntimeVariable.LLMQueryTime,
                RuntimeVariable.LLMStageSavedTests,
                RuntimeVariable.LLMInputTokens,
                RuntimeVariable.LLMOutputTokens,
                RuntimeVariable.ParsedStatements,
                RuntimeVariable.ParsableStatements,
                RuntimeVariable.FinalLength,
                RuntimeVariable.FinalSize,
                RuntimeVariable.CoverageTimeline,
            ],
            project_name=project_name
        ),
        stopping=StoppingConfiguration(maximum_search_time=maximum_search_time)
    )

    try:
        Globl.set_conf(conf)
        return await _gen_tests()
    finally:
        Globl.reset()


async def _gen_tests():
    try:
        # result = await dese.run_pynguin()
        result = await generator.run_pynguin()
    except Exception as e:
        result = "Failed to generate test cases!"
        logger.exception(result)

    if result == generator.ReturnCode.SETUP_FAILED:
        result = "Failed: result is ReturnCode.SETUP_FAILED!"

    return result
