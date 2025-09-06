import os
import time
import asyncio

from typing import List
from pathlib import Path

from app.temp import dese
from vendor.custom_logger import getLogger

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
):
    if project_name is None:
        project_name = project_path.split('/')[-1]

    project_path = str(Path(project_path).resolve(True))

    conf = Configuration(
        module_name=module_name,
        project_path=project_path,
        algorithm=algorithm,
        statistics_output=StatisticsOutputConfiguration(
            report_dir=report_dir,
            run_id=run_id,
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
