"""
This script automates test generation and qualification for Python modules using various algorithms.

Main functionalities:
- Supports multiple modes via --mode:
    - DEMO:
        • Runs test generation on specified modules.
        • For debugging and/or development purposes only.
        • Requires:
            - --modules (list of modules to test)
            - --algorithms with exactly one value (e.g., --algorithms CODAMOSA)
            - --project should remain default ('repositories/examples')
        • Does NOT perform qualification or filter modules.
    
    - QUAL:
        • Automatically discovers modules in the project and qualifies them.
        • Qualification means: can the module successfully be tested using 'MOSA'?
        • Good modules are marked; bad ones are excluded.
        • Recommended:
            - --project pointing to a custom repo (e.g., my_project)
            - Verbose flag (-v) for logging output
    
    - EXPER:
        • Runs test generation experiments on already-qualified modules.
        • Uses all algorithms specified via --algorithms.
        • Automatically checks if an experiment already exists via `check_legit()`.
        • Skips unqualified modules or duplicates.
    
Examples:

# 1. DEMO mode (test one or more specific modules with CODAMOSA)
./run.py --mode DEMO --modules my_project.module1 my_project.module2 --algorithms CODAMOSA

# 2. QUAL mode (auto-qualify all modules inside a custom project)
./run.py --mode QUAL --project my_project

# 3. EXPER mode (run 3 algorithms on already qualified modules)
./run.py --mode EXPER --project my_project --algorithms CODAMOSA DYNAMOSA DEEPMOSA

# 4. Skip cache saving (for debug or testing)
./run.py --mode QUAL --project my_project --disable-write-cache
"""

import argparse
import asyncio
import logging
import os
import sys
from enum import Enum
from pathlib import Path

from grappa import should

from app.pynguin.main import gen_tests
from app.qualify import qualify
from app.qualify.qualify import QualifyStatus
from custom_logger import getLogger
from pynguin.config import Algorithm
from pynguin.config.main import ExportStrategy

from .utils import (ALGORITHM_MAP, DEEPMOSA_VERSION, check_legit,
                    find_all_modules)

logger = getLogger(__name__)
sys.setrecursionlimit(2000)

QUALIFY_SECONDS = 60
ROOT_REPO = '../root-repo'
PROJECT_PATH = "repositories/examples"


class EvalMode(Enum):
    DEMO = "DEMO"
    QUAL = "QUAL"
    EXPER = "EXPER"


# load arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=EvalMode,
                        choices=list(EvalMode), required=True)
    parser.add_argument("--project", type=str, default=PROJECT_PATH)
    parser.add_argument("--modules", nargs="+", type=str, default=None)
    parser.add_argument("--disable-write-cache", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--algorithms", nargs="+", type=Algorithm,
                        default=[Algorithm.CODAMOSA, Algorithm.DYNAMOSA, Algorithm.DEEPMOSA])
    args = parser.parse_args()

    if args.mode == EvalMode.DEMO:
        args.project | should.equal(PROJECT_PATH)
        # args.project = os.path.join(ROOT_REPO, args.project)

        args.modules | should.not_be.none
        args.algorithms | should.have.length(1)
    else:
        args.project | should.not_be.equal(PROJECT_PATH)
        args.project = os.path.join(ROOT_REPO, args.project)

    PROJECT_PATH = args.project = str(Path(args.project).resolve(True))


def evaluation_params(uppercased_algo: str, id: int):
    algo = ALGORITHM_MAP[uppercased_algo]
    if uppercased_algo == 'DEEPMOSA':
        uppercased_algo += DEEPMOSA_VERSION
    params = {
        'project_path': PROJECT_PATH,
        'algorithm': algo[0],
        'run_id': f"{uppercased_algo}_{id}",
        'export_strategies': [ExportStrategy.PY_TEST],
        'report_dir': f"report/{uppercased_algo.lower()}"
    }
    return params


async def run_with_timeout(
    module_name: str,
    uppercased_algo: str,
    maximum_search_time: int,
    id: int,
    disable_log: bool = False,
):
    task = asyncio.create_task(gen_tests(
        module_name=module_name,
        maximum_search_time=maximum_search_time,
        **evaluation_params(uppercased_algo, id)
    ))

    timeout = maximum_search_time + 5*60

    try:
        if disable_log:
            logging.disable(logging.CRITICAL)
        result = await asyncio.wait_for(task, timeout=timeout)
        if disable_log:
            logging.disable(logging.NOTSET)
    except BaseException as e:
        if disable_log:
            logging.disable(logging.NOTSET)
        result = "Failed to generate test cases!"
        logger.exception(result)
    return result


if __name__ == "__main__":
    # Find all testable modules
    modules = find_all_modules(PROJECT_PATH)
    logger.info(f"Found {len(modules)} modules")

    modules = [
        module_name for module_name in modules
        if 'conftest' not in module_name
        and 'tests.' not in module_name
    ]

    if args.disable_write_cache:
        qualify.disable_write_cache()
    qualify.set_project_path(PROJECT_PATH)

    if "QUAL" in args.mode.value:
        for module_name in modules:
            status = qualify.get_status(module_name)
            if status != QualifyStatus.NOT_QUALIFIED:
                continue

            logger.info(f"Qualifying module {module_name}")
            result = asyncio.run(run_with_timeout(
                module_name, 'MOSA', QUALIFY_SECONDS, 0, not args.verbose
            ))
            if not isinstance(result, dict):
                if not isinstance(result, str):
                    logger.warning(f'Invalid gen_tests return value: {result}')
                status = QualifyStatus.BAD
                qualify.add_bad(module_name)
            else:
                coverage = result['test_suite_coverage_report'].line_coverage
                if coverage >= 0.999:
                    status = QualifyStatus.BAD
                    qualify.add_bad(module_name)
                else:
                    status = QualifyStatus.GOOD
                    qualify.add_good(module_name)
                    logger.info(f"Collected module {module_name}")

    filtered_modules: list[str] = []
    for module_name in modules:
        status = qualify.get_status(module_name)
        if status == QualifyStatus.GOOD:
            logger.info(f"Collected module {module_name}")
            filtered_modules.append(module_name)

    modules = filtered_modules
    if args.modules is not None:
        modules = args.modules
    logger.info(f"Collected {len(modules)} qualified modules")

    if args.mode == EvalMode.DEMO:
        for module_name in modules:
            algo_name = args.algorithms[0].value
            asyncio.run(
                gen_tests(
                    project_path=PROJECT_PATH,
                    module_name=module_name,
                    seed=0,
                    maximum_search_time=60,
                    algorithm=args.algorithms[0],
                    report_dir=f"report/{algo_name.lower()}",
                    run_id="tmp",
                    export_strategies=[ExportStrategy.PY_TEST],
                    initial_population_seeding=False
                )
            )

    elif "EXPER" in args.mode.value:
        for module_name in modules:
            for algo in args.algorithms:
                strat = algo.value
                new_id = check_legit(strat, module_name)
                if new_id is False:
                    continue

                logger.info(
                    "Testing module %s with %s",
                    module_name, strat
                )
                result = asyncio.run(run_with_timeout(
                    module_name, strat, 600, new_id
                ))

                # if isinstance(result, dict): continue
                if isinstance(result, dict):
                    exit(17)

                logger.warning(str(result))
                qualify.add_ignored(module_name)

                if qualify._cache.is_write_cache_disabled:
                    exit(0)
                exit(17)
