from __future__ import annotations

import sys
import asyncio
import simple_parsing

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pynguin.environ as environ
import pynguin.config as config

from pynguin.globl import Globl
from vendor.custom_logger import getLogger
from pynguin.__version__ import __version__
from app.pynguin.generator import run_pynguin, ReturnCode

if TYPE_CHECKING:
    import argparse

_logger = getLogger(__name__)


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = simple_parsing.ArgumentParser(
        add_option_string_dash_variants=simple_parsing.DashVariant.UNDERSCORE_AND_DASH,
        description="Pynguin is an automatic unit test generation framework for Python",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    parser.add_argument(
        "--log-file",
        "--log_file",
        help="Path to an optional log file.",
        type=Path,
    )
    parser.add_arguments(config.Configuration, dest="config")

    return parser


def _expand_arguments_if_necessary(arguments: list[str]) -> list[str]:
    """Expand command-line arguments, if necessary.

    This is a hacky way to pass comma separated output variables.  The reason to have
    this is an issue with the automatically-generated bash scripts for Pynguin cluster
    execution, for which I am not able to solve the (I assume) globbing issues.  This
    function allows to provide the output variables either separated by spaces or by
    commas, which works as a work-around for the aforementioned problem.

    This function replaces the commas for the ``--output-variables`` parameter and
    the ``--coverage-metrics`` by spaces that can then be handled by the argument-
    parsing code.

    Args:
        arguments: The list of command-line arguments
    Returns:
        The (potentially) processed list of command-line arguments
    """
    if (
        "--output_variables" not in arguments
        and "--output-variables" not in arguments
        and "--coverage_metrics" not in arguments
        and "--coverage-metrics" not in arguments
    ):
        return arguments
    if "--output_variables" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--output_variables")
    elif "--output-variables" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--output-variables")

    if "--coverage_metrics" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--coverage_metrics")
    elif "--coverage-metrics" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--coverage-metrics")
    return arguments


def _parse_comma_separated_option(arguments: list[str], option: str) -> list[str]:
    index = arguments.index(option)
    if "," not in arguments[index + 1]:
        return arguments
    variables = arguments[index + 1].split(",")
    return arguments[: index + 1] + variables + arguments[index + 2 :]


def _setup_output_path(output_path: str) -> None:
    path = Path(output_path).resolve()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


async def async_main(argv: list[str] | None = None):
    """Entry point for the CLI of the Pynguin automatic unit test generation framework.

    This method behaves like a standard UNIX command-line application, i.e.,
    the return value `0` signals a successful execution.  Any other return value
    signals some errors.  This is, e.g., the case if the framework was not able
    to generate one successfully running test case for the class under test.

    Args:
        argv: List of command-line arguments

    Returns:
        An integer representing the success of the program run.  0 means
        success, all non-zero exit codes indicate errors.
    """
    if environ.PYNGUIN_DANGER_AWARE is None:
        print(  # noqa: T201
            "Environment variable PYNGUIN_DANGER_AWARE not set.",
            "Aborting to avoid harming your system.",
            "Please refer to the documentation",
            "(https://pynguin.readthedocs.io/en/latest/user/quickstart.html)",
            "to see why this happens and what you must do to prevent it.",
            sep='\n' 
        )
        return -1

    if argv is None:
        argv = sys.argv
    if len(argv) <= 1:
        argv.append("--help")
    argv = _expand_arguments_if_necessary(argv[1:])

    argument_parser = _create_argument_parser()
    parsed = argument_parser.parse_args(argv)

    conf = cast(config.Configuration, parsed.config)

    _setup_output_path(conf.test_case_output.output_path)

    if conf.statistics_output.project_name == "":
        conf.statistics_output.project_name = conf.project_path.split('/')[-1]

    conf.project_path = str(Path(conf.project_path).resolve(True))

    try:
        Globl.set_conf(conf)
        result = await run_pynguin()
        if result == ReturnCode.SETUP_FAILED:
            result = "Failed: result is ReturnCode.SETUP_FAILED!"
    except Exception as e:
        result = f"Failed to generate test cases: {type(e)}\n {e.__traceback__}"
    finally:
        Globl.reset()

    return result


def main():
    asyncio.run(async_main())