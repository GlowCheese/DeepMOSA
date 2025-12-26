import sys
from collections import defaultdict
from pathlib import Path

from libs.custom_logger import getLogger

from . import utils

_logger = getLogger(__name__)


project_name = sys.argv[1]
project_path = Path("experiments/projects") / project_name

assert project_path.exists()

all_modules = utils.find_all_modules(project_path)

_logger.info("==============================")
_logger.info("Showing module statistics")
_logger.info("")

_num_runs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
_sum_cvrg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

for module_name in all_modules:
    rows = utils.read_module_statistics(project_name, module_name)
    if not rows:
        continue

    for r in rows:
        _num_runs[module_name][r.configuration_id] += 1
        _sum_cvrg[module_name][r.configuration_id] += r.branch_coverage

    _logger.info("---")
    _logger.info("Module name: %s", module_name)
    _logger.info("Lines of code: %s", rows[0].line_nos)
    _logger.info("Predicates: %s", rows[0].predicates)
    _logger.info("Accessible objects: %s", rows[0].accessible_objects_under_test)


_logger.info("")
_logger.info("")
_logger.info("")
_logger.info("==============================")
_logger.info("Showing coverage statistics")
_logger.info("")

for module_name in all_modules:
    _logger.info("")
    _logger.info("---")
    _logger.info("On module: %s", module_name)

    for config_id in _num_runs[module_name]:
        num_runs = _num_runs[module_name][config_id]
        sum_cvrg = _sum_cvrg[module_name][config_id]

        _logger.info("%s: %s runs, %s cvrg", config_id, num_runs, round(sum_cvrg / num_runs, 2))
