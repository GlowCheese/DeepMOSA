import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from libs.custom_logger import getLogger

from . import utils

_logger = getLogger(__name__)


### Helper functions
# ------------------
# Random helper functions that I don't even
# know how to give a name
#


def print_table(a: list[list[str]], alg: str | None = None):
    alg = alg or "<" * len(a[0])
    mx_len = [max(len(a[i][j]) for i in range(len(a))) for j in range(len(a[0]))]
    for r in a:
        msg = ""
        for i in range(len(r)):
            msg += f"{r[i]:{alg[i]}{mx_len[i]}} "
        _logger.info(msg)


@lru_cache(maxsize=None)
def get_project_modules(project_name: str):
    base_report_path = Path("pynguin_report") / project_name
    project_path = Path("experiments/projects") / project_name

    assert project_path.exists()
    all_modules = utils.find_all_modules(project_name, project_path)

    fully_covered_modules: set[str] = set()
    for module_name in all_modules.copy():
        check = True
        for config in utils.RUN_CONFIGS:
            report_dir = base_report_path / module_name / config.config_id
            if rows := utils.read_module_statistics(report_dir):
                check &= all(r.branch_coverage > 0.99 for r in rows)
        if check:
            fully_covered_modules.add(module_name)

    return all_modules.difference(fully_covered_modules)


@lru_cache(maxsize=None)
def get_module_statistics(project_name: str, module_name: str, config_id: str):
    base_report_path = Path("pynguin_report") / project_name
    report_dir = base_report_path / module_name / config_id
    return utils.read_module_statistics(report_dir) or []


@lru_cache(maxsize=None)
def get_num_runs_and_avg_cvrg(project_name: str, module_name: str, config_id: str):
    num_runs, sum_cvrg = 0, 0
    for r in get_module_statistics(project_name, module_name, config_id):
        assert r.configuration_id == config_id
        num_runs += 1
        sum_cvrg += r.branch_coverage
    return num_runs, sum_cvrg / num_runs if num_runs else 0


@lru_cache(maxsize=None)
def does_module_have_runs(project_name: str, module_name: str):
    for run_config in utils.RUN_CONFIGS:
        stat = get_module_statistics(project_name, module_name, run_config.config_id)
        if stat:
            return True
    else:
        return False


@lru_cache(maxsize=None)
def get_project_cvrg_for_config(project_name: str, config_id: str):
    tot_cvrg, tot_goals = 0, 0
    for module_name in get_project_modules(project_name):
        rows = get_module_statistics(project_name, module_name, config_id)
        tot_goals += sum(r.goals for r in rows)
        tot_cvrg += sum(r.goals * r.branch_coverage for r in rows)
    return tot_goals, tot_cvrg / tot_goals if tot_goals != 0 else None


### Showing statistics
# --------------------
# Showing overall statistics (no param)
# or project statictics (project name as param)
#

if len(sys.argv) >= 2:
    project_name = sys.argv[1]
    all_modules = get_project_modules(project_name)

    _logger.info("==============================")
    _logger.info("Showing module statistics")
    _logger.info("")

    for module_name in all_modules:
        stat = get_module_statistics(project_name, module_name, utils.RUN_CONFIGS[0].config_id)[0]

        _logger.info("---")
        _logger.info("Module name: %s", module_name)
        _logger.info("Lines of code: %s", stat.line_nos)
        _logger.info("Goals: %s", stat.goals)
        _logger.info("Predicates: %s", stat.predicates)
        _logger.info("Accessible objects: %s", stat.accessible_objects_under_test)

    _logger.info("")
    _logger.info("")
    _logger.info("==============================")
    _logger.info("Showing coverage statistics")
    _logger.info("")

    for module_name in all_modules:
        if not does_module_have_runs(project_name, module_name):
            continue

        _logger.info("---")
        _logger.info("On module: %s", module_name)

        table = []

        for run_config in utils.RUN_CONFIGS:
            config_id = run_config.config_id
            num_runs, avg_cvrg = get_num_runs_and_avg_cvrg(project_name, module_name, config_id)
            if num_runs == 0:
                continue
            table.append([f"{config_id}:", f"{num_runs} runs,", f"{avg_cvrg:.2f} cvrg"])

        print_table(table)

else:
    # show basic projects infomation (project name, number of modules, ...)
    # show average coverage given by each config for each project

    _logger.info("==============================")
    _logger.info("Showing project statistics")
    _logger.info("")

    table = []
    total_num_modules = 0

    for base_report_path in Path("pynguin_report").iterdir():
        project_name = base_report_path.name
        num_modules = len(get_project_modules(project_name))

        total_num_modules += num_modules
        table.append([f"{project_name}:", f"{num_modules} modules"])

    table.append(["Total:", f"{total_num_modules} modules"])
    print_table(table)

    _logger.info("")
    _logger.info("")
    _logger.info("==============================")
    _logger.info("Showing branch coverage report")
    _logger.info("")

    table.clear()
    table.append(["Project", *(config.config_id[:8] for config in utils.RUN_CONFIGS)])

    tot_cvrg: dict[str, float] = defaultdict(float)
    tot_goals: dict[str, float] = defaultdict(float)

    for base_report_path in Path("pynguin_report").iterdir():
        project_name = base_report_path.name
        row = [f"{project_name}:"]
        for config in utils.RUN_CONFIGS:
            goals, cvrg = get_project_cvrg_for_config(project_name, config.config_id)
            if cvrg is None:
                break
            tot_goals[config.config_id] += goals
            tot_cvrg[config.config_id] += goals * cvrg
            row.append(f"{cvrg:.2f}")

        if len(row) != len(table[0]):
            continue
        table.append(row)

    table.append(
        [
            "Overall",
            *(
                f"{tot_cvrg[c_id] / tot_goals[c_id]:.2f}"
                for c in utils.RUN_CONFIGS
                if (c_id := c.config_id)
            ),
        ]
    )

    print_table(table, "<" + "^" * len(utils.RUN_CONFIGS))
