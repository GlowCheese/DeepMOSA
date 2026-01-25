import sys
from collections import defaultdict
from functools import lru_cache
from typing import Any

from libs.custom_logger import getLogger

from . import utils

_logger = getLogger("export")


@lru_cache(maxsize=None)
def find_all_modules(project_name: str, *, fully_run: bool = False):
    """Similar to utils.find_all_modules, except that it doesn't
    include modules where all baselines achieved 100% coverage."""

    all_modules = utils.find_all_modules(project_name)

    fully_covered_modules: set[str] = set()
    for module_name in all_modules.copy():
        should_exclude = True
        for config in utils.RUN_CONFIGS:
            rows = utils.read_module_statistics(project_name, module_name, config.config_id)
            if rows is not None:
                should_exclude &= all(r.branch_coverage > 0.99 for r in rows)
                if fully_run:
                    should_exclude &= len(rows) != utils.NUM_RUNS_PER_MODULE
        if should_exclude:
            fully_covered_modules.add(module_name)

    return all_modules.difference(fully_covered_modules)


def remove_10m(name: str):
    parts = name.partition("-10m")
    return parts[0] + parts[2]


OVERALL = "overall"


def print_metric_table(
    label: str,
    data: dict[tuple[str, str], Any],
    formatter=lambda v: str(v),
):
    projects = set(k[0] for k in data.keys() if k[0] != OVERALL)

    metric_table = []
    metric_table.append(["", *projects, "Overall"])

    for llm_id in set(c.config_id.partition("10m-")[2] for c in utils.RUN_CONFIGS):
        llm_id = llm_id or "-10m"
        configs = [c for c in utils.RUN_CONFIGS if c.config_id.endswith(llm_id)]
        if llm_id != "-10m":
            metric_table.append([f"On {llm_id} model:"] + [""] * (len(projects) + 1))

        for run_config in configs:
            config_id = run_config.config_id
            row = [f"{config_id}:"]
            for project_name in projects:
                row.append(formatter(data[(project_name, config_id)]))
            row.append(formatter(data[(OVERALL, config_id)]))
            metric_table.append(row)

        metric_table.append([""] * (len(projects) + 2))

    _logger.info("--- %s", label)
    utils.print_table(metric_table, "<" + ">" * len(data.keys()))
    _logger.info("")


### Showing statistics
# --------------------
# Showing overall statistics (no param)
# or project statictics (project name as param)
#

if len(sys.argv) >= 2:
    project_name = sys.argv[1]
    all_modules = find_all_modules(project_name) if len(sys.argv) == 2 else [sys.argv[2]]

    _logger.info("==============================")
    _logger.info("Showing module statistics")
    _logger.info("")

    for module_name in all_modules:
        stat = utils.read_module_statistics(
            project_name, module_name, utils.RUN_CONFIGS[0].config_id
        )[0]  # type: ignore

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
        _logger.info("---")
        _logger.info("On module: %s", module_name)

        table = []

        for run_config in utils.RUN_CONFIGS:
            config_id = run_config.config_id

            rows = utils.read_module_statistics(project_name, module_name, config_id) or []
            if (num_runs := len(rows)) == 0:
                continue

            avg_cvrg = sum(r.branch_coverage for r in rows) / num_runs
            cvrgs = ", ".join(f"{100 * r.branch_coverage:.1f}" for r in rows)
            table.append(
                [
                    f"{config_id}:",
                    f"{num_runs} runs,",
                    f"{100 * avg_cvrg:.1f} cvrg   ({cvrgs})",
                ]
            )

        utils.print_table(table)

else:
    # show basic projects infomation (project name, number of modules, ...)
    # show average coverage given by each config for each project

    _logger.info("==============================")
    _logger.info("Showing project statistics")
    _logger.info("")

    table = []
    total_num_modules = 0
    total_num_focals = 0
    all_projects = utils.find_all_projects()
    all_projects = list(sorted(all_projects))

    for project_name in all_projects:
        modules = find_all_modules(project_name, fully_run=True)
        num_modules = len(modules)

        total_num_modules += num_modules
        num_focals = 0
        for module in modules:
            stat = utils.read_module_statistics(
                project_name, module, utils.RUN_CONFIGS[0].config_id
            )
            assert stat
            num_focals += stat[0].accessible_objects_under_test

        total_num_focals += num_focals
        table.append([f"{project_name}:", f"{num_modules} modules", f"{num_focals} callables"])

    table.append(["Total:", f"{total_num_modules} modules", f"{total_num_focals} callables"])
    utils.print_table(table)

    # tracking metrics for each pair [project, config_id]
    metrics: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))

    for project_name in all_projects:
        modules = find_all_modules(project_name, fully_run=True)
        for run_config in utils.RUN_CONFIGS:
            config_id = run_config.config_id
            pair = (project_name, config_id)
            for module in modules:
                rows = utils.read_module_statistics(project_name, module, config_id) or []
                for r in rows:
                    metrics["num_runs"][pair] += 1
                    metrics["goals"][pair] += r.goals
                    metrics["sum_cvrg"][pair] += r.goals * r.branch_coverage
                    metrics["llm_calls"][pair] += r.llm_calls
                    metrics["query_time"][pair] += r.llm_query_time
                    metrics["input_toks"][pair] += r.llm_input_tokens
                    metrics["output_toks"][pair] += r.llm_output_tokens
            if metrics["num_runs"][pair] != len(modules) * utils.NUM_RUNS_PER_MODULE:
                # metrics["avg_cvrg"][pair] = (
                #     metrics["num_runs"][pair] - len(modules) * utils.NUM_RUNS_PER_MODULE
                # )
                metrics["avg_cvrg"][pair] = -1
                metrics["goals"][pair] = metrics["sum_cvrg"][pair] = -1
                metrics["llm_calls"][pair] = metrics["query_time"][pair] = -1
                metrics["input_toks"][pair] = metrics["output_toks"][pair] = -1
            elif metrics["goals"][pair]:
                metrics["avg_cvrg"][pair] = metrics["sum_cvrg"][pair] / metrics["goals"][pair]

    for run_config in utils.RUN_CONFIGS:
        config_id = run_config.config_id
        pair = (OVERALL, config_id)
        for met in metrics.keys():
            metrics[met][pair] = 0
            for project_name in all_projects:
                v = metrics[met][(project_name, config_id)]
                if v < 0:
                    metrics[met][pair] = -1
                    break
                metrics[met][pair] += v
        if metrics["goals"][pair] and metrics["sum_cvrg"][pair] != -1:
            metrics["avg_cvrg"][pair] = metrics["sum_cvrg"][pair] / metrics["goals"][pair]

    _logger.info("")
    _logger.info("")
    _logger.info("=======================")
    _logger.info("Showing configs metrics")
    _logger.info("")

    print_metric_table(
        "Branch coverage:",
        metrics["avg_cvrg"],
        lambda v: "-" if v == -1 else f"{100 * v:.1f}",
    )
    print_metric_table(
        "Num LLM calls:",
        metrics["llm_calls"],
        lambda v: "-" if v == -1 else f"{v:.0f}",
    )
    print_metric_table(
        "LLM query time:",
        metrics["query_time"],
        lambda v: "-" if v == -1 else f"{v:.0f}",
    )
    print_metric_table(
        "LLM input tokens:",
        metrics["input_toks"],
        lambda v: "-" if v == -1 else f"{v:.0f}",
    )
    print_metric_table(
        "LLM output tokens:",
        metrics["output_toks"],
        lambda v: "-" if v == -1 else f"{v:.0f}",
    )

    _logger.info("")
    _logger.info("")
    _logger.info("==============================")
    _logger.info("Consistency check")
    _logger.info("")

    total_requires = 0
    total_completed = 0

    for project_name in all_projects:
        modules = find_all_modules(project_name)
        total_requires += len(modules) * len(utils.RUN_CONFIGS) * utils.NUM_RUNS_PER_MODULE

        for module_name in modules:
            inconsistencies = []
            for run_config in utils.RUN_CONFIGS:
                config_id = run_config.config_id
                rows = utils.read_module_statistics(project_name, module_name, config_id) or []
                num_runs = len(rows)
                total_completed += num_runs
                if num_runs != utils.NUM_RUNS_PER_MODULE:
                    # if num_runs != 2 and num_runs != 0:
                    inconsistencies.append((config_id, num_runs))
            if inconsistencies:
                _logger.info(
                    "On module %s: %s",
                    module_name,
                    ", ".join(
                        f"{config_id} ({num_runs} runs)" for config_id, num_runs in inconsistencies
                    ),
                )

    _logger.info(
        "Completion: %s / %s (%s %%)",
        total_completed,
        total_requires,
        round(100 * total_completed / total_requires, 1),
    )
