from libs.custom_logger import getLogger

from .. import utils

_logger = getLogger("diff")

projects = utils.find_all_projects()

diffs = []

for project in projects:
    modules = utils.find_all_modules(project)

    for module in modules:
        stat1 = utils.read_module_statistics(project, module, "deepmosa-10m-deepseek")
        stat2 = utils.read_module_statistics(project, module, "codamosa-10m-deepseek")

        if not stat1 or len(stat1) != utils.NUM_RUNS_PER_MODULE:
            continue

        if not stat2 or len(stat2) != utils.NUM_RUNS_PER_MODULE:
            continue

        cvrg1 = sum(s.branch_coverage for s in stat1) / utils.NUM_RUNS_PER_MODULE
        cvrg2 = sum(s.branch_coverage for s in stat2) / utils.NUM_RUNS_PER_MODULE

        diffs.append((module, cvrg2, cvrg1 - cvrg2, cvrg1))

diffs.sort(key=lambda e: -e[2])
betters = [e for e in diffs[:20] if e[2] > 0]
worses = [e for e in diffs[-1:-21:-1] if e[2] < 0]

_logger.info("DeepMOSA performs the best on:")
_logger.info("-----------------------")
for e in betters:
    _logger.info(
        "%s:   %s%% + %s%% = %s%%",
        f"{e[0]:>30}",
        round(100 * e[1], 1),
        round(100 * e[2], 1),
        round(100 * e[3], 1),
    )

_logger.info("")
_logger.info("")
_logger.info("Worse on:")
_logger.info("-----------------------")

for e in worses:
    _logger.info(
        "%s:   %s%% - %s%% = %s%%",
        f"{e[0]:>30}",
        round(100 * e[1], 1),
        -round(100 * e[2], 1),
        round(100 * e[3], 1),
    )
