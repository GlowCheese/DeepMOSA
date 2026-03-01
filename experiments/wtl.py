import statistics
from collections import defaultdict

from scipy.stats import mannwhitneyu

from libs.custom_logger import getLogger

from . import utils

_logger = getLogger("trend")


def a12(x, y):
    wins = 0
    ties = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                wins += 1
            elif xi == yj:
                ties += 1
    return (wins + 0.5 * ties) / (len(x) * len(y))


def show_wtl(versuses: list[tuple[str, str]], branch_cov: dict[str, list[float]]):
    table = [["Comparison", "W/T/L", "U-test", "A12"]]

    for ours, baseline in versuses:
        ours_cov = branch_cov[ours]
        baseline_cov = branch_cov[baseline]

        wins = ties = loses = 0
        for c1, c2 in zip(ours_cov, baseline_cov):
            if c1 > c2:
                wins += 1
            elif c1 == c2:
                ties += 1
            else:
                loses += 1
        u, p = mannwhitneyu(ours_cov, baseline_cov, alternative="greater")
        table.append(
            [
                f"{ours} vs. {baseline}",
                f"{wins} / {ties} / {loses}",
                f"{p:.3f}",
                f"{a12(ours_cov, baseline_cov):.3f}",
            ]
        )

    utils.print_table(table)


if __name__ == "__main__":
    versuses = [
        ("deepmosa-10m-deepseek", "dynamosa-10m"),
        ("deepmosa-10m-deepseek", "codamosa-10m-deepseek"),
        ("deepmosa-10m-devstral", "dynamosa-10m"),
        ("deepmosa-10m-devstral", "codamosa-10m-devstral"),
    ]

    configs = [
        c
        for c in utils.RUN_CONFIGS
        if c.config_id
        in [
            "dynamosa-10m",
            "deepmosa-10m-deepseek",
            "codamosa-10m-deepseek",
            "deepmosa-10m-devstral",
            "codamosa-10m-devstral",
        ]
    ]

    all_modules = [
        (project, module)
        for project in utils.find_all_projects()
        for module in utils.find_all_filtered_modules(project, fully_run=True, configs=configs)
    ]

    all_modules = list(sorted(all_modules))

    branch_cov: dict[str, list[float]] = defaultdict(list)

    for config in configs:
        config_id = config.config_id
        for project, module in all_modules:
            rows = utils.read_module_statistics(project, module, config_id)
            assert rows and len(rows) == utils.NUM_RUNS_PER_MODULE
            branch_cov[config_id].append(statistics.mean([r.branch_coverage for r in rows]))

    show_wtl(versuses, branch_cov)
