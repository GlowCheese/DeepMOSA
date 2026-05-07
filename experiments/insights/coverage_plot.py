import os

import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel

from libs.custom_logger import getLogger

from .. import utils

_logger = getLogger("trend")


SECS = 600


class PltStyle(BaseModel):
    label: str
    linestyle: str
    marker: str | None


config_styles: dict[str, PltStyle] = {
    "dynamosa-10m": PltStyle(label="DynaMOSA", marker=None, linestyle="dotted"),
    "codamosa-10m-deepseek": PltStyle(label="CodaMOSA", marker="X", linestyle="solid"),
    "deepmosa-10m-deepseek": PltStyle(label="DeepMOSA (ours)", marker=None, linestyle="solid"),
}


save_base = "pynguin_report/cvrg_timeline"
os.makedirs(os.path.dirname(save_base), exist_ok=True)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5))


project_modules: dict[str, list[str]] = {}
configs = [c for c in utils.RUN_CONFIGS if c.config_id in config_styles.keys()]


tot_goals = 0
for project in utils.find_all_projects():
    project_modules[project] = list(utils.find_all_filtered_modules(project, configs=configs))
    for module in project_modules[project]:
        rows = utils.read_module_statistics(project, module, configs[0].config_id)
        assert rows
        tot_goals += sum(r.goals for r in rows)

for config in configs:
    aggre = [0.0] * SECS
    config_id = config.config_id

    for project in utils.find_all_projects():
        for module in project_modules[project]:
            rows = utils.read_module_statistics(project, module, config_id)
            assert rows
            for r in rows:
                assert len(r.coverage_timeline) == SECS
                for i in range(SECS):
                    aggre[i] += r.coverage_timeline[i] * r.goals

    for i in range(SECS):
        aggre[i] = 100 * aggre[i] / tot_goals

    sns.lineplot(
        x=range(SECS),
        y=aggre,
        label=config_styles[config_id].label,
        color="black",
        marker=config_styles[config_id].marker,
        linestyle=config_styles[config_id].linestyle,
        markevery=70,
        markersize=7,
        linewidth=2,
    )

plt.xlim(right=600)
plt.xlabel("Time (seconds)", fontsize=22, labelpad=12)
plt.ylabel("Branch coverage (%)", fontsize=22, labelpad=12)
plt.legend(fontsize=21)

plt.xticks(fontsize=19)
plt.yticks(range(20, 61, 8), fontsize=19)

plt.tight_layout()

plt.savefig(f"{save_base}.png", dpi=300)
plt.savefig(f"{save_base}.eps")
