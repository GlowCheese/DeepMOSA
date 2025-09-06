import os

import seaborn as sns
import matplotlib.pyplot as plt

from .utils import ExperimentStatistics

algo_map = {
    "DEEPMOSAv3": ("DeepMOSA (ours)", 'solid', None),
    "DEEPMOSAv4": ("DSyncMOSA", (0, (3, 5, 1, 5)), None),
    "CODAMOSA": ("CodaMOSA", 'solid', 'X'),
    "DYNAMOSA": ("DynaMOSA", 'dotted', None),
}

save_base = 'pynguin-report/plot'
os.makedirs(os.path.dirname(save_base), exist_ok=True)

sns.set_theme(style='whitegrid')

exs = ExperimentStatistics()
exs.load_everything()
exs.analyse()

plt.figure(figsize=(9, 5))
sum_pred = sum(exs.module_props[m].predicates for m in exs.common_modules)

for algo, stats in exs.module_stats.items():
    aggre = [0] * 600
    
    for m in exs.common_modules:
        multp = stats[m].branch_coverage / stats[m].coverage_at[-1]
        pred = exs.module_props[m].predicates
        for i, cvrg in enumerate(stats[m].coverage_at):
            aggre[i] += pred * cvrg * multp

    for i in range(600):
        aggre[i] = 100 * aggre[i] / sum_pred

    sns.lineplot(
        x=range(600),
        y=aggre,
        label=algo_map[algo][0],
        color='black',
        linestyle=algo_map[algo][1],
        marker=algo_map[algo][2],
        markevery=70,
        markersize=7,
    )

plt.xlim(right=600)
plt.xlabel('Time (seconds)', fontsize=15)
plt.ylabel('Branch coverage (%)', fontsize=15)
plt.legend(fontsize=12)

plt.savefig(f'{save_base}.png', dpi=300)
plt.savefig(f'{save_base}.eps')