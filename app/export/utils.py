import math
import os
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.evals.utils import BASE_STAT_PATH, EXT_STAT_PATHS
from app.qualify import qualify
from app.qualify.qualify import QualifyStatus


TARGET_ALGORITHMS = (
    "DYNAMOSA",
    "CODAMOSA",
    "DEEPMOSAv3",
)


@dataclass
class ModuleProperties:
    module_name: str
    project_name: str
    loc: int
    predicates: int
    code_objects: int


@dataclass
class ModuleStatistics:
    line_coverage: float
    branch_coverage: float
    llm_calls: int
    llm_query_time: float
    llm_saved_tests: int
    llm_input_tokens: int
    llm_output_tokens: int
    parsed_stmts: int
    parsable_stmts: int
    coverage_at: list[float]


class ExperimentStatistics:
    def __init__(self):
        self.algorithms = TARGET_ALGORITHMS

        self.module_props: dict[str, ModuleProperties] = {}
        self.module_stats: dict[
            str, dict[str, ModuleStatistics]] = defaultdict(dict)

        self.common_modules: set[str] | None = None
        self.common_projects: set[str] | None = None
        self.project_to_modules: dict[str, list[str]] | None = None

    def to_float(self, x):
        return 0.0 if math.isnan(x) else x

    def load(self, base_path: str):
        print(f"Loading statistics with base_path: {base_path}")

        new_data = defaultdict(set)
        dup_data = defaultdict(set)

        for _algo in TARGET_ALGORITHMS:
            algo = _algo.lower()
            stat_path = base_path.format(algo)
            if not os.path.isfile(stat_path):
                continue

            df = pd.read_csv(stat_path)

            for row in df.itertuples(index=False):
                qualify.set_project_name(row.ProjectName)
                status = qualify.get_status(row.TargetModule)
                if status != QualifyStatus.GOOD:
                    continue

                # print(_algo, row.TargetModule)
                
                props = ModuleProperties(
                    row.TargetModule,
                    row.ProjectName,
                    row.LineNos,
                    row.Predicates,
                    row.CodeObjects
                )
                if row.TargetModule not in self.module_props:
                    self.module_props[row.TargetModule] = props
                assert self.module_props[row.TargetModule] == props

                coverage_at = []
                for field in row._fields:
                    if field.startswith("CT_"):
                        value = getattr(row, field)
                        value = value.partition(',')[0][1:]
                        coverage_at.append(float(value))
                assert(len(coverage_at) == 600)

                if row.TargetModule not in self.module_stats[_algo]:
                    new_data[_algo].add(row.ProjectName)
                    self.module_stats[_algo][row.TargetModule] = \
                        ModuleStatistics(
                            row.LineCoverage,
                            row.BranchCoverage,
                            self.to_float(row.LLMCalls),
                            self.to_float(row.LLMQueryTime),
                            self.to_float(row.LLMStageSavedTests),
                            self.to_float(row.LLMInputTokens),
                            self.to_float(row.LLMOutputTokens),
                            self.to_float(row.ParsedStatements),
                            self.to_float(row.ParsableStatements),
                            coverage_at
                        )
                else:
                    dup_data[_algo].add(row.TargetModule)

        if new_data:
            print("- New data:")
            for k, v in new_data.items():
                print(f"    {k}: {', '.join(v)}")

        if dup_data:
            print("- Duplicated data:")
            for k, v in dup_data.items():
                print(f"    {k}: {', '.join(v)}")
                
    def load_everything(self):
        self.load(BASE_STAT_PATH)
        for path in EXT_STAT_PATHS:
            self.load(path)
        print("Load summary:")
        for algo, stats in self.module_stats.items():
            s = defaultdict(list)
            for module_name in stats.keys():
                s[self.module_props[module_name].project_name].append(module_name)
            print(f"- {algo}: {', '.join(f'{prj} ({len(items)})' for prj, items in s.items())}")

                
    def analyse(self):
        self.common_modules = set(self.module_props.keys())
        for stats in self.module_stats.values():
            self.common_modules &= set(stats.keys())

        assert self.common_modules, "No common modules found"

        for algo, stats in self.module_stats.items():
            self.module_stats[algo] = {
                k: v for k, v in stats.items()
                if k in self.common_modules
            }

        self.common_projects = set()
        self.project_to_modules = defaultdict(list)

        for mod in self.common_modules:
            project = self.module_props[mod].project_name
            self.common_projects.add(project)
            self.project_to_modules[project].append(mod)

    def __get_modules(self, project: str | None = None):
        return self.common_modules if project is None \
            else self.project_to_modules[project]
    
    def _algo_average_report(
        self, algo: str, metric: str,
        project: str | None = None,
        weight: str | None = None,
    ):
        modules = self.__get_modules(project)
        return np.average(
            list(
                getattr(self.module_stats[algo][mod], metric)
                for mod in modules
            ),
            weights=None if weight is None else list(
                getattr(self.module_props[mod], weight)
                for mod in modules
            )
        )

    def _algo_sum_report(
        self, algo: str, metric: str,
        project: str | None = None,
    ):
        modules = self.__get_modules(project)
        return np.sum(
            list(
                getattr(self.module_stats[algo][mod], metric)
                for mod in modules
            ),
        )

    def __log_custom_row(self, title, *args, t_len=17, d_len=10, t_direction='>', d_direction='>'):
        args = list(args)
        for i, value in enumerate(args):
            if isinstance(value, float):
                args[i] = f'{value:{d_direction}{d_len}.{d_len-6}f}'
        print(f"{title:{t_direction}{t_len}} ||{'|'.join(f' {value:^{d_len}} ' for value in args)}")

    def _project_report(self, title: str, project: str | None = None):
        print(f"{f' {title} ':-^{7+13*len(TARGET_ALGORITHMS)}}")
        print()

        for algo in self.algorithms:
            print(
                f"* {algo:<10} average coverage:",
                round(100 * self._algo_average_report(
                    algo, 'branch_coverage',
                    project, 'predicates'
                ), 1)
            )

        print()
        print("* LLM-based reports:")
        print()

        self.__log_custom_row('', *TARGET_ALGORITHMS)
        for metric in (
            'llm_calls', 'llm_query_time', 'llm_saved_tests',
            'llm_input_tokens', 'llm_output_tokens',
        ):
            self.__log_custom_row(
                metric, *(
                    self._algo_average_report(algo, metric, project)
                    for algo in TARGET_ALGORITHMS
                )
            )
        self.__log_custom_row(
            'stmts_parse_ratio', *(
                100 * self._algo_sum_report(algo, 'parsed_stmts', project)
                    / self._algo_sum_report(algo, 'parsable_stmts', project)
                for algo in TARGET_ALGORITHMS
            )
        )

        print()
        print("* Baselines H2H:")
        print()

        def short_algo(algo):
            return algo[:4] + algo[8:]

        self.__log_custom_row('', *[short_algo(algo) for algo in TARGET_ALGORITHMS], t_len=6, d_len=6)

        for r in range(len(TARGET_ALGORITHMS)):
            h2h = [0] * len(TARGET_ALGORITHMS)
            for c in range(len(TARGET_ALGORITHMS)):
                if r == c: continue
                for mod in self.__get_modules(project):
                    if self.module_stats[TARGET_ALGORITHMS[c]][mod].branch_coverage \
                     > self.module_stats[TARGET_ALGORITHMS[r]][mod].branch_coverage:
                        h2h[c] += 1
            self.__log_custom_row(short_algo(TARGET_ALGORITHMS[r]), *h2h, t_len=6, d_len=6)
        
        print()

   
    def _project_props_report(self):
        print(f"{' Project Properties ':-^{7+13*len(TARGET_ALGORITHMS)}}")
        print()
        mx_proj_len = max(len(p) for p in self.common_projects)
        for project in self.common_projects:
            s = f"* {project:>{mx_proj_len}}: "
            modules = self.project_to_modules[project]

            num_modules = len(modules)
            sum_loc = num_preds = num_objs = 0

            for module in self.project_to_modules[project]:
                props = self.module_props[module]
                sum_loc += props.loc
                num_preds += props.predicates
                num_objs += props.code_objects
            
            s += f"{num_modules:>3} modules, "
            s += f"{sum_loc:>6} LOCs, "
            s += f"{num_objs:>5} Units, "
            s += f"{2*num_preds:>5} Branches"

            print(s)

        print()


    def _coverage_report_detail(self):
        print(f"{f' Coverage Report Detail ':-^{7+13*len(TARGET_ALGORITHMS)}}")
        print()

        mlen = max(len(m) for m in self.common_modules)
        self.__log_custom_row('', *TARGET_ALGORITHMS, t_len=mlen)
        for mod in sorted(self.common_modules):
            self.__log_custom_row(
                mod,
                *list(
                    self.module_stats[algo][mod].branch_coverage
                    for algo in TARGET_ALGORITHMS
                ),
                t_len=mlen, t_direction='<', d_direction='^'
            )

        print()


    def report(self):        
        self._coverage_report_detail()

        for project in self.common_projects:
            self._project_report(f"Report for project: {project}", project)

        self._project_report("Overall Reports")
        self._project_props_report()
