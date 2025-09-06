from .utils import ExperimentStatistics
from app.evals.utils import BASE_STAT_PATH

stats = ExperimentStatistics()
stats.load_everything()
stats.analyse()

lim = 100
s = [0] * 4

print('\\midrule')

for i, mod in enumerate(sorted(stats.project_to_modules['thonny'])):
    props = stats.module_props[mod]
    mod_name = props.module_name.replace('_', '\\_')
    if i < lim:
        print(f"\\texttt{{{mod_name}}} & {props.loc} & {props.code_objects} & {2*props.predicates} \\\\")
    elif i == lim:
        print('... \\\\')

    s[0] += 1; s[1] += props.loc; s[2] += props.code_objects; s[3] += 2*props.predicates

print('\\midrule')
print(f'\\textbf{{Total ({s[0]} modules):}} & {s[1]} & {s[2]} & {s[3]} \\\\')

print('\\bottomrule')
