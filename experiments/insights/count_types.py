import sys
from collections import defaultdict

import astroid

from .. import utils

cnt = defaultdict(int)


def add_anno_type(anno):
    if anno is None:
        return
    if isinstance(anno, astroid.Name):
        cnt[anno.name] += 1
    elif isinstance(anno, astroid.Subscript):
        add_anno_type(anno.value)
        add_anno_type(anno.slice)
    elif isinstance(anno, astroid.Tuple):
        for item in anno.elts:
            add_anno_type(item)
    elif isinstance(anno, astroid.BinOp):
        add_anno_type(anno.left)
        add_anno_type(anno.right)
    elif isinstance(anno, astroid.Const):
        cnt[str(anno.kind)] += 1
    elif isinstance(anno, astroid.List):
        for item in anno.elts:
            add_anno_type(item)
    elif isinstance(anno, astroid.Attribute):
        cnt[anno.attrname] += 1
    else:
        raise TypeError(f"Unexpected annotation type: {anno}")


for project in sorted(utils.find_all_projects()):
    project_path = utils.BASE_PROJECT_PATH / project
    print(project_path)
    sys.path.append(str(project_path))
    for module_name in sorted(utils.find_all_filtered_modules(project)):
        # find the astroid module
        module = astroid.MANAGER.ast_from_module_name(module_name)

        # find all FunctionDef nodes
        functions = module.nodes_of_class(astroid.FunctionDef)
        for func in functions:
            # find the args type hint
            annotations = func.args.annotations
            for anno in annotations:
                add_anno_type(anno)

# Analyze the top 10 most common type annotations (amount & percentage)
total = sum(cnt.values())
for name, count in sorted(cnt.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {count} ({count / total * 100:.2f}%)")
