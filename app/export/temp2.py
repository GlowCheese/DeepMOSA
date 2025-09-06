import ast
import sys
import inspect
import argparse
import importlib

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("project_name", type=str)
parser.add_argument("module_name", type=str)

args = parser.parse_args()

project_name: str = args.project_name
module_name: str = args.module_name

project_path = (Path('../root-repo') / project_name).resolve(True)

sys.path.insert(0, str(project_path))

module = importlib.import_module(module_name)

source = inspect.getsource(module)

tree = ast.parse(source)

print(ast.dump(tree, indent=2))


class DeepestBranchVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.result = 0
        self.curr_depth = 0

    def update_depth(self, delta: int):
        self.curr_depth += delta

        self.result = max(self.result, self.curr_depth)

    def visit_IfExp(self, node: ast.IfExp):
        self.visit(node.test)
        self.update_depth(1)
        self.visit(node.body)
        self.visit(node.orelse)
        self.update_depth(-1)

    def visit_list(self, _list: list):
        for item in _list:
            self.visit(item)

    def visit_For(self, node: ast.For):
        self.visit(node.iter)
        self.update_depth(1)
        self.visit_list(node.body)
        self.visit_list(node.orelse)
        self.update_depth(-1)

    def visit_While(self, node: ast.While):
        self.visit(node.test)
        self.update_depth(1)
        self.visit_list(node.body)
        self.visit_list(node.orelse)
        self.update_depth(-1)

    def visit_If(self, node: ast.If):
        self.visit(node.test)
        self.update_depth(1)
        self.visit_list(node.body)
        self.visit_list(node.orelse)
        self.update_depth(-1)

    def visit_Try(self, node: ast.Try):
        self.visit_list(node.body)
        self.update_depth(1)
        self.visit_list(node.handlers)
        self.visit_list(node.orelse)
        self.visit_list(node.finalbody)
        self.update_depth(-1)

    def visit_comprehension(self, node: ast.comprehension):
        # comprehension is such a troublesome...
        self.update_depth(1)
        self.update_depth(-1)

    def visit_Match(self, node: ast.Match):
        self.visit(node.subject)
        self.update_depth(1)
        self.visit_list(node.cases)
        self.update_depth(-1)


visitor = DeepestBranchVisitor()
visitor.visit(tree)
assert visitor.curr_depth == 0
print(visitor.result)
