import ast
from typing import Any

from libs.custom_logger import getLogger
from pynguin.llm.deepmosa import outputfixers

_logger = getLogger(__name__)


class AssertionRemover(ast.NodeTransformer):
    def visit_Assert(self, node: ast.Assert) -> Any:
        return ast.Expr(node.test)


src = """
def test_something():
    for i in range(5):
        print(i)
    assert call(5)
"""

module_node = ast.parse(src)
module_node = AssertionRemover().visit(module_node)
tests = outputfixers.rewrite_tests(module_node)

print(tests["test_something"])
