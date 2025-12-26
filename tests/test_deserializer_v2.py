import ast

from pynguin.configuration import (
    Algorithm,
    Configuration,
    LLMConfiguration,
    StatisticsOutputConfiguration,
    StoppingConfiguration,
    TestCaseOutputConfiguration,
    set_configuration,
)

config = Configuration(
    project_path="experiments/projects/minbpe",
    module_name="minbpe.regex",
    algorithm=Algorithm.DEEPMOSA,
    test_case_output=TestCaseOutputConfiguration(
        output_path="tests_temp",
    ),
    statistics_output=StatisticsOutputConfiguration(
        project_name="minbpe",
    ),
    stopping=StoppingConfiguration(
        maximum_search_time=90,
    ),
    llm=LLMConfiguration(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
    ),
)

set_configuration(config)

from pynguin.export.pytestexporter import PyTestExporter
from pynguin.generator import prepare_everything
from pynguin.llm.ast_to_testcase import AstToTestCaseVisitor
from pynguin.llm.deepmosa.stmtdeserializer_v2 import StatementDeserializerV2

test_cluster, executor, constant_provider = prepare_everything()

visitor = AstToTestCaseVisitor(
    include_nontest_functions=False,
    statement_deserializer=StatementDeserializerV2(test_cluster, True),
)

raw_test_str = """
def test_encode_without_special_tokens():
    a = "hello"
    b = "world"
    c = a + b
    assert a + b == c
"""

visitor.visit(ast.parse(raw_test_str))

testcase_str = PyTestExporter(wrap_code=False).export_sequences_to_str(visitor.testcases)  # type: ignore

print(testcase_str)
