# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_7
import json.encoder as module_6
import re as module_5

import cookiecutter.exceptions as module_1
import cookiecutter.generate as module_0
import cookiecutter.utils as module_4
import jinja2.environment as module_3
import jinja2.loaders as module_2
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = 'm4_:!3np8NAqtB[&8'
    var_2 = {var_1: var_1}
    var_3 = module_0.is_copy_only_path(var_0, var_2)
    assert var_3 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.generate_context(default_context=var_0, extra_context=var_3)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.is_copy_only_path(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    module_0.generate_context()

def test_case_3():
    var_0 = None
    with pytest.raises(module_1.EmptyDirNameException):
        module_0.render_and_create_dir(var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.generate_files(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = False
    var_1 = '_copy_without_render'
    var_2 = [var_1]
    var_3 = {var_1: var_2, var_1: var_0}
    var_4 = module_0.apply_overwrites_to_context(var_3, var_3)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.generate_context()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = ''
    var_2 = {var_1: var_1, var_1: var_1}
    module_0.apply_overwrites_to_context(var_0, var_2, in_dictionary_variable=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'cookiecutter'
    var_1 = []
    var_2 = {var_0: var_0, var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = module_0.apply_overwrites_to_context(var_2, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_4 = module_2.FileSystemLoader(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert var_4.searchpath == ['cookiecutter']
    assert var_4.encoding == 'utf-8'
    assert var_4.followlinks is False
    var_5 = module_3.Environment(loader=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'jinja2.environment.Environment'
    assert var_5.block_start_string == '{%'
    assert var_5.block_end_string == '%}'
    assert var_5.variable_start_string == '{{'
    assert var_5.variable_end_string == '}}'
    assert var_5.comment_start_string == '{#'
    assert var_5.comment_end_string == '#}'
    assert var_5.line_statement_prefix is None
    assert var_5.line_comment_prefix is None
    assert var_5.trim_blocks is False
    assert var_5.lstrip_blocks is False
    assert var_5.newline_sequence == '\n'
    assert var_5.keep_trailing_newline is False
    assert var_5.optimized is True
    assert var_5.finalize is None
    assert var_5.autoescape is False
    assert f'{type(var_5.filters).__module__}.{type(var_5.filters).__qualname__}' == 'builtins.dict'
    assert len(var_5.filters) == 54
    assert f'{type(var_5.tests).__module__}.{type(var_5.tests).__qualname__}' == 'builtins.dict'
    assert len(var_5.tests) == 39
    assert f'{type(var_5.globals).__module__}.{type(var_5.globals).__qualname__}' == 'builtins.dict'
    assert len(var_5.globals) == 6
    assert f'{type(var_5.loader).__module__}.{type(var_5.loader).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert f'{type(var_5.cache).__module__}.{type(var_5.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_5.cache) == 0
    assert var_5.bytecode_cache is None
    assert var_5.auto_reload is True
    assert var_5.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_5.extensions == {}
    assert var_5.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    module_4.rmtree(var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'U=ggl4['
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.apply_overwrites_to_context(var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.render_and_create_dir(var_0, var_2, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = True
    var_1 = None
    module_0.generate_files(var_1, var_0, accept_hooks=var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = True
    var_1 = '@Y"9u\tWP'
    var_2 = {}
    var_3 = module_3.Environment(loader=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.environment.Environment'
    assert var_3.block_start_string == '{%'
    assert var_3.block_end_string == '%}'
    assert var_3.variable_start_string == '{{'
    assert var_3.variable_end_string == '}}'
    assert var_3.comment_start_string == '{#'
    assert var_3.comment_end_string == '#}'
    assert var_3.line_statement_prefix is None
    assert var_3.line_comment_prefix is None
    assert var_3.trim_blocks is False
    assert var_3.lstrip_blocks is False
    assert var_3.newline_sequence == '\n'
    assert var_3.keep_trailing_newline is False
    assert var_3.optimized is True
    assert var_3.finalize is None
    assert var_3.autoescape is False
    assert f'{type(var_3.filters).__module__}.{type(var_3.filters).__qualname__}' == 'builtins.dict'
    assert len(var_3.filters) == 54
    assert f'{type(var_3.tests).__module__}.{type(var_3.tests).__qualname__}' == 'builtins.dict'
    assert len(var_3.tests) == 39
    assert f'{type(var_3.globals).__module__}.{type(var_3.globals).__qualname__}' == 'builtins.dict'
    assert len(var_3.globals) == 6
    assert var_3.loader == {}
    assert f'{type(var_3.cache).__module__}.{type(var_3.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_3.cache) == 0
    assert var_3.bytecode_cache is None
    assert var_3.auto_reload is True
    assert var_3.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_3.extensions == {}
    assert var_3.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    module_0.render_and_create_dir(var_1, var_2, var_1, var_3, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'Uggl34['
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_5.escape(var_0)
    assert var_2 == 'Uggl34\\['
    assert module_5.ASCII == module_5.RegexFlag.ASCII
    assert module_5.A == module_5.RegexFlag.ASCII
    assert module_5.IGNORECASE == module_5.RegexFlag.IGNORECASE
    assert module_5.I == module_5.RegexFlag.IGNORECASE
    assert module_5.LOCALE == module_5.RegexFlag.LOCALE
    assert module_5.L == module_5.RegexFlag.LOCALE
    assert module_5.UNICODE == module_5.RegexFlag.UNICODE
    assert module_5.U == module_5.RegexFlag.UNICODE
    assert module_5.MULTILINE == module_5.RegexFlag.MULTILINE
    assert module_5.M == module_5.RegexFlag.MULTILINE
    assert module_5.DOTALL == module_5.RegexFlag.DOTALL
    assert module_5.S == module_5.RegexFlag.DOTALL
    assert module_5.VERBOSE == module_5.RegexFlag.VERBOSE
    assert module_5.X == module_5.RegexFlag.VERBOSE
    assert module_5.TEMPLATE == module_5.RegexFlag.TEMPLATE
    assert module_5.T == module_5.RegexFlag.TEMPLATE
    assert module_5.DEBUG == module_5.RegexFlag.DEBUG
    module_0.apply_overwrites_to_context(var_2, var_1, in_dictionary_variable=var_2)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '8(12nG%1:OT*x|'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_3 = {var_0: var_0, var_0: var_0}
    var_4 = module_5.escape(var_0)
    assert var_4 == '8\\(12nG%1:OT\\*x\\|'
    assert module_5.ASCII == module_5.RegexFlag.ASCII
    assert module_5.A == module_5.RegexFlag.ASCII
    assert module_5.IGNORECASE == module_5.RegexFlag.IGNORECASE
    assert module_5.I == module_5.RegexFlag.IGNORECASE
    assert module_5.LOCALE == module_5.RegexFlag.LOCALE
    assert module_5.L == module_5.RegexFlag.LOCALE
    assert module_5.UNICODE == module_5.RegexFlag.UNICODE
    assert module_5.U == module_5.RegexFlag.UNICODE
    assert module_5.MULTILINE == module_5.RegexFlag.MULTILINE
    assert module_5.M == module_5.RegexFlag.MULTILINE
    assert module_5.DOTALL == module_5.RegexFlag.DOTALL
    assert module_5.S == module_5.RegexFlag.DOTALL
    assert module_5.VERBOSE == module_5.RegexFlag.VERBOSE
    assert module_5.X == module_5.RegexFlag.VERBOSE
    assert module_5.TEMPLATE == module_5.RegexFlag.TEMPLATE
    assert module_5.T == module_5.RegexFlag.TEMPLATE
    assert module_5.DEBUG == module_5.RegexFlag.DEBUG
    var_5 = module_0.apply_overwrites_to_context(var_4, var_3)
    module_0.render_and_create_dir(var_3, var_3, var_4, var_4)

def test_case_13():
    var_0 = 'cookiecutter'
    var_1 = 'bool_var'
    var_2 = 'dict_var'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = 'y'
    var_8 = 'z'
    var_9 = [var_3, var_7, var_8]
    var_10 = 'key1'
    var_11 = 'key2'
    var_12 = 'value1'
    var_13 = 'value2'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = {var_2: var_6, var_12: var_9, var_1: var_14, var_2: var_14}
    var_16 = {var_0: var_15}
    var_17 = [var_7, var_8]
    var_18 = 'no'
    var_19 = 'key3'
    var_20 = 'new_value2'
    var_21 = {var_11: var_20, var_19: var_11}
    var_22 = {var_10: var_4, var_3: var_17, var_1: var_18, var_2: var_21}
    var_23 = var_16[var_0]
    var_24 = module_0.apply_overwrites_to_context(var_23, var_22)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_25 = 'cookiecutter'
    var_26 = var_16[var_25]
    var_27 = 'choice_var'
    var_28 = 'd'
    var_29 = {var_27: var_28}
    var_30 = module_0.apply_overwrites_to_context(var_26, var_29)

def test_case_14():
    var_0 = 'cookiecutter'
    var_1 = 'choice_var'
    var_2 = 'multi_choice_var'
    var_3 = '7^;'
    var_4 = 'b'
    var_5 = [var_2, var_4, var_4]
    var_6 = 'z'
    var_7 = True
    var_8 = 'key2'
    var_9 = {var_1: var_0, var_8: var_0}
    var_10 = {var_1: var_5, var_2: var_5, var_3: var_7, var_4: var_9}
    var_11 = {var_0: var_10}
    var_12 = 'value3'
    var_13 = {var_8: var_10, var_1: var_12}
    var_14 = {var_1: var_4, var_2: var_5, var_3: var_6, var_8: var_13}
    var_15 = var_11[var_0]
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_15, var_14)

def test_case_15():
    var_0 = 'cookiecutter'
    var_1 = 'choice_var'
    var_2 = 'multi_choice_var'
    var_3 = 'bool_var'
    var_4 = 'dict_var'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = 'x'
    var_10 = 'y'
    var_11 = 'z'
    var_12 = [var_9, var_10, var_11]
    var_13 = True
    var_14 = 'key1'
    var_15 = 'key2'
    var_16 = 'value1'
    var_17 = 'value2'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = {var_1: var_8, var_2: var_12, var_3: var_13, var_4: var_18}
    var_20 = {var_0: var_19}
    var_21 = [var_10, var_11]
    var_22 = 'no'
    var_23 = 'key3'
    var_24 = 'new_value2'
    var_25 = 'value3'
    var_26 = {var_15: var_24, var_23: var_25}
    var_27 = {var_1: var_6, var_2: var_21, var_3: var_22, var_4: var_26}
    var_28 = var_20[var_0]
    var_29 = module_0.apply_overwrites_to_context(var_28, var_27)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_30 = 'cookiecutter'
    var_31 = var_20[var_30]
    var_32 = module_0.apply_overwrites_to_context(var_31, var_18)
    var_33 = 'cookiecutter'
    var_34 = var_20[var_33]
    var_35 = 'multi_choice_var'
    var_36 = 'w'
    var_37 = [var_36]
    var_38 = {var_35: var_37}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_34, var_38)

def test_case_16():
    var_0 = 'F>c^5cS|rx'
    var_1 = module_6.py_encode_basestring_ascii(var_0)
    assert var_1 == '"F>c^5cS|rx"'
    assert f'{type(module_6.ESCAPE).__module__}.{type(module_6.ESCAPE).__qualname__}' == 're.Pattern'
    assert f'{type(module_6.ESCAPE_ASCII).__module__}.{type(module_6.ESCAPE_ASCII).__qualname__}' == 're.Pattern'
    assert f'{type(module_6.HAS_UTF8).__module__}.{type(module_6.HAS_UTF8).__qualname__}' == 're.Pattern'
    assert module_6.ESCAPE_DCT == {'\\': '\\\\', '"': '\\"', '\x08': '\\b', '\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\x00': '\\u0000', '\x01': '\\u0001', '\x02': '\\u0002', '\x03': '\\u0003', '\x04': '\\u0004', '\x05': '\\u0005', '\x06': '\\u0006', '\x07': '\\u0007', '\x0b': '\\u000b', '\x0e': '\\u000e', '\x0f': '\\u000f', '\x10': '\\u0010', '\x11': '\\u0011', '\x12': '\\u0012', '\x13': '\\u0013', '\x14': '\\u0014', '\x15': '\\u0015', '\x16': '\\u0016', '\x17': '\\u0017', '\x18': '\\u0018', '\x19': '\\u0019', '\x1a': '\\u001a', '\x1b': '\\u001b', '\x1c': '\\u001c', '\x1d': '\\u001d', '\x1e': '\\u001e', '\x1f': '\\u001f'}
    assert module_6.i == 31
    assert module_6.INFINITY == pytest.approx(1e309, abs=0.01, rel=0.01)
    var_2 = True
    var_3 = {var_0: var_0, var_0: var_1}
    var_4 = {var_1: var_1, var_0: var_1, var_0: var_2, var_1: var_3}
    var_5 = {var_1: var_1, var_1: var_4, var_1: var_4}
    var_6 = {var_0: var_1, var_1: var_1, var_0: var_0, var_0: var_1}
    var_7 = var_5[var_1]
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_7, var_6)

def test_case_17():
    var_0 = 'cookiecutter'
    var_1 = 'choice_var'
    var_2 = 'bool_var'
    var_3 = 'dict_var'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = [var_4, var_5, var_6]
    var_8 = 'y'
    var_9 = 'z'
    var_10 = [var_4, var_8, var_9]
    var_11 = True
    var_12 = 'key1'
    var_13 = 'key2'
    var_14 = 'value1'
    var_15 = 'value2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = {var_1: var_7, var_14: var_10, var_2: var_11, var_3: var_16}
    var_18 = {var_0: var_17}
    var_19 = [var_8, var_9]
    var_20 = 'no'
    var_21 = 'key3'
    var_22 = 'new_value2'
    var_23 = 'value3'
    var_24 = {var_13: var_22, var_21: var_23}
    var_25 = {var_1: var_5, var_23: var_19, var_2: var_20, var_3: var_24}
    var_26 = var_18[var_0]
    var_27 = module_0.apply_overwrites_to_context(var_26, var_25)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_28 = 'cookiecutter'
    var_29 = var_18[var_28]
    var_30 = 'choice_var'
    var_31 = 'd'
    var_32 = {var_30: var_31}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_29, var_32)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = '@Y"9u\tWP'
    var_1 = module_7._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_3.Environment(loader=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'jinja2.environment.Environment'
    assert var_2.block_start_string == '{%'
    assert var_2.block_end_string == '%}'
    assert var_2.variable_start_string == '{{'
    assert var_2.variable_end_string == '}}'
    assert var_2.comment_start_string == '{#'
    assert var_2.comment_end_string == '#}'
    assert var_2.line_statement_prefix is None
    assert var_2.line_comment_prefix is None
    assert var_2.trim_blocks is False
    assert var_2.lstrip_blocks is False
    assert var_2.newline_sequence == '\n'
    assert var_2.keep_trailing_newline is False
    assert var_2.optimized is True
    assert var_2.finalize is None
    assert var_2.autoescape is False
    assert f'{type(var_2.filters).__module__}.{type(var_2.filters).__qualname__}' == 'builtins.dict'
    assert len(var_2.filters) == 54
    assert f'{type(var_2.tests).__module__}.{type(var_2.tests).__qualname__}' == 'builtins.dict'
    assert len(var_2.tests) == 39
    assert f'{type(var_2.globals).__module__}.{type(var_2.globals).__qualname__}' == 'builtins.dict'
    assert len(var_2.globals) == 6
    assert f'{type(var_2.loader).__module__}.{type(var_2.loader).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.loader) == 0
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.extensions == {}
    assert var_2.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    module_0.generate_file(var_0, var_0, var_1, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = ''
    var_1 = module_7._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_3.Environment(loader=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'jinja2.environment.Environment'
    assert var_2.block_start_string == '{%'
    assert var_2.block_end_string == '%}'
    assert var_2.variable_start_string == '{{'
    assert var_2.variable_end_string == '}}'
    assert var_2.comment_start_string == '{#'
    assert var_2.comment_end_string == '#}'
    assert var_2.line_statement_prefix is None
    assert var_2.line_comment_prefix is None
    assert var_2.trim_blocks is False
    assert var_2.lstrip_blocks is False
    assert var_2.newline_sequence == '\n'
    assert var_2.keep_trailing_newline is False
    assert var_2.optimized is True
    assert var_2.finalize is None
    assert var_2.autoescape is False
    assert f'{type(var_2.filters).__module__}.{type(var_2.filters).__qualname__}' == 'builtins.dict'
    assert len(var_2.filters) == 54
    assert f'{type(var_2.tests).__module__}.{type(var_2.tests).__qualname__}' == 'builtins.dict'
    assert len(var_2.tests) == 39
    assert f'{type(var_2.globals).__module__}.{type(var_2.globals).__qualname__}' == 'builtins.dict'
    assert len(var_2.globals) == 6
    assert f'{type(var_2.loader).__module__}.{type(var_2.loader).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.loader) == 0
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.extensions == {}
    assert var_2.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    module_0.generate_file(var_0, var_0, var_1, var_2, var_0)

def test_case_20():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'images/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'scripts/script.py'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

def test_case_21():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'images/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_9 = 'images/photo.jpg'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True
    var_11 = 's+rip-/scr?it.py'
    var_12 = module_0.is_copy_only_path(var_11, var_6)
    assert var_12 is False
    var_13 = 'README.md'
    var_14 = module_0.is_copy_only_path(var_13, var_6)
    assert var_14 is False

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = []
    var_3 = '\x0c\x0bm<\x0cK#5D"GI\r)'
    var_4 = {var_3: var_2}
    var_5 = module_0.apply_overwrites_to_context(var_4, var_4, in_dictionary_variable=var_4)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_6 = {var_1: var_1, var_0: var_2, var_1: var_2, var_1: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_6, var_6)
    var_8 = module_2.FileSystemLoader(var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert var_8.searchpath == ['cookiecutter']
    assert var_8.encoding == 'utf-8'
    assert var_8.followlinks is False
    var_9 = module_3.Environment(loader=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.environment.Environment'
    assert var_9.block_start_string == '{%'
    assert var_9.block_end_string == '%}'
    assert var_9.variable_start_string == '{{'
    assert var_9.variable_end_string == '}}'
    assert var_9.comment_start_string == '{#'
    assert var_9.comment_end_string == '#}'
    assert var_9.line_statement_prefix is None
    assert var_9.line_comment_prefix is None
    assert var_9.trim_blocks is False
    assert var_9.lstrip_blocks is False
    assert var_9.newline_sequence == '\n'
    assert var_9.keep_trailing_newline is False
    assert var_9.optimized is True
    assert var_9.finalize is None
    assert var_9.autoescape is False
    assert f'{type(var_9.filters).__module__}.{type(var_9.filters).__qualname__}' == 'builtins.dict'
    assert len(var_9.filters) == 54
    assert f'{type(var_9.tests).__module__}.{type(var_9.tests).__qualname__}' == 'builtins.dict'
    assert len(var_9.tests) == 39
    assert f'{type(var_9.globals).__module__}.{type(var_9.globals).__qualname__}' == 'builtins.dict'
    assert len(var_9.globals) == 6
    assert f'{type(var_9.loader).__module__}.{type(var_9.loader).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert f'{type(var_9.cache).__module__}.{type(var_9.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_9.cache) == 0
    assert var_9.bytecode_cache is None
    assert var_9.auto_reload is True
    assert var_9.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_9.extensions == {}
    assert var_9.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    var_10 = 'Hello {{ name }}!'
    var_11 = 'H1+x0eTs}GoW\x0c^aPN4V'
    module_0.generate_file(var_11, var_7, var_6, var_7, var_10)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = '-_Ztz>>fy=n^['
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = 'coki!c5Yutter'
    var_3 = False
    var_4 = {var_2: var_1}
    var_5 = '.'
    var_6 = module_2.FileSystemLoader(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert var_6.searchpath == ['.']
    assert var_6.encoding == 'utf-8'
    assert var_6.followlinks is False
    var_7 = module_3.Environment(loader=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'jinja2.environment.Environment'
    assert var_7.block_start_string == '{%'
    assert var_7.block_end_string == '%}'
    assert var_7.variable_start_string == '{{'
    assert var_7.variable_end_string == '}}'
    assert var_7.comment_start_string == '{#'
    assert var_7.comment_end_string == '#}'
    assert var_7.line_statement_prefix is None
    assert var_7.line_comment_prefix is None
    assert var_7.trim_blocks is False
    assert var_7.lstrip_blocks is False
    assert var_7.newline_sequence == '\n'
    assert var_7.keep_trailing_newline is False
    assert var_7.optimized is True
    assert var_7.finalize is None
    assert var_7.autoescape is False
    assert f'{type(var_7.filters).__module__}.{type(var_7.filters).__qualname__}' == 'builtins.dict'
    assert len(var_7.filters) == 54
    assert f'{type(var_7.tests).__module__}.{type(var_7.tests).__qualname__}' == 'builtins.dict'
    assert len(var_7.tests) == 39
    assert f'{type(var_7.globals).__module__}.{type(var_7.globals).__qualname__}' == 'builtins.dict'
    assert len(var_7.globals) == 6
    assert f'{type(var_7.loader).__module__}.{type(var_7.loader).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert f'{type(var_7.cache).__module__}.{type(var_7.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_7.cache) == 0
    assert var_7.bytecode_cache is None
    assert var_7.auto_reload is True
    assert var_7.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_7.extensions == {}
    assert var_7.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    var_8 = module_0.generate_file(var_5, var_5, var_4, var_7, var_3)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.generate_context(extra_context=var_8)