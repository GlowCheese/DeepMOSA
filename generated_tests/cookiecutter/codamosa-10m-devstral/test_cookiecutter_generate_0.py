# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.exceptions as module_1
import re as module_2
import jinja2.environment as module_3
import json.decoder as module_4
import cookiecutter.utils as module_5
import codecs as module_6
import jinja2.loaders as module_7

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

def test_case_5():
    var_0 = ''
    var_1 = {var_0: var_0}
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

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = True
    var_1 = None
    module_0.generate_files(var_1, var_0, accept_hooks=var_1)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'f'
    var_1 = None
    module_0.render_and_create_dir(var_0, var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'Uggl34['
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_2.escape(var_0)
    assert var_2 == 'Uggl34\\['
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    module_0.apply_overwrites_to_context(var_2, var_1, in_dictionary_variable=var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
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
    var_4 = module_2.escape(var_0)
    assert var_4 == '8\\(12nG%1:OT\\*x\\|'
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    var_5 = module_0.apply_overwrites_to_context(var_4, var_3)
    module_0.render_and_create_dir(var_3, var_3, var_4, var_4)

def test_case_10():
    var_0 = 'value1'
    var_1 = 'c?oie1'
    var_2 = [var_1, var_0]
    var_3 = {var_0: var_2}
    var_4 = 'invalid_choice'
    var_5 = {var_0: var_4}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_3, var_5)

def test_case_11():
    var_0 = 'var1'
    var_1 = 'value1'
    var_2 = {var_0: var_1, var_0: var_1}
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
    var_4 = 'choice_var'
    var_5 = 'choice2'
    var_6 = [var_5, var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_4: var_5}
    var_9 = module_0.apply_overwrites_to_context(var_7, var_8)

def test_case_12():
    var_0 = '/tmp/test_projiect'
    var_1 = {var_0: var_0}
    var_2 = module_3.Environment(loader=var_0)
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
    assert var_2.loader == '/tmp/test_projiect'
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
    var_3 = module_0.generate_file(var_0, var_0, var_1, var_2)
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

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = '/tmp/test_projiect'
    var_1 = {var_0: var_0}
    var_2 = module_3.Environment(loader=var_0)
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
    assert var_2.loader == '/tmp/test_projiect'
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
    var_3 = module_0.generate_file(var_0, var_0, var_1, var_2)
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
    module_0.generate_context(var_2)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '/tmp/test_projiect'
    var_1 = {var_0: var_0}
    var_2 = module_3.Environment(loader=var_0)
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
    assert var_2.loader == '/tmp/test_projiect'
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
    var_3 = 'i.hH<T&QoWqZ2F9'
    var_4 = False
    module_0.render_and_create_dir(var_3, var_1, var_3, var_2, var_4)

def test_case_15():
    var_0 = '/tmp/test_projiect'
    var_1 = {}
    var_2 = module_3.Environment(loader=var_0)
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
    assert var_2.loader == '/tmp/test_projiect'
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
    with pytest.raises(module_1.OutputDirExistsException):
        module_0.render_and_create_dir(var_0, var_1, var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = '/tmp/test_projiect'
    var_1 = {var_0: var_0}
    var_2 = module_3.Environment(loader=var_0)
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
    assert var_2.loader == '/tmp/test_projiect'
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
    var_3 = None
    var_4 = True
    var_5 = module_0.render_and_create_dir(var_0, var_1, var_0, var_2, var_4)
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
    module_4.py_scanstring(var_3, var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = ''
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1}
    var_3 = module_5.create_env_with_context(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
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
    assert var_3.keep_trailing_newline is True
    assert var_3.optimized is True
    assert var_3.finalize is None
    assert var_3.autoescape is False
    assert f'{type(var_3.filters).__module__}.{type(var_3.filters).__qualname__}' == 'builtins.dict'
    assert len(var_3.filters) == 56
    assert f'{type(var_3.tests).__module__}.{type(var_3.tests).__qualname__}' == 'builtins.dict'
    assert len(var_3.tests) == 39
    assert f'{type(var_3.globals).__module__}.{type(var_3.globals).__qualname__}' == 'builtins.dict'
    assert len(var_3.globals) == 8
    assert var_3.loader is None
    assert f'{type(var_3.cache).__module__}.{type(var_3.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_3.cache) == 0
    assert var_3.bytecode_cache is None
    assert var_3.auto_reload is True
    assert var_3.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_3.datetime_format == '%Y-%m-%d'
    assert f'{type(var_3.extensions).__module__}.{type(var_3.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_3.extensions) == 5
    assert var_3.is_async is False
    assert f'{type(module_5.annotations).__module__}.{type(module_5.annotations).__qualname__}' == '__future__._Feature'
    assert module_5.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_5.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_5.annotations.compiler_flag == 16777216
    assert module_5.TYPE_CHECKING is False
    assert f'{type(module_5.logger).__module__}.{type(module_5.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_5.logger.filters == []
    assert module_5.logger.name == 'cookiecutter.utils'
    assert module_5.logger.level == 0
    assert f'{type(module_5.logger.parent).__module__}.{type(module_5.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_5.logger.propagate is True
    assert module_5.logger.handlers == []
    assert module_5.logger.disabled is False
    assert f'{type(module_5.logger.manager).__module__}.{type(module_5.logger.manager).__qualname__}' == 'logging.Manager'
    var_4 = True
    module_0.generate_file(var_0, var_0, var_2, var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = False
    var_1 = 'I '
    var_2 = {var_1: var_0, var_1: var_0}
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
    module_6.open(var_3, encoding=var_3, buffering=var_3)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'name'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1}
    var_3 = module_7.FileSystemLoader(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert var_3.searchpath == ['name']
    assert var_3.encoding == 'utf-8'
    assert var_3.followlinks is False
    var_4 = False
    var_5 = module_0.apply_overwrites_to_context(var_2, var_1, in_dictionary_variable=var_4)
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
    var_6 = module_3.Environment(loader=var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'jinja2.environment.Environment'
    assert var_6.block_start_string == '{%'
    assert var_6.block_end_string == '%}'
    assert var_6.variable_start_string == '{{'
    assert var_6.variable_end_string == '}}'
    assert var_6.comment_start_string == '{#'
    assert var_6.comment_end_string == '#}'
    assert var_6.line_statement_prefix is None
    assert var_6.line_comment_prefix is None
    assert var_6.trim_blocks is False
    assert var_6.lstrip_blocks is False
    assert var_6.newline_sequence == '\n'
    assert var_6.keep_trailing_newline is False
    assert var_6.optimized is True
    assert var_6.finalize is None
    assert var_6.autoescape is False
    assert f'{type(var_6.filters).__module__}.{type(var_6.filters).__qualname__}' == 'builtins.dict'
    assert len(var_6.filters) == 54
    assert f'{type(var_6.tests).__module__}.{type(var_6.tests).__qualname__}' == 'builtins.dict'
    assert len(var_6.tests) == 39
    assert f'{type(var_6.globals).__module__}.{type(var_6.globals).__qualname__}' == 'builtins.dict'
    assert len(var_6.globals) == 6
    assert f'{type(var_6.loader).__module__}.{type(var_6.loader).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert f'{type(var_6.cache).__module__}.{type(var_6.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_6.cache) == 0
    assert var_6.bytecode_cache is None
    assert var_6.auto_reload is True
    assert var_6.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_6.extensions == {}
    assert var_6.is_async is False
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
    var_7 = True
    module_0.generate_file(var_0, var_0, var_2, var_6, var_7)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'JK{> cNP)|S\t\x0b7k'
    var_1 = 'te"k_template.txt'
    var_2 = 'name'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_3}
    var_5 = module_7.FileSystemLoader(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert var_5.searchpath == ['te"k_template.txt']
    assert var_5.encoding == 'utf-8'
    assert var_5.followlinks is False
    var_6 = module_0.apply_overwrites_to_context(var_4, var_4)
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
    module_3.Environment(var_6, comment_start_string=var_2, comment_end_string=var_6, newline_sequence=var_6, keep_trailing_newline=var_6, auto_reload=var_6, bytecode_cache=var_6)

def test_case_21():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
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
    var_9 = 'docs/readme.md'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True
    var_11 = 'file.py'
    var_12 = module_0.is_copy_only_path(var_11, var_6)
    assert var_12 is False
    var_13 = 'src/main.py'
    var_14 = module_0.is_copy_only_path(var_13, var_6)
    assert var_14 is False
    var_15 = {}
    var_16 = {var_0: var_15}
    var_17 = module_0.is_copy_only_path(var_7, var_16)
    assert var_17 is False
    var_18 = {}
    var_19 = module_0.is_copy_only_path(var_7, var_18)
    assert var_19 is False

def test_case_22():
    var_0 = 'var1'
    var_1 = 'va:2'
    var_2 = 'value1'
    var_3 = {var_0: var_2, var_1: var_1}
    var_4 = 'new_value1'
    var_5 = {var_0: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
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
    var_7 = {var_0: var_2}
    var_8 = module_0.apply_overwrites_to_context(var_7, var_3)
    var_9 = 'nested_var1'
    var_10 = 'nested_var2'
    var_11 = {var_9: var_4, var_10: var_1}
    var_12 = {var_0: var_11}
    var_13 = module_0.apply_overwrites_to_context(var_12, var_12)
    var_14 = 'choice1'
    var_15 = 'choice2'
    var_16 = 'choice3'
    var_17 = [var_14, var_15, var_16]
    var_18 = {var_0: var_17}
    var_19 = module_0.apply_overwrites_to_context(var_18, var_18)
    var_20 = {var_0: var_18}
    var_21 = module_0.apply_overwrites_to_context(var_20, var_20)

def test_case_23():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
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
    var_8 = {var_0: var_2}
    var_9 = {var_0: var_5, var_1: var_3}
    var_10 = module_0.apply_overwrites_to_context(var_8, var_9)
    var_11 = {var_3: var_2}
    var_12 = {var_0: var_11}
    var_13 = module_0.apply_overwrites_to_context(var_12, var_12)
    var_14 = 'choice1'
    var_15 = 'choice2'
    var_16 = 'choice3'
    var_17 = [var_14, var_15, var_16]
    var_18 = {var_0: var_17}
    var_19 = {var_0: var_17}
    var_20 = module_0.apply_overwrites_to_context(var_18, var_19)
    var_21 = [var_14, var_15, var_16]
    var_22 = {var_0: var_21}
    var_23 = 'choice4'
    var_24 = [var_23]
    var_25 = {var_0: var_24}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_22, var_25)

def test_case_24():
    var_0 = 'NY\\;oGPA?3~}eHA6"'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_6.make_encoding_map(var_1)
    assert module_6.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_6.BOM_LE == b'\xff\xfe'
    assert module_6.BOM_UTF16_LE == b'\xff\xfe'
    assert module_6.BOM_BE == b'\xfe\xff'
    assert module_6.BOM_UTF16_BE == b'\xfe\xff'
    assert module_6.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_6.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_6.BOM == b'\xff\xfe'
    assert module_6.BOM_UTF16 == b'\xff\xfe'
    assert module_6.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_6.BOM32_LE == b'\xff\xfe'
    assert module_6.BOM32_BE == b'\xfe\xff'
    assert module_6.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_6.BOM64_BE == b'\x00\x00\xfe\xff'
    var_3 = True
    var_4 = 'o?\nai+}^'
    var_5 = {var_4: var_2, var_0: var_3}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_5, var_2, in_dictionary_variable=var_3)