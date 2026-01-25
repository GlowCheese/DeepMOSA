# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.utils as module_0
import jinja2.environment as module_1
import pytest


def test_case_0():
    var_0 = module_0.work_in()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_0.gen).__module__}.{type(var_0.gen).__qualname__}' == 'builtins.generator'
    assert var_0.args == ()
    assert var_0.kwds == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.utils'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

def test_case_1():
    var_0 = 'st'
    with pytest.raises(OSError):
        module_0.make_sure_path_exists(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.make_sure_path_exists(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.force_delete(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.rmtree(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.make_executable(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.simple_filter(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.create_tmp_repo_dir(var_0)

def test_case_8():
    var_0 = {}
    var_1 = module_0.create_env_with_context(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_1.block_start_string == '{%'
    assert var_1.block_end_string == '%}'
    assert var_1.variable_start_string == '{{'
    assert var_1.variable_end_string == '}}'
    assert var_1.comment_start_string == '{#'
    assert var_1.comment_end_string == '#}'
    assert var_1.line_statement_prefix is None
    assert var_1.line_comment_prefix is None
    assert var_1.trim_blocks is False
    assert var_1.lstrip_blocks is False
    assert var_1.newline_sequence == '\n'
    assert var_1.keep_trailing_newline is True
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 56
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 8
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.datetime_format == '%Y-%m-%d'
    assert f'{type(var_1.extensions).__module__}.{type(var_1.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_1.extensions) == 5
    assert var_1.is_async is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.utils'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

def test_case_9():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = module_0.simple_filter(var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.utils'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_3 = [var_2]
    var_4 = module_1.Environment(extensions=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'jinja2.environment.Environment'
    assert var_4.block_start_string == '{%'
    assert var_4.block_end_string == '%}'
    assert var_4.variable_start_string == '{{'
    assert var_4.variable_end_string == '}}'
    assert var_4.comment_start_string == '{#'
    assert var_4.comment_end_string == '#}'
    assert var_4.line_statement_prefix is None
    assert var_4.line_comment_prefix is None
    assert var_4.trim_blocks is False
    assert var_4.lstrip_blocks is False
    assert var_4.newline_sequence == '\n'
    assert var_4.keep_trailing_newline is False
    assert var_4.optimized is True
    assert var_4.finalize is None
    assert var_4.autoescape is False
    assert f'{type(var_4.filters).__module__}.{type(var_4.filters).__qualname__}' == 'builtins.dict'
    assert len(var_4.filters) == 55
    assert f'{type(var_4.tests).__module__}.{type(var_4.tests).__qualname__}' == 'builtins.dict'
    assert len(var_4.tests) == 39
    assert f'{type(var_4.globals).__module__}.{type(var_4.globals).__qualname__}' == 'builtins.dict'
    assert len(var_4.globals) == 6
    assert var_4.loader is None
    assert f'{type(var_4.cache).__module__}.{type(var_4.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_4.cache) == 0
    assert var_4.bytecode_cache is None
    assert var_4.auto_reload is True
    assert var_4.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(var_4.extensions).__module__}.{type(var_4.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_4.extensions) == 1
    assert var_4.is_async is False
    assert module_1.BLOCK_END_STRING == '%}'
    assert module_1.BLOCK_START_STRING == '{%'
    assert module_1.COMMENT_END_STRING == '#}'
    assert module_1.COMMENT_START_STRING == '{#'
    assert f'{type(module_1.DEFAULT_FILTERS).__module__}.{type(module_1.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_1.DEFAULT_FILTERS) == 54
    assert f'{type(module_1.DEFAULT_NAMESPACE).__module__}.{type(module_1.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_1.DEFAULT_NAMESPACE) == 6
    assert module_1.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_1.DEFAULT_TESTS).__module__}.{type(module_1.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_1.DEFAULT_TESTS) == 39
    assert module_1.KEEP_TRAILING_NEWLINE is False
    assert module_1.LINE_COMMENT_PREFIX is None
    assert module_1.LINE_STATEMENT_PREFIX is None
    assert module_1.LSTRIP_BLOCKS is False
    assert module_1.NEWLINE_SEQUENCE == '\n'
    assert module_1.TRIM_BLOCKS is False
    assert module_1.VARIABLE_END_STRING == '}}'
    assert module_1.VARIABLE_START_STRING == '{{'
    assert f'{type(module_1.missing).__module__}.{type(module_1.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_1.Environment.sandboxed is False
    assert module_1.Environment.overlayed is False
    assert module_1.Environment.linked_to is None
    assert module_1.Environment.shared is False
    assert f'{type(module_1.Environment.lexer).__module__}.{type(module_1.Environment.lexer).__qualname__}' == 'builtins.property'
    var_5 = bool('double' in var_4.filters)
    var_6 = 'double'
    with pytest.raises(KeyError):
        var_7 = var_4.filters[var_6]