# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import jinja2.environment as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = module_0.Environment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.environment.Environment'
    assert var_0.block_start_string == '{%'
    assert var_0.block_end_string == '%}'
    assert var_0.variable_start_string == '{{'
    assert var_0.variable_end_string == '}}'
    assert var_0.comment_start_string == '{#'
    assert var_0.comment_end_string == '#}'
    assert var_0.line_statement_prefix is None
    assert var_0.line_comment_prefix is None
    assert var_0.trim_blocks is False
    assert var_0.lstrip_blocks is False
    assert var_0.newline_sequence == '\n'
    assert var_0.keep_trailing_newline is False
    assert var_0.optimized is True
    assert var_0.finalize is None
    assert var_0.autoescape is False
    assert f'{type(var_0.filters).__module__}.{type(var_0.filters).__qualname__}' == 'builtins.dict'
    assert len(var_0.filters) == 54
    assert f'{type(var_0.tests).__module__}.{type(var_0.tests).__qualname__}' == 'builtins.dict'
    assert len(var_0.tests) == 39
    assert f'{type(var_0.globals).__module__}.{type(var_0.globals).__qualname__}' == 'builtins.dict'
    assert len(var_0.globals) == 6
    assert var_0.loader is None
    assert f'{type(var_0.cache).__module__}.{type(var_0.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_0.cache) == 0
    assert var_0.bytecode_cache is None
    assert var_0.auto_reload is True
    assert var_0.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_0.extensions == {}
    assert var_0.is_async is False
    assert module_0.BLOCK_END_STRING == '%}'
    assert module_0.BLOCK_START_STRING == '{%'
    assert module_0.COMMENT_END_STRING == '#}'
    assert module_0.COMMENT_START_STRING == '{#'
    assert f'{type(module_0.DEFAULT_FILTERS).__module__}.{type(module_0.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_0.DEFAULT_FILTERS) == 54
    assert f'{type(module_0.DEFAULT_NAMESPACE).__module__}.{type(module_0.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_0.DEFAULT_NAMESPACE) == 6
    assert module_0.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_0.DEFAULT_TESTS).__module__}.{type(module_0.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_0.DEFAULT_TESTS) == 39
    assert module_0.KEEP_TRAILING_NEWLINE is False
    assert module_0.LINE_COMMENT_PREFIX is None
    assert module_0.LINE_STATEMENT_PREFIX is None
    assert module_0.LSTRIP_BLOCKS is False
    assert module_0.NEWLINE_SEQUENCE == '\n'
    assert module_0.TRIM_BLOCKS is False
    assert module_0.VARIABLE_END_STRING == '}}'
    assert module_0.VARIABLE_START_STRING == '{{'
    assert f'{type(module_0.missing).__module__}.{type(module_0.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_0.Environment.sandboxed is False
    assert module_0.Environment.overlayed is False
    assert module_0.Environment.linked_to is None
    assert module_0.Environment.shared is False
    assert f'{type(module_0.Environment.lexer).__module__}.{type(module_0.Environment.lexer).__qualname__}' == 'builtins.property'

def test_case_1():
    var_0 = 'st'
    with pytest.raises(OSError):
        module_1.make_sure_path_exists(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_1.make_sure_path_exists(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_1.force_delete(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_1.rmtree(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_1.make_executable(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_1.simple_filter(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_1.create_tmp_repo_dir(var_0)

def test_case_8():
    var_0 = {}
    var_1 = module_1.create_env_with_context(var_0)
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
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert f'{type(module_1.logger).__module__}.{type(module_1.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_1.logger.filters == []
    assert module_1.logger.name == 'cookiecutter.utils'
    assert module_1.logger.level == 0
    assert f'{type(module_1.logger.parent).__module__}.{type(module_1.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_1.logger.propagate is True
    assert module_1.logger.handlers == []
    assert module_1.logger.disabled is False
    assert f'{type(module_1.logger.manager).__module__}.{type(module_1.logger.manager).__qualname__}' == 'logging.Manager'