# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.environment as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0.ExtensionLoaderMixin()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.StrictEnvironment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
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
    assert len(var_0.filters) == 56
    assert f'{type(var_0.tests).__module__}.{type(var_0.tests).__qualname__}' == 'builtins.dict'
    assert len(var_0.tests) == 39
    assert f'{type(var_0.globals).__module__}.{type(var_0.globals).__qualname__}' == 'builtins.dict'
    assert len(var_0.globals) == 8
    assert var_0.loader is None
    assert f'{type(var_0.cache).__module__}.{type(var_0.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_0.cache) == 0
    assert var_0.bytecode_cache is None
    assert var_0.auto_reload is True
    assert var_0.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_0.datetime_format == '%Y-%m-%d'
    assert f'{type(var_0.extensions).__module__}.{type(var_0.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_0.extensions) == 5
    assert var_0.is_async is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = 'aBaMf'
    var_2 = 'c'
    var_3 = {var_1: var_1, var_1: var_0, var_1: var_0, var_2: var_0}
    module_0.ExtensionLoaderMixin(context=var_3)

def test_case_2():
    var_0 = module_0.StrictEnvironment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
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
    assert len(var_0.filters) == 56
    assert f'{type(var_0.tests).__module__}.{type(var_0.tests).__qualname__}' == 'builtins.dict'
    assert len(var_0.tests) == 39
    assert f'{type(var_0.globals).__module__}.{type(var_0.globals).__qualname__}' == 'builtins.dict'
    assert len(var_0.globals) == 8
    assert var_0.loader is None
    assert f'{type(var_0.cache).__module__}.{type(var_0.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_0.cache) == 0
    assert var_0.bytecode_cache is None
    assert var_0.auto_reload is True
    assert var_0.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_0.datetime_format == '%Y-%m-%d'
    assert f'{type(var_0.extensions).__module__}.{type(var_0.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_0.extensions) == 5
    assert var_0.is_async is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '58C\\1\ncf'
    module_0.ExtensionLoaderMixin(context=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_0.StrictEnvironment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
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
    assert len(var_0.filters) == 56
    assert f'{type(var_0.tests).__module__}.{type(var_0.tests).__qualname__}' == 'builtins.dict'
    assert len(var_0.tests) == 39
    assert f'{type(var_0.globals).__module__}.{type(var_0.globals).__qualname__}' == 'builtins.dict'
    assert len(var_0.globals) == 8
    assert var_0.loader is None
    assert f'{type(var_0.cache).__module__}.{type(var_0.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_0.cache) == 0
    assert var_0.bytecode_cache is None
    assert var_0.auto_reload is True
    assert var_0.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_0.datetime_format == '%Y-%m-%d'
    assert f'{type(var_0.extensions).__module__}.{type(var_0.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_0.extensions) == 5
    assert var_0.is_async is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = {}
    var_2 = module_0.StrictEnvironment()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
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
    assert len(var_2.filters) == 56
    assert f'{type(var_2.tests).__module__}.{type(var_2.tests).__qualname__}' == 'builtins.dict'
    assert len(var_2.tests) == 39
    assert f'{type(var_2.globals).__module__}.{type(var_2.globals).__qualname__}' == 'builtins.dict'
    assert len(var_2.globals) == 8
    assert var_2.loader is None
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.datetime_format == '%Y-%m-%d'
    assert f'{type(var_2.extensions).__module__}.{type(var_2.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_2.extensions) == 5
    assert var_2.is_async is False
    var_3 = 'cookiecutter'
    var_4 = '_extensions'
    var_5 = 'jinja2.ext.DebugExtension'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_3: var_7}
    var_9 = module_0.StrictEnvironment()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
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
    assert len(var_9.filters) == 56
    assert f'{type(var_9.tests).__module__}.{type(var_9.tests).__qualname__}' == 'builtins.dict'
    assert len(var_9.tests) == 39
    assert f'{type(var_9.globals).__module__}.{type(var_9.globals).__qualname__}' == 'builtins.dict'
    assert len(var_9.globals) == 8
    assert var_9.loader is None
    assert f'{type(var_9.cache).__module__}.{type(var_9.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_9.cache) == 0
    assert var_9.bytecode_cache is None
    assert var_9.auto_reload is True
    assert var_9.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_9.datetime_format == '%Y-%m-%d'
    assert f'{type(var_9.extensions).__module__}.{type(var_9.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_9.extensions) == 5
    assert var_9.is_async is False
    module_0.ExtensionLoaderMixin(context=var_8)