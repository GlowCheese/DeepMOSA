# Check out: https://github.com/GlowCheese/deepmosa
import locale as module_4
import platform as module_1
import uuid as module_3

import cookiecutter.extensions as module_0
import jinja2.environment as module_2
import pytest


def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.JsonifyExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.RandomStringExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_1.python_compiler()
    assert var_0 == 'GCC 14.2.0'
    module_0.SlugifyExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.UUIDExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.TimeExtension(var_0)

def test_case_6():
    var_0 = module_2.Environment()
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
    assert module_2.BLOCK_END_STRING == '%}'
    assert module_2.BLOCK_START_STRING == '{%'
    assert module_2.COMMENT_END_STRING == '#}'
    assert module_2.COMMENT_START_STRING == '{#'
    assert f'{type(module_2.DEFAULT_FILTERS).__module__}.{type(module_2.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_FILTERS) == 54
    assert f'{type(module_2.DEFAULT_NAMESPACE).__module__}.{type(module_2.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_NAMESPACE) == 6
    assert module_2.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_2.DEFAULT_TESTS).__module__}.{type(module_2.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_TESTS) == 39
    assert module_2.KEEP_TRAILING_NEWLINE is False
    assert module_2.LINE_COMMENT_PREFIX is None
    assert module_2.LINE_STATEMENT_PREFIX is None
    assert module_2.LSTRIP_BLOCKS is False
    assert module_2.NEWLINE_SEQUENCE == '\n'
    assert module_2.TRIM_BLOCKS is False
    assert module_2.VARIABLE_END_STRING == '}}'
    assert module_2.VARIABLE_START_STRING == '{{'
    assert f'{type(module_2.missing).__module__}.{type(module_2.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_2.Environment.sandboxed is False
    assert module_2.Environment.overlayed is False
    assert module_2.Environment.linked_to is None
    assert module_2.Environment.shared is False
    assert f'{type(module_2.Environment.lexer).__module__}.{type(module_2.Environment.lexer).__qualname__}' == 'builtins.property'
    var_1 = module_0.SlugifyExtension(var_0)
    assert len(var_0.filters) == 55
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.extensions.SlugifyExtension'
    assert f'{type(var_1.environment).__module__}.{type(var_1.environment).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_SEPARATOR == '-'
    assert module_0.SlugifyExtension.identifier == 'cookiecutter.extensions.SlugifyExtension'
    var_2 = 'slugify'
    var_3 = var_0.filters[var_2]
    var_4 = 'Test String'
    var_5 = var_3(var_4)
    var_6 = bool('slugify' in var_0.filters)
    assert var_6 is True

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_2.Environment()
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
    assert module_2.BLOCK_END_STRING == '%}'
    assert module_2.BLOCK_START_STRING == '{%'
    assert module_2.COMMENT_END_STRING == '#}'
    assert module_2.COMMENT_START_STRING == '{#'
    assert f'{type(module_2.DEFAULT_FILTERS).__module__}.{type(module_2.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_FILTERS) == 54
    assert f'{type(module_2.DEFAULT_NAMESPACE).__module__}.{type(module_2.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_NAMESPACE) == 6
    assert module_2.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_2.DEFAULT_TESTS).__module__}.{type(module_2.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_TESTS) == 39
    assert module_2.KEEP_TRAILING_NEWLINE is False
    assert module_2.LINE_COMMENT_PREFIX is None
    assert module_2.LINE_STATEMENT_PREFIX is None
    assert module_2.LSTRIP_BLOCKS is False
    assert module_2.NEWLINE_SEQUENCE == '\n'
    assert module_2.TRIM_BLOCKS is False
    assert module_2.VARIABLE_END_STRING == '}}'
    assert module_2.VARIABLE_START_STRING == '{{'
    assert f'{type(module_2.missing).__module__}.{type(module_2.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_2.Environment.sandboxed is False
    assert module_2.Environment.overlayed is False
    assert module_2.Environment.linked_to is None
    assert module_2.Environment.shared is False
    assert f'{type(module_2.Environment.lexer).__module__}.{type(module_2.Environment.lexer).__qualname__}' == 'builtins.property'
    var_1 = module_0.UUIDExtension(var_0)
    assert len(var_0.globals) == 7
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.extensions.UUIDExtension'
    assert f'{type(var_1.environment).__module__}.{type(var_1.environment).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_SEPARATOR == '-'
    assert module_0.UUIDExtension.identifier == 'cookiecutter.extensions.UUIDExtension'
    var_2 = bool('uuid4' in var_0.globals)
    assert var_2 is True
    var_3 = 'uuid4'
    var_4 = var_0.globals[var_3]
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = var_0.globals[var_3]
    var_8 = var_7()
    var_9 = 4
    var_10 = module_3.UUID(var_8, version=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'uuid.UUID'
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.UUID.bytes).__module__}.{type(module_3.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.bytes_le).__module__}.{type(module_3.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.fields).__module__}.{type(module_3.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_low).__module__}.{type(module_3.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_mid).__module__}.{type(module_3.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_hi_version).__module__}.{type(module_3.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_hi_variant).__module__}.{type(module_3.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_low).__module__}.{type(module_3.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time).__module__}.{type(module_3.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq).__module__}.{type(module_3.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.node).__module__}.{type(module_3.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.hex).__module__}.{type(module_3.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.urn).__module__}.{type(module_3.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.variant).__module__}.{type(module_3.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.version).__module__}.{type(module_3.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.int).__module__}.{type(module_3.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.UUID.is_safe).__module__}.{type(module_3.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    module_4.str(var_10)

def test_case_8():
    var_0 = module_2.Environment()
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
    assert module_2.BLOCK_END_STRING == '%}'
    assert module_2.BLOCK_START_STRING == '{%'
    assert module_2.COMMENT_END_STRING == '#}'
    assert module_2.COMMENT_START_STRING == '{#'
    assert f'{type(module_2.DEFAULT_FILTERS).__module__}.{type(module_2.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_FILTERS) == 54
    assert f'{type(module_2.DEFAULT_NAMESPACE).__module__}.{type(module_2.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_NAMESPACE) == 6
    assert module_2.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_2.DEFAULT_TESTS).__module__}.{type(module_2.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_TESTS) == 39
    assert module_2.KEEP_TRAILING_NEWLINE is False
    assert module_2.LINE_COMMENT_PREFIX is None
    assert module_2.LINE_STATEMENT_PREFIX is None
    assert module_2.LSTRIP_BLOCKS is False
    assert module_2.NEWLINE_SEQUENCE == '\n'
    assert module_2.TRIM_BLOCKS is False
    assert module_2.VARIABLE_END_STRING == '}}'
    assert module_2.VARIABLE_START_STRING == '{{'
    assert f'{type(module_2.missing).__module__}.{type(module_2.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_2.Environment.sandboxed is False
    assert module_2.Environment.overlayed is False
    assert module_2.Environment.linked_to is None
    assert module_2.Environment.shared is False
    assert f'{type(module_2.Environment.lexer).__module__}.{type(module_2.Environment.lexer).__qualname__}' == 'builtins.property'
    var_1 = module_0.RandomStringExtension(var_0)
    assert len(var_0.globals) == 7
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.extensions.RandomStringExtension'
    assert f'{type(var_1.environment).__module__}.{type(var_1.environment).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_SEPARATOR == '-'
    assert module_0.RandomStringExtension.identifier == 'cookiecutter.extensions.RandomStringExtension'
    var_2 = bool('random_ascii_string' in var_0.globals)
    assert var_2 is True
    var_3 = 'random_ascii_string'
    var_4 = var_0.globals[var_3]
    var_5 = 10
    var_6 = False
    var_7 = var_4(var_5, var_6)
    var_8 = len(var_7)
    assert var_8 == 10
    var_9 = 15
    var_10 = True
    var_11 = var_4(var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 15

def test_case_9():
    var_0 = module_2.Environment()
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
    assert module_2.BLOCK_END_STRING == '%}'
    assert module_2.BLOCK_START_STRING == '{%'
    assert module_2.COMMENT_END_STRING == '#}'
    assert module_2.COMMENT_START_STRING == '{#'
    assert f'{type(module_2.DEFAULT_FILTERS).__module__}.{type(module_2.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_FILTERS) == 54
    assert f'{type(module_2.DEFAULT_NAMESPACE).__module__}.{type(module_2.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_NAMESPACE) == 6
    assert module_2.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_2.DEFAULT_TESTS).__module__}.{type(module_2.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_TESTS) == 39
    assert module_2.KEEP_TRAILING_NEWLINE is False
    assert module_2.LINE_COMMENT_PREFIX is None
    assert module_2.LINE_STATEMENT_PREFIX is None
    assert module_2.LSTRIP_BLOCKS is False
    assert module_2.NEWLINE_SEQUENCE == '\n'
    assert module_2.TRIM_BLOCKS is False
    assert module_2.VARIABLE_END_STRING == '}}'
    assert module_2.VARIABLE_START_STRING == '{{'
    assert f'{type(module_2.missing).__module__}.{type(module_2.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_2.Environment.sandboxed is False
    assert module_2.Environment.overlayed is False
    assert module_2.Environment.linked_to is None
    assert module_2.Environment.shared is False
    assert f'{type(module_2.Environment.lexer).__module__}.{type(module_2.Environment.lexer).__qualname__}' == 'builtins.property'
    var_1 = module_0.RandomStringExtension(var_0)
    assert len(var_0.globals) == 7
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.extensions.RandomStringExtension'
    assert f'{type(var_1.environment).__module__}.{type(var_1.environment).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_SEPARATOR == '-'
    assert module_0.RandomStringExtension.identifier == 'cookiecutter.extensions.RandomStringExtension'
    var_2 = bool('random_ascii_string' in var_0.globals)
    assert var_2 is True
    var_3 = 'random_ascii_string'
    var_4 = var_0.globals[var_3]
    var_5 = False
    with pytest.raises(TypeError):
        var_6 = var_4(var_4, var_5)