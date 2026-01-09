# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_5
import inspect as module_6

import jinja2.environment as module_1
import markupsafe as module_3
import pytest
import typesystem.fields as module_4
import typesystem.forms as module_0
import typesystem.schemas as module_2


def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

def test_case_1():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = module_2.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Form(env=var_0, schema=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_3.schema).__module__}.{type(var_3.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.values is None
    assert var_3.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4 = var_3.__html__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'markupsafe.Markup'
    assert len(var_4) == 0
    assert f'{type(module_3.Markup.escape).__module__}.{type(module_3.Markup.escape).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '3L\x0b*+G'
    module_0.Jinja2Forms(package=var_0)

def test_case_3():
    var_0 = 'N-g'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'

def test_case_4():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = module_2.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Jinja2Forms(directory=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    var_4 = module_0.Form(env=var_0, schema=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = var_3.create_form(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_5.env).__module__}.{type(var_5.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values is None
    assert var_5.errors is None
    var_6 = None
    var_7 = var_4.__html__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'markupsafe.Markup'
    assert len(var_7) == 0
    assert f'{type(module_3.Markup.escape).__module__}.{type(module_3.Markup.escape).__qualname__}' == 'builtins.method'
    var_8 = var_4.validate(var_6)
    assert f'{type(var_4.errors).__module__}.{type(var_4.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.errors) == 1
    assert var_4.data is None
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    var_9 = var_4.template_for_field(var_6)
    assert var_9 == 'forms/input.html'
    var_10 = module_0.Form(env=var_6, schema=var_2, values=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.forms.Form'
    assert var_10.env is None
    assert f'{type(var_10.schema).__module__}.{type(var_10.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_10.values is None
    assert var_10.errors is None
    var_11 = var_4.__html__()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'markupsafe.Markup'
    assert len(var_11) == 0

def test_case_5():
    var_0 = 'N-g'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        var_1.load_template_env()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '10S|="Xp_\x0cD5Q$A5'
    module_0.Jinja2Forms(directory=var_0, package=var_0)

def test_case_7():
    var_0 = {}
    var_1 = module_2.Schema(var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = module_0.Form(env=var_0, schema=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.forms.Form'
    assert var_2.env == {}
    assert f'{type(var_2.schema).__module__}.{type(var_2.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.values is None
    assert var_2.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_3 = var_2.validate()
    assert f'{type(var_2.errors).__module__}.{type(var_2.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.errors) == 1
    assert var_2.data is None
    var_4 = var_2.__html__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'markupsafe.Markup'
    assert len(var_4) == 0
    assert f'{type(module_3.Markup.escape).__module__}.{type(module_3.Markup.escape).__qualname__}' == 'builtins.method'
    var_5 = 'date'
    var_6 = {}
    var_7 = module_4.String(format=var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format == 'date'
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = var_2.template_for_field(var_7)
    assert var_8 == 'forms/input.html'

def test_case_8():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = module_2.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Form(env=var_0, schema=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_3.schema).__module__}.{type(var_3.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.values is None
    assert var_3.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'

def test_case_9():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = module_2.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Form(env=var_0, schema=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_3.schema).__module__}.{type(var_3.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.values is None
    assert var_3.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4 = 'uRl'
    var_5 = module_4.String(format=var_4, **var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length is None
    assert var_5.min_length is None
    assert var_5.format == 'uRl'
    assert var_5.coerce_types is True
    assert var_5.pattern is None
    assert var_5.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = var_3.input_type_for_field(var_5)
    assert var_6 == 'text'

def test_case_10():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = {}
    var_3 = module_2.Schema(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_0, schema=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = var_4.input_type_for_field(var_0)
    assert var_5 == 'text'

def test_case_11():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = module_2.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Form(env=var_0, schema=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_3.schema).__module__}.{type(var_3.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.values is None
    assert var_3.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4 = var_3.__str__()
    assert var_4 == ''

def test_case_12():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = module_2.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Form(env=var_0, schema=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_3.schema).__module__}.{type(var_3.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.values is None
    assert var_3.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4 = 'date'
    var_5 = module_4.String(format=var_4, **var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length is None
    assert var_5.min_length is None
    assert var_5.format == 'date'
    assert var_5.coerce_types is True
    assert var_5.pattern is None
    assert var_5.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = var_3.template_for_field(var_5)
    assert var_6 == 'forms/input.html'

def test_case_13():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = module_2.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Form(env=var_0, schema=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_3.schema).__module__}.{type(var_3.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.values is None
    assert var_3.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4 = None
    var_5 = var_3.__html__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'markupsafe.Markup'
    assert len(var_5) == 0
    assert f'{type(module_3.Markup.escape).__module__}.{type(module_3.Markup.escape).__qualname__}' == 'builtins.method'
    var_6 = var_3.validate(var_4)
    assert f'{type(var_3.errors).__module__}.{type(var_3.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.errors) == 1
    assert var_3.data is None
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    with pytest.raises(AssertionError):
        var_3.validate()

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = module_2.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Form(env=var_0, schema=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_3.schema).__module__}.{type(var_3.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.values is None
    assert var_3.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4 = 'J'
    var_3.render_field(field_name=var_4, field=var_2)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = {}
    var_3 = module_2.Schema(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_0, schema=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = 'Optional Field'
    var_6 = True
    var_7 = 'title'
    var_8 = 'allow_null'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_4.String(**var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.default is None
    assert var_10.title == 'Optional Field'
    assert var_10.description == ''
    assert var_10.allow_null is True
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length is None
    assert var_10.min_length is None
    assert var_10.format is None
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_11 = 'optional_field'
    var_12 = None
    var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_12)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = {}
    var_3 = module_2.Schema(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_0, schema=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = 'Required Field'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_4.String(**var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == 'Required Field'
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format is None
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = 'required_field'
    var_10 = None
    var_4.render_field(field_name=var_9, field=var_8, value=var_10, error=var_10)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = {}
    var_3 = module_2.Schema(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_0, schema=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = 'option1'
    var_6 = 'Option 1'
    var_7 = (var_5, var_6)
    var_8 = 'Option 2'
    var_9 = (var_8, var_8)
    var_10 = [var_7, var_9]
    var_11 = {}
    var_12 = module_4.Choice(choices=var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Choice'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.choices == [('option1', 'Option 1'), ('Option 2', 'Option 2')]
    assert var_12.coerce_types is True
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_13 = 'choice_field'
    var_14 = None
    var_4.render_field(field_name=var_13, field=var_12, value=var_5, error=var_14)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = {}
    var_3 = module_2.Schema(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_0, schema=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = 'Agree to terms'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_4.Boolean(**var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_8.title == 'Agree to terms'
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.coerce_types is True
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_4.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_4.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_9 = 'agree'
    var_10 = True
    var_11 = None
    var_4.render_field(field_name=var_9, field=var_8, value=var_10, error=var_11)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = module_1.Environment()
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
    var_1 = {}
    var_2 = {}
    var_3 = module_2.Schema(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_0, schema=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = 'text'
    var_6 = {}
    var_7 = module_4.String(format=var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format == 'text'
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = 'description'
    var_9 = 'Some text'
    var_10 = None
    var_4.render_field(field_name=var_8, field=var_7, value=var_9, error=var_10)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_1.Environment()
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
    var_1 = module_4.Field()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.Field.errors == {}
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_2.Schema(var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert f'{type(var_4.required).__module__}.{type(var_4.required).__qualname__}' == 'builtins.list'
    assert len(var_4.required) == 1
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_0.Form(env=var_0, schema=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_5.env).__module__}.{type(var_5.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values is None
    assert var_5.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5.__html__()

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_5._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = {}
    var_2 = module_2.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Form(env=var_0, schema=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'enum._EnumDict'
    assert len(var_3.env) == 0
    assert f'{type(var_3.schema).__module__}.{type(var_3.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.values is None
    assert var_3.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    module_6.getmembers(var_3, var_0)