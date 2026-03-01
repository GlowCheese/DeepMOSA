# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0
import jinja2.filters as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.fields as module_4
import typesystem.schemas as module_5
import markupsafe as module_6

def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_1.do_striptags(var_0)
    assert var_1 == 'None'
    assert f'{type(module_1.F).__module__}.{type(module_1.F).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.K).__module__}.{type(module_1.K).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.V).__module__}.{type(module_1.V).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.FILTERS).__module__}.{type(module_1.FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FILTERS) == 54
    module_0.Form(env=var_1, schema=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'N-g'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    module_0.Jinja2Forms(package=var_0)

def test_case_3():
    var_0 = 'N-g'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'N-g'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_2 = None
    var_3 = ''
    var_4 = '\t\t<O'
    var_5 = '\\aBd\r7&GO2h'
    var_6 = {var_0: var_1, var_3: var_0, var_4: var_1, var_5: var_5}
    var_1.create_form(var_2, var_6)

def test_case_5():
    var_0 = 'N-g'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        var_1.load_template_env()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'Q'
    module_0.Jinja2Forms(directory=var_0, package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'forms/checkbox.html'
    var_1 = 'forms/select.html'
    var_2 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_3 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_4 = '<select name="{{ field_name }}">{{ value }}</select>'
    var_5 = {var_3: var_2, var_0: var_3, var_0: var_2, var_1: var_4}
    var_6 = module_2.DictLoader(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_6.mapping == {'<textarea name="{{ field_name }}">{{ value }}</textarea>': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">', 'forms/checkbox.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">', 'forms/select.html': '<select name="{{ field_name }}">{{ value }}</select>'}
    var_7 = True
    var_8 = module_3.Environment(autoescape=var_7, loader=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'jinja2.environment.Environment'
    assert var_8.block_start_string == '{%'
    assert var_8.block_end_string == '%}'
    assert var_8.variable_start_string == '{{'
    assert var_8.variable_end_string == '}}'
    assert var_8.comment_start_string == '{#'
    assert var_8.comment_end_string == '#}'
    assert var_8.line_statement_prefix is None
    assert var_8.line_comment_prefix is None
    assert var_8.trim_blocks is False
    assert var_8.lstrip_blocks is False
    assert var_8.newline_sequence == '\n'
    assert var_8.keep_trailing_newline is False
    assert var_8.optimized is True
    assert var_8.finalize is None
    assert var_8.autoescape is True
    assert f'{type(var_8.filters).__module__}.{type(var_8.filters).__qualname__}' == 'builtins.dict'
    assert len(var_8.filters) == 54
    assert f'{type(var_8.tests).__module__}.{type(var_8.tests).__qualname__}' == 'builtins.dict'
    assert len(var_8.tests) == 39
    assert f'{type(var_8.globals).__module__}.{type(var_8.globals).__qualname__}' == 'builtins.dict'
    assert len(var_8.globals) == 6
    assert f'{type(var_8.loader).__module__}.{type(var_8.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_8.cache).__module__}.{type(var_8.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_8.cache) == 0
    assert var_8.bytecode_cache is None
    assert var_8.auto_reload is True
    assert var_8.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_8.extensions == {}
    assert var_8.is_async is False
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
    var_9 = 'email'
    var_10 = 'John'
    var_11 = 'john@example.com'
    var_12 = module_4.String()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.allow_blank is False
    assert var_12.trim_whitespace is True
    assert var_12.max_length is None
    assert var_12.min_length is None
    assert var_12.format is None
    assert var_12.coerce_types is True
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_13 = {var_10: var_12}
    var_14 = module_5.Schema(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.fields).__module__}.{type(var_14.fields).__qualname__}' == 'builtins.dict'
    assert len(var_14.fields) == 1
    assert var_14.required == ['John']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_15 = {var_9: var_11}
    var_16 = module_0.Form(env=var_8, schema=var_14, values=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_16.env).__module__}.{type(var_16.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_16.schema).__module__}.{type(var_16.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_16.values == {}
    assert var_16.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_16.__html__()

def test_case_8():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_4 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_5 = '<select name="{{ field_name }}">{{ value }}</select>'
    var_6 = {var_0: var_3, var_1: var_4, var_1: var_3, var_2: var_5}
    var_7 = module_2.DictLoader(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_7.mapping == {'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">', 'forms/checkbox.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">', 'forms/select.html': '<select name="{{ field_name }}">{{ value }}</select>'}
    var_8 = True
    var_9 = module_3.Environment(autoescape=var_8, loader=var_7)
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
    assert var_9.autoescape is True
    assert f'{type(var_9.filters).__module__}.{type(var_9.filters).__qualname__}' == 'builtins.dict'
    assert len(var_9.filters) == 54
    assert f'{type(var_9.tests).__module__}.{type(var_9.tests).__qualname__}' == 'builtins.dict'
    assert len(var_9.tests) == 39
    assert f'{type(var_9.globals).__module__}.{type(var_9.globals).__qualname__}' == 'builtins.dict'
    assert len(var_9.globals) == 6
    assert f'{type(var_9.loader).__module__}.{type(var_9.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
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
    var_10 = 'email'
    var_11 = 'John'
    var_12 = 'john@example.com'
    var_13 = module_4.String()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.String'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.allow_blank is False
    assert var_13.trim_whitespace is True
    assert var_13.max_length is None
    assert var_13.min_length is None
    assert var_13.format is None
    assert var_13.coerce_types is True
    assert var_13.pattern is None
    assert var_13.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_14 = {var_11: var_13}
    var_15 = module_5.Schema(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.fields).__module__}.{type(var_15.fields).__qualname__}' == 'builtins.dict'
    assert len(var_15.fields) == 1
    assert var_15.required == ['John']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_16 = {var_10: var_12}
    var_17 = module_0.Form(env=var_9, schema=var_15, values=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_17.env).__module__}.{type(var_17.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_17.schema).__module__}.{type(var_17.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_17.values == {}
    assert var_17.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_18 = var_17.__html__()
    assert len(var_9.cache) == 1
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'markupsafe.Markup'
    assert len(var_18) == 44
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'