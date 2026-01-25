# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import jinja2.loaders as module_3
import jinja2.environment as module_4
import jinja2.filters as module_5
import re as module_6

def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'equred_fW*ld'
    var_1 = module_1.String()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = {var_0: var_1}
    var_3 = module_2.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ['equred_fW*ld']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_3, schema=var_3, values=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values == {}
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4.render_fields()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'pn'
    module_0.Jinja2Forms(package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'pn'
    module_0.Jinja2Forms(directory=var_0, package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = "'~?&+Yj15! 3Zut%|\x0c"
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_2 = 'pn'
    module_0.Jinja2Forms(directory=var_2, package=var_2)

def test_case_5():
    var_0 = 'ReaPN'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        var_1.load_template_env()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'ReaPN'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_1.create_form(var_1)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'required_field'
    var_1 = module_1.String()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = {var_0: var_1}
    var_3 = module_2.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ['required_field']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_3, schema=var_3, values=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values == {}
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = var_4.validate(var_2)
    assert var_4.values is None
    assert f'{type(var_4.errors).__module__}.{type(var_4.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.errors) == 1
    assert f'{type(var_4.data).__module__}.{type(var_4.data).__qualname__}' == 'builtins.dict'
    assert len(var_4.data) == 1
    var_4.render_field(field_name=var_0, field=var_1, value=var_1, error=var_5)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_3.BaseLoader()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert module_3.BaseLoader.has_source_access is True
    var_1 = module_4.Environment(loader=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.environment.Environment'
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
    assert var_1.keep_trailing_newline is False
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 54
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 6
    assert f'{type(var_1.loader).__module__}.{type(var_1.loader).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    var_2 = 'password'
    var_3 = module_1.String(format=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'password'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_2: var_3}
    var_5 = module_2.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['password']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = 'secret'
    var_7 = {var_2: var_6}
    var_8 = module_0.Form(env=var_1, schema=var_5, values=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_8.schema).__module__}.{type(var_8.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.values == {'password': 'secret'}
    assert var_8.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_9 = var_8.validate(var_7)
    assert var_8.data == {'password': 'secret'}
    var_10 = var_5.fields[var_2]
    var_11 = None
    var_8.render_field(field_name=var_2, field=var_10, value=var_6, error=var_11)

def test_case_9():
    var_0 = 'forms/select.html'
    var_1 = '<select />'
    var_2 = {var_0: var_1}
    var_3 = module_3.DictLoader(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_3.mapping == {'forms/select.html': '<select />'}
    var_4 = module_4.Environment(loader=var_3)
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
    assert len(var_4.filters) == 54
    assert f'{type(var_4.tests).__module__}.{type(var_4.tests).__qualname__}' == 'builtins.dict'
    assert len(var_4.tests) == 39
    assert f'{type(var_4.globals).__module__}.{type(var_4.globals).__qualname__}' == 'builtins.dict'
    assert len(var_4.globals) == 6
    assert f'{type(var_4.loader).__module__}.{type(var_4.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_4.cache).__module__}.{type(var_4.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_4.cache) == 0
    assert var_4.bytecode_cache is None
    assert var_4.auto_reload is True
    assert var_4.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_4.extensions == {}
    assert var_4.is_async is False
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    var_5 = 'color'
    var_6 = 'red'
    var_7 = 'green'
    var_8 = 'blue'
    var_9 = [var_6, var_7, var_8]
    var_10 = module_1.Choice(choices=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Choice'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.choices == [('red', 'red'), ('green', 'green'), ('blue', 'blue')]
    assert var_10.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_11 = {var_5: var_10}
    var_12 = module_2.Schema(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.fields).__module__}.{type(var_12.fields).__qualname__}' == 'builtins.dict'
    assert len(var_12.fields) == 1
    assert var_12.required == ['color']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_13 = {var_5: var_6}
    var_14 = module_0.Form(env=var_4, schema=var_12, values=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_14.env).__module__}.{type(var_14.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_14.schema).__module__}.{type(var_14.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_14.values == {'color': 'red'}
    assert var_14.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_15 = {var_5: var_6}
    var_16 = var_14.validate(var_15)
    assert var_14.data == {'color': 'red'}
    var_17 = var_14.render_fields()
    assert var_17 == '<select />'
    assert len(var_4.cache) == 1

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_3.BaseLoader()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert module_3.BaseLoader.has_source_access is True
    var_1 = module_4.Environment(loader=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.environment.Environment'
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
    assert var_1.keep_trailing_newline is False
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 54
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 6
    assert f'{type(var_1.loader).__module__}.{type(var_1.loader).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    var_2 = 'agree'
    var_3 = module_1.Boolean()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_4 = {var_2: var_3}
    var_5 = module_2.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['agree']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = True
    var_7 = {var_2: var_6}
    var_8 = module_0.Form(env=var_1, schema=var_5, values=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_8.schema).__module__}.{type(var_8.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.values == {'agree': True}
    assert var_8.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_9 = {var_2: var_6}
    var_10 = var_8.validate(var_9)
    assert var_8.data == {'agree': True}
    var_11 = var_5.fields[var_2]
    var_12 = None
    var_8.render_field(field_name=var_2, field=var_11, value=var_6, error=var_12)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_3.BaseLoader()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert module_3.BaseLoader.has_source_access is True
    var_1 = module_4.Environment(loader=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.environment.Environment'
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
    assert var_1.keep_trailing_newline is False
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 54
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 6
    assert f'{type(var_1.loader).__module__}.{type(var_1.loader).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    var_2 = 'description'
    var_3 = 'text'
    var_4 = module_1.String(format=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is None
    assert var_4.format == 'text'
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = {var_2: var_4}
    var_6 = module_2.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['description']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = 'Long text'
    var_8 = {var_2: var_7}
    var_9 = module_0.Form(env=var_1, schema=var_6, values=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_9.schema).__module__}.{type(var_9.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.values == {'description': 'Long text'}
    assert var_9.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_10 = {var_2: var_7}
    var_11 = var_9.validate(var_10)
    assert var_9.data == {'description': 'Long text'}
    var_12 = var_6.fields[var_2]
    var_13 = None
    var_9.render_field(field_name=var_2, field=var_12, value=var_7, error=var_13)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_3.BaseLoader()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert module_3.BaseLoader.has_source_access is True
    var_1 = module_4.Environment(loader=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.environment.Environment'
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
    assert var_1.keep_trailing_newline is False
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 54
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 6
    assert f'{type(var_1.loader).__module__}.{type(var_1.loader).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    var_2 = 'required_field'
    var_3 = module_1.String()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format is None
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_2: var_3}
    var_5 = module_2.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['required_field']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = module_0.Form(env=var_1, schema=var_5, values=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_8.schema).__module__}.{type(var_8.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.values == {'required_field': 'value'}
    assert var_8.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_9 = var_5.fields[var_2]
    var_8.__str__()

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = module_3.BaseLoader()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert module_3.BaseLoader.has_source_access is True
    var_1 = 'required_field'
    var_2 = module_1.String()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['required_field']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = 'GK}QS7?JMTH]|9NJH'
    var_6 = {var_1: var_5}
    var_7 = module_0.Form(env=var_4, schema=var_4, values=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values == {'required_field': 'GK}QS7?JMTH]|9NJH'}
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = module_0.Jinja2Forms(directory=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    var_9 = var_7.validate()
    assert var_7.values is None
    assert f'{type(var_7.errors).__module__}.{type(var_7.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.errors) == 1
    assert var_7.data is None
    var_7.render_fields()

def test_case_14():
    var_0 = 'equred_fW*ld'
    var_1 = module_1.String()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = {var_0: var_1}
    var_3 = module_2.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ['equred_fW*ld']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_3, schema=var_3, values=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values == {}
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/input.html'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'equred_fW*ld'
    var_1 = module_1.String()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = {var_0: var_1}
    var_3 = module_2.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ['equred_fW*ld']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_3, schema=var_3, values=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values == {}
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    module_5.do_striptags(var_4)

def test_case_16():
    var_0 = 'forms/input.html'
    var_1 = '<input />'
    var_2 = {var_0: var_1}
    var_3 = module_3.DictLoader(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_3.mapping == {'forms/input.html': '<input />'}
    var_4 = module_4.Environment(loader=var_3)
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
    assert len(var_4.filters) == 54
    assert f'{type(var_4.tests).__module__}.{type(var_4.tests).__qualname__}' == 'builtins.dict'
    assert len(var_4.tests) == 39
    assert f'{type(var_4.globals).__module__}.{type(var_4.globals).__qualname__}' == 'builtins.dict'
    assert len(var_4.globals) == 6
    assert f'{type(var_4.loader).__module__}.{type(var_4.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_4.cache).__module__}.{type(var_4.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_4.cache) == 0
    assert var_4.bytecode_cache is None
    assert var_4.auto_reload is True
    assert var_4.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_4.extensions == {}
    assert var_4.is_async is False
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    var_5 = 'password'
    var_6 = module_1.String(format=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.String'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.allow_blank is False
    assert var_6.trim_whitespace is True
    assert var_6.max_length is None
    assert var_6.min_length is None
    assert var_6.format == 'password'
    assert var_6.coerce_types is True
    assert var_6.pattern is None
    assert var_6.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 1
    assert var_8.required == ['password']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_9 = 'secret'
    var_10 = {var_5: var_9}
    var_11 = module_0.Form(env=var_4, schema=var_8, values=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_11.env).__module__}.{type(var_11.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_11.schema).__module__}.{type(var_11.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.values == {'password': 'secret'}
    assert var_11.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_12 = {var_5: var_9}
    var_13 = var_11.validate(var_12)
    assert var_11.data == {'password': 'secret'}
    var_14 = var_11.render_fields()
    assert var_14 == '<input />'
    assert len(var_4.cache) == 1

def test_case_17():
    var_0 = module_3.BaseLoader()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert module_3.BaseLoader.has_source_access is True
    var_1 = module_4.Environment(loader=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.environment.Environment'
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
    assert var_1.keep_trailing_newline is False
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 54
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 6
    assert f'{type(var_1.loader).__module__}.{type(var_1.loader).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    var_2 = 'required_field'
    var_3 = module_1.String()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format is None
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_2: var_3}
    var_5 = module_2.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['required_field']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = module_0.Form(env=var_1, schema=var_5, values=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_8.schema).__module__}.{type(var_8.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.values == {'required_field': 'value'}
    assert var_8.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_9 = {var_2: var_6}
    var_10 = var_8.validate(var_9)
    assert var_8.data == {'required_field': 'value'}
    with pytest.raises(AssertionError):
        var_8.validate(var_10)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'equred_fW*ld'
    var_1 = module_1.String()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = {var_0: var_1}
    var_3 = module_2.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ['equred_fW*ld']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_3, schema=var_3, values=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values == {}
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = module_6.escape(var_0)
    assert var_5 == 'equred_fW\\*ld'
    assert module_6.ASCII == module_6.RegexFlag.ASCII
    assert module_6.A == module_6.RegexFlag.ASCII
    assert module_6.IGNORECASE == module_6.RegexFlag.IGNORECASE
    assert module_6.I == module_6.RegexFlag.IGNORECASE
    assert module_6.LOCALE == module_6.RegexFlag.LOCALE
    assert module_6.L == module_6.RegexFlag.LOCALE
    assert module_6.UNICODE == module_6.RegexFlag.UNICODE
    assert module_6.U == module_6.RegexFlag.UNICODE
    assert module_6.MULTILINE == module_6.RegexFlag.MULTILINE
    assert module_6.M == module_6.RegexFlag.MULTILINE
    assert module_6.DOTALL == module_6.RegexFlag.DOTALL
    assert module_6.S == module_6.RegexFlag.DOTALL
    assert module_6.VERBOSE == module_6.RegexFlag.VERBOSE
    assert module_6.X == module_6.RegexFlag.VERBOSE
    assert module_6.TEMPLATE == module_6.RegexFlag.TEMPLATE
    assert module_6.T == module_6.RegexFlag.TEMPLATE
    assert module_6.DEBUG == module_6.RegexFlag.DEBUG
    var_4.render_field(field_name=var_5, field=var_5, value=var_5)

def test_case_19():
    var_0 = 'forms/input.html'
    var_1 = '<input />'
    var_2 = {var_0: var_1}
    var_3 = module_3.DictLoader(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_3.mapping == {'forms/input.html': '<input />'}
    var_4 = module_4.Environment(loader=var_3)
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
    assert len(var_4.filters) == 54
    assert f'{type(var_4.tests).__module__}.{type(var_4.tests).__qualname__}' == 'builtins.dict'
    assert len(var_4.tests) == 39
    assert f'{type(var_4.globals).__module__}.{type(var_4.globals).__qualname__}' == 'builtins.dict'
    assert len(var_4.globals) == 6
    assert f'{type(var_4.loader).__module__}.{type(var_4.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_4.cache).__module__}.{type(var_4.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_4.cache) == 0
    assert var_4.bytecode_cache is None
    assert var_4.auto_reload is True
    assert var_4.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_4.extensions == {}
    assert var_4.is_async is False
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    var_5 = 'name'
    var_6 = module_1.String()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.String'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.allow_blank is False
    assert var_6.trim_whitespace is True
    assert var_6.max_length is None
    assert var_6.min_length is None
    assert var_6.format is None
    assert var_6.coerce_types is True
    assert var_6.pattern is None
    assert var_6.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 1
    assert var_8.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_9 = 'John'
    var_10 = {var_5: var_9}
    var_11 = module_0.Form(env=var_4, schema=var_8, values=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_11.env).__module__}.{type(var_11.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_11.schema).__module__}.{type(var_11.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.values == {'name': 'John'}
    assert var_11.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_12 = {var_5: var_9}
    var_13 = var_11.validate(var_12)
    assert var_11.data == {'name': 'John'}
    var_14 = var_11.render_fields()
    assert var_14 == '<input />'
    assert len(var_4.cache) == 1

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'read_only_field'
    var_1 = 'normal_field'
    var_2 = True
    var_3 = module_1.Field(read_only=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_4 = module_1.Field()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_2.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 2
    assert var_6.required == ['normal_field']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_4.Environment()
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
    assert var_7.loader is None
    assert f'{type(var_7.cache).__module__}.{type(var_7.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_7.cache) == 0
    assert var_7.bytecode_cache is None
    assert var_7.auto_reload is True
    assert var_7.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_7.extensions == {}
    assert var_7.is_async is False
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    var_8 = {}
    var_9 = module_0.Form(env=var_7, schema=var_6, values=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_9.schema).__module__}.{type(var_9.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.values == {}
    assert var_9.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_10 = var_9.validate()
    assert var_9.values is None
    assert f'{type(var_9.errors).__module__}.{type(var_9.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9.errors) == 1
    assert var_9.data is None
    var_9.render_fields()

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = True
    var_1 = 'default_value'
    var_2 = module_1.Field(default=var_1, allow_null=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.default == 'default_value'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_3 = module_4.Environment()
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
    assert var_3.loader is None
    assert f'{type(var_3.cache).__module__}.{type(var_3.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_3.cache) == 0
    assert var_3.bytecode_cache is None
    assert var_3.auto_reload is True
    assert var_3.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_3.extensions == {}
    assert var_3.is_async is False
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    var_4 = {}
    var_5 = module_2.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.fields == {}
    assert var_5.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = {}
    var_7 = module_0.Form(env=var_3, schema=var_5, values=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values == {}
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = 'test_field'
    var_7.render_field(field_name=var_8, field=var_2)