# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import jinja2.environment as module_3
import markupsafe as module_4
import jinja2.loaders as module_5

def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

def test_case_1():
    pass

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'gu.Um;'
    module_0.Jinja2Forms(package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'name'
    var_1 = module_1.String(format=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'name'
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
    assert var_3.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_0, schema=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert var_4.env == 'name'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4.__html__()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'lSAj3-Px>SX)TmcSrmr'
    module_0.Jinja2Forms(directory=var_0, package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'gu.Um;'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_1.load_template_env(directory=var_1, package=var_1)

def test_case_6():
    var_0 = 'gu.Um;'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_2 = None
    with pytest.raises(AssertionError):
        var_1.load_template_env(directory=var_2, package=var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'mZNe'
    var_1 = None
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_2.env).__module__}.{type(var_2.env).__qualname__}' == 'jinja2.environment.Environment'
    var_3 = module_1.String(format=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'mZNe'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_0: var_3}
    var_5 = module_2.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['mZNe']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = var_2.create_form(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_6.env).__module__}.{type(var_6.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_6.schema).__module__}.{type(var_6.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.values is None
    assert var_6.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_6.__html__()

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'HXname'
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
    var_3 = module_1.Object(properties=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.properties).__module__}.{type(var_3.properties).__qualname__}' == 'builtins.dict'
    assert len(var_3.properties) == 1
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is True
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = {var_0: var_3}
    var_5 = module_2.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['HXname']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = module_3.Environment()
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
    assert var_6.loader is None
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
    var_7 = module_0.Form(env=var_6, schema=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_7.__html__()

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'me'
    var_1 = module_1.String(format=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'me'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = {}
    var_3 = module_2.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_1, schema=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = var_4.__html__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'markupsafe.Markup'
    assert len(var_5) == 0
    assert f'{type(module_4.Markup.escape).__module__}.{type(module_4.Markup.escape).__qualname__}' == 'builtins.method'
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
    assert f'{type(module_4.annotations).__module__}.{type(module_4.annotations).__qualname__}' == '__future__._Feature'
    assert module_4.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_4.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_4.annotations.compiler_flag == 16777216
    var_7 = var_4.validate()
    assert f'{type(var_4.errors).__module__}.{type(var_4.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.errors) == 1
    assert var_4.data is None
    module_0.Jinja2Forms(directory=var_0, package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'name'
    var_1 = module_3.Environment()
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
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
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
    var_2 = False
    var_3 = module_1.String(format=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'name'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_0: var_3}
    var_5 = module_2.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = module_0.Form(env=var_1, schema=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_6.env).__module__}.{type(var_6.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_6.schema).__module__}.{type(var_6.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.values is None
    assert var_6.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_7 = module_1.String()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format is None
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    var_8 = {var_0: var_7}
    var_9 = module_1.Object(properties=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Object'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.properties).__module__}.{type(var_9.properties).__qualname__}' == 'builtins.dict'
    assert len(var_9.properties) == 1
    assert var_9.pattern_properties == {}
    assert var_9.additional_properties is True
    assert var_9.property_names is None
    assert var_9.min_properties is None
    assert var_9.max_properties is None
    assert var_9.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_10 = var_6.validate()
    assert f'{type(var_6.errors).__module__}.{type(var_6.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.errors) == 1
    assert var_6.data is None
    var_6.__html__()

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'name'
    var_1 = module_3.Environment()
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
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
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
    var_2 = True
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
    var_4 = {var_0: var_3}
    var_5 = module_2.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = module_0.Form(env=var_1, schema=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_6.env).__module__}.{type(var_6.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_6.schema).__module__}.{type(var_6.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.values is None
    assert var_6.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_7 = var_6.validate(var_4)
    assert f'{type(var_6.errors).__module__}.{type(var_6.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.errors) == 1
    assert f'{type(var_6.data).__module__}.{type(var_6.data).__qualname__}' == 'builtins.dict'
    assert len(var_6.data) == 1
    var_6.__html__()

def test_case_12():
    var_0 = 'name'
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
    var_2 = module_3.Environment()
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
    assert var_2.loader is None
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
    var_3 = True
    var_4 = module_1.String()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is None
    assert var_4.format is None
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    var_5 = {var_0: var_4}
    var_6 = module_2.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_3.Environment()
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
    var_8 = module_0.Form(env=var_7, schema=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_8.schema).__module__}.{type(var_8.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.values is None
    assert var_8.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_9 = {}
    var_10 = var_8.validate()
    assert f'{type(var_8.errors).__module__}.{type(var_8.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8.errors) == 1
    assert var_8.data is None
    with pytest.raises(AssertionError):
        var_8.validate(var_9)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'p'
    var_1 = module_1.String(format=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'p'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = {}
    var_3 = module_2.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_1, schema=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = var_4.__html__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'markupsafe.Markup'
    assert len(var_5) == 0
    assert f'{type(module_4.Markup.escape).__module__}.{type(module_4.Markup.escape).__qualname__}' == 'builtins.method'
    module_0.Jinja2Forms(directory=var_0, package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'name'
    var_1 = module_3.Environment()
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
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
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
    var_2 = True
    var_3 = module_1.String(format=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'name'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_0: var_3}
    var_5 = module_2.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = module_0.Form(env=var_1, schema=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_6.env).__module__}.{type(var_6.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_6.schema).__module__}.{type(var_6.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.values is None
    assert var_6.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_7 = '$'
    var_8 = module_1.String()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
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
    var_9 = {var_0: var_8}
    var_10 = module_1.Object(properties=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Object'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.properties).__module__}.{type(var_10.properties).__qualname__}' == 'builtins.dict'
    assert len(var_10.properties) == 1
    assert var_10.pattern_properties == {}
    assert var_10.additional_properties is True
    assert var_10.property_names is None
    assert var_10.min_properties is None
    assert var_10.max_properties is None
    assert var_10.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_11 = {var_7: var_10}
    var_12 = module_2.Schema(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.fields).__module__}.{type(var_12.fields).__qualname__}' == 'builtins.dict'
    assert len(var_12.fields) == 1
    assert var_12.required == ['$']
    var_13 = None
    var_14 = var_6.template_for_field(var_13)
    assert var_14 == 'forms/input.html'
    var_15 = module_3.Environment()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'jinja2.environment.Environment'
    assert var_15.block_start_string == '{%'
    assert var_15.block_end_string == '%}'
    assert var_15.variable_start_string == '{{'
    assert var_15.variable_end_string == '}}'
    assert var_15.comment_start_string == '{#'
    assert var_15.comment_end_string == '#}'
    assert var_15.line_statement_prefix is None
    assert var_15.line_comment_prefix is None
    assert var_15.trim_blocks is False
    assert var_15.lstrip_blocks is False
    assert var_15.newline_sequence == '\n'
    assert var_15.keep_trailing_newline is False
    assert var_15.optimized is True
    assert var_15.finalize is None
    assert var_15.autoescape is False
    assert f'{type(var_15.filters).__module__}.{type(var_15.filters).__qualname__}' == 'builtins.dict'
    assert len(var_15.filters) == 54
    assert f'{type(var_15.tests).__module__}.{type(var_15.tests).__qualname__}' == 'builtins.dict'
    assert len(var_15.tests) == 39
    assert f'{type(var_15.globals).__module__}.{type(var_15.globals).__qualname__}' == 'builtins.dict'
    assert len(var_15.globals) == 6
    assert var_15.loader is None
    assert f'{type(var_15.cache).__module__}.{type(var_15.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_15.cache) == 0
    assert var_15.bytecode_cache is None
    assert var_15.auto_reload is True
    assert var_15.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_15.extensions == {}
    assert var_15.is_async is False
    var_16 = module_0.Form(env=var_15, schema=var_12)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_16.env).__module__}.{type(var_16.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_16.schema).__module__}.{type(var_16.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_16.values is None
    assert var_16.errors is None
    var_16.__html__()

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'name'
    var_1 = module_3.Environment()
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
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
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
    var_2 = True
    var_3 = None
    var_4 = module_1.String(allow_blank=var_2, format=var_3, coerce_types=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.default == ''
    assert var_4.allow_blank is True
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is None
    assert var_4.format is None
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = {var_0: var_4}
    var_6 = module_2.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_1, schema=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_7.__html__()

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'name'
    var_1 = False
    var_2 = module_1.String(format=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'name'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = module_2.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_0.Form(env=var_0, schema=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert var_5.env == 'name'
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values is None
    assert var_5.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
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
    var_5.__str__()

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_5 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <textarea id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>{{ value }}</textarea>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_6 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <select id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>\n                {% for choice in field.choices %}\n                    <option value="{{ choice.value }}" {% if choice.value == value %}selected{% endif %}>{{ choice.text }}</option>\n                {% endfor %}\n            </select>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_7 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" value="true" {% if value %}checked{% endif %} {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_5.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'forms/input.html': '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        ', 'forms/textarea.html': '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <textarea id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>{{ value }}</textarea>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        ', 'forms/select.html': '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <select id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>\n                {% for choice in field.choices %}\n                    <option value="{{ choice.value }}" {% if choice.value == value %}selected{% endif %}>{{ choice.text }}</option>\n                {% endfor %}\n            </select>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        ', 'forms/checkbox.html': '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" value="true" {% if value %}checked{% endif %} {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '}
    var_10 = module_3.Environment(loader=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'jinja2.environment.Environment'
    assert var_10.block_start_string == '{%'
    assert var_10.block_end_string == '%}'
    assert var_10.variable_start_string == '{{'
    assert var_10.variable_end_string == '}}'
    assert var_10.comment_start_string == '{#'
    assert var_10.comment_end_string == '#}'
    assert var_10.line_statement_prefix is None
    assert var_10.line_comment_prefix is None
    assert var_10.trim_blocks is False
    assert var_10.lstrip_blocks is False
    assert var_10.newline_sequence == '\n'
    assert var_10.keep_trailing_newline is False
    assert var_10.optimized is True
    assert var_10.finalize is None
    assert var_10.autoescape is False
    assert f'{type(var_10.filters).__module__}.{type(var_10.filters).__qualname__}' == 'builtins.dict'
    assert len(var_10.filters) == 54
    assert f'{type(var_10.tests).__module__}.{type(var_10.tests).__qualname__}' == 'builtins.dict'
    assert len(var_10.tests) == 39
    assert f'{type(var_10.globals).__module__}.{type(var_10.globals).__qualname__}' == 'builtins.dict'
    assert len(var_10.globals) == 6
    assert f'{type(var_10.loader).__module__}.{type(var_10.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_10.cache).__module__}.{type(var_10.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_10.cache) == 0
    assert var_10.bytecode_cache is None
    assert var_10.auto_reload is True
    assert var_10.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_10.extensions == {}
    assert var_10.is_async is False
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
    var_11 = 'username'
    var_12 = 'password'
    var_13 = 'description'
    var_14 = 'role'
    var_15 = 'is_active'
    var_16 = 'Username'
    var_17 = module_1.String()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.String'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.allow_blank is False
    assert var_17.trim_whitespace is True
    assert var_17.max_length is None
    assert var_17.min_length is None
    assert var_17.format is None
    assert var_17.coerce_types is True
    assert var_17.pattern is None
    assert var_17.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_18 = 'Password'
    var_19 = module_1.String(format=var_12)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.String'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.allow_blank is False
    assert var_19.trim_whitespace is True
    assert var_19.max_length is None
    assert var_19.min_length is None
    assert var_19.format == 'password'
    assert var_19.coerce_types is True
    assert var_19.pattern is None
    assert var_19.pattern_regex is None
    var_20 = 'Description'
    var_21 = 'text'
    var_22 = module_1.String(format=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.String'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.allow_blank is False
    assert var_22.trim_whitespace is True
    assert var_22.max_length is None
    assert var_22.min_length is None
    assert var_22.format == 'text'
    assert var_22.coerce_types is True
    assert var_22.pattern is None
    assert var_22.pattern_regex is None
    var_23 = 'Role'
    var_24 = 'admin'
    var_25 = 'Admin'
    var_26 = (var_24, var_25)
    var_27 = 'user'
    var_28 = 'User'
    var_29 = (var_27, var_28)
    var_30 = [var_26, var_29]
    var_31 = module_1.Choice(choices=var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Choice'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.choices == [('admin', 'Admin'), ('user', 'User')]
    assert var_31.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_32 = 'Is Active'
    var_33 = module_1.Boolean()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_34 = {var_11: var_17, var_12: var_19, var_13: var_22, var_14: var_31, var_15: var_33}
    var_35 = module_2.Schema(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.fields).__module__}.{type(var_35.fields).__qualname__}' == 'builtins.dict'
    assert len(var_35.fields) == 5
    assert var_35.required == ['username', 'password', 'description', 'role', 'is_active']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_36 = module_0.Form(env=var_10, schema=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_36.env).__module__}.{type(var_36.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_36.schema).__module__}.{type(var_36.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_36.values is None
    assert var_36.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_37 = var_35.fields[var_11]
    var_38 = var_36.render_field(field_name=var_11, field=var_37)
    assert var_38 == '\n        <div>\n            <label for="username">Username</label>\n            <input type="text" id="username" name="username" value="" required>\n            <p></p>\n        </div>\n    '
    assert var_38 == '\n        <div>\n            <label for="username">username</label>\n            <input type="text" id="username" name="username" value="None" required>\n            \n        </div>\n        '
    assert len(var_10.cache) == 1
    var_39 = var_35.fields[var_12]
    var_40 = var_36.render_field(field_name=var_12, field=var_39)
    assert var_40 == '\n        <div>\n            <label for="password">Password</label>\n            <input type="password" id="password" name="password" value="" required>\n            <p></p>\n        </div>\n    '
    assert var_40 == '\n        <div>\n            <label for="password">password</label>\n            <input type="password" id="password" name="password" value="" required>\n            \n        </div>\n        '
    var_41 = var_35.fields[var_13]
    var_42 = var_36.render_field(field_name=var_13, field=var_41)
    assert var_42 == '\n        <div>\n            <label for="description">Description</label>\n            <textarea id="description" name="description" required></textarea>\n            <p></p>\n        </div>\n    '
    assert var_42 == '\n        <div>\n            <label for="description">description</label>\n            <textarea id="description" name="description" required>None</textarea>\n            \n        </div>\n        '
    assert len(var_10.cache) == 2
    var_36.render_field(field_name=var_14, field=var_11)
    assert var_43 == '\n        <div>\n            <label for="role">Role</label>\n            <select id="role" name="role" required>\n                <option value="admin">Admin</option>\n                <option value="user">User</option>\n            </select>\n            <p></p>\n        </div>\n    '

def test_case_18():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_5 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <textarea id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>{{ value }}</textarea>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_6 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <select id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>\n                {% for choice in field.choices %}\n                    <option value="{{ choice.value }}" {% if choice.value == value %}selected{% endif %}>{{ choice.text }}</option>\n                {% endfor %}\n            </select>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_7 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" value="true" {% if value %}checked{% endif %} {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_5.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'forms/input.html': '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        ', 'forms/textarea.html': '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <textarea id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>{{ value }}</textarea>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        ', 'forms/select.html': '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <select id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>\n                {% for choice in field.choices %}\n                    <option value="{{ choice.value }}" {% if choice.value == value %}selected{% endif %}>{{ choice.text }}</option>\n                {% endfor %}\n            </select>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        ', 'forms/checkbox.html': '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" value="true" {% if value %}checked{% endif %} {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '}
    var_10 = module_3.Environment(loader=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'jinja2.environment.Environment'
    assert var_10.block_start_string == '{%'
    assert var_10.block_end_string == '%}'
    assert var_10.variable_start_string == '{{'
    assert var_10.variable_end_string == '}}'
    assert var_10.comment_start_string == '{#'
    assert var_10.comment_end_string == '#}'
    assert var_10.line_statement_prefix is None
    assert var_10.line_comment_prefix is None
    assert var_10.trim_blocks is False
    assert var_10.lstrip_blocks is False
    assert var_10.newline_sequence == '\n'
    assert var_10.keep_trailing_newline is False
    assert var_10.optimized is True
    assert var_10.finalize is None
    assert var_10.autoescape is False
    assert f'{type(var_10.filters).__module__}.{type(var_10.filters).__qualname__}' == 'builtins.dict'
    assert len(var_10.filters) == 54
    assert f'{type(var_10.tests).__module__}.{type(var_10.tests).__qualname__}' == 'builtins.dict'
    assert len(var_10.tests) == 39
    assert f'{type(var_10.globals).__module__}.{type(var_10.globals).__qualname__}' == 'builtins.dict'
    assert len(var_10.globals) == 6
    assert f'{type(var_10.loader).__module__}.{type(var_10.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_10.cache).__module__}.{type(var_10.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_10.cache) == 0
    assert var_10.bytecode_cache is None
    assert var_10.auto_reload is True
    assert var_10.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_10.extensions == {}
    assert var_10.is_async is False
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
    var_11 = 'username'
    var_12 = 'password'
    var_13 = 'description'
    var_14 = 'role'
    var_15 = 'is_active'
    var_16 = 'Username'
    var_17 = module_1.String()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.String'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.allow_blank is False
    assert var_17.trim_whitespace is True
    assert var_17.max_length is None
    assert var_17.min_length is None
    assert var_17.format is None
    assert var_17.coerce_types is True
    assert var_17.pattern is None
    assert var_17.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_18 = 'Password'
    var_19 = module_1.String(format=var_12)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.String'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.allow_blank is False
    assert var_19.trim_whitespace is True
    assert var_19.max_length is None
    assert var_19.min_length is None
    assert var_19.format == 'password'
    assert var_19.coerce_types is True
    assert var_19.pattern is None
    assert var_19.pattern_regex is None
    var_20 = 'Description'
    var_21 = 'text'
    var_22 = module_1.String(format=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.String'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.allow_blank is False
    assert var_22.trim_whitespace is True
    assert var_22.max_length is None
    assert var_22.min_length is None
    assert var_22.format == 'text'
    assert var_22.coerce_types is True
    assert var_22.pattern is None
    assert var_22.pattern_regex is None
    var_23 = 'Role'
    var_24 = 'admin'
    var_25 = 'Admin'
    var_26 = (var_24, var_25)
    var_27 = 'user'
    var_28 = 'User'
    var_29 = (var_27, var_28)
    var_30 = [var_26, var_29]
    var_31 = module_1.Choice(choices=var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Choice'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.choices == [('admin', 'Admin'), ('user', 'User')]
    assert var_31.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_32 = 'Is Active'
    var_33 = module_1.Boolean()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_34 = {var_11: var_17, var_12: var_19, var_13: var_22, var_14: var_31, var_15: var_33}
    var_35 = module_2.Schema(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.fields).__module__}.{type(var_35.fields).__qualname__}' == 'builtins.dict'
    assert len(var_35.fields) == 5
    assert var_35.required == ['username', 'password', 'description', 'role', 'is_active']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_36 = module_0.Form(env=var_10, schema=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_36.env).__module__}.{type(var_36.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_36.schema).__module__}.{type(var_36.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_36.values is None
    assert var_36.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_37 = var_35.fields[var_11]
    var_38 = var_36.render_field(field_name=var_11, field=var_37)
    assert var_38 == '\n        <div>\n            <label for="username">username</label>\n            <input type="text" id="username" name="username" value="None" required>\n            \n        </div>\n        '
    assert len(var_10.cache) == 1
    var_39 = var_35.fields[var_12]
    var_40 = var_36.render_field(field_name=var_12, field=var_39)
    assert var_40 == '\n        <div>\n            <label for="password">password</label>\n            <input type="password" id="password" name="password" value="" required>\n            \n        </div>\n        '
    var_41 = var_35.fields[var_13]
    var_42 = var_36.render_field(field_name=var_13, field=var_41)
    assert var_42 == '\n        <div>\n            <label for="description">description</label>\n            <textarea id="description" name="description" required>None</textarea>\n            \n        </div>\n        '
    assert len(var_10.cache) == 2
    var_43 = var_35.fields[var_14]
    var_44 = var_36.render_field(field_name=var_14, field=var_43)
    assert var_44 == '\n        <div>\n            <label for="role">role</label>\n            <select id="role" name="role" required>\n                \n                    <option value="" ></option>\n                \n                    <option value="" ></option>\n                \n            </select>\n            \n        </div>\n        '
    assert len(var_10.cache) == 3
    var_45 = var_35.fields[var_15]
    var_46 = var_36.render_field(field_name=var_15, field=var_45)
    assert var_46 == '\n        <div>\n            <label for="is-active">is_active</label>\n            <input type="checkbox" id="is-active" name="is_active" value="true"  required>\n            \n        </div>\n        '
    assert len(var_10.cache) == 4