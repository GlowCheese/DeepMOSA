# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import jinja2.loaders as module_3
import jinja2.environment as module_4
import jinja2.exceptions as module_5
import jinja2.nodes as module_6
import markupsafe as module_7

def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

def test_case_1():
    var_0 = ''
    var_1 = module_1.Boolean()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_2 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_1}
    var_3 = module_2.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ['']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Form(env=var_2, schema=var_3, values=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'builtins.dict'
    assert len(var_4.env) == 1
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_4.values).__module__}.{type(var_4.values).__qualname__}' == 'builtins.dict'
    assert len(var_4.values) == 1
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    with pytest.raises(AttributeError):
        var_5 = str(var_4)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'N'
    module_0.Jinja2Forms(directory=var_0, package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'cN'
    module_0.Jinja2Forms(package=var_0)

def test_case_4():
    var_0 = 'forms/input.html'
    var_1 = module_3.DictLoader(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_1.mapping == 'forms/input.html'
    var_2 = module_4.Environment(loader=var_1)
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
    assert f'{type(var_2.loader).__module__}.{type(var_2.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.extensions == {}
    assert var_2.is_async is False
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
    var_3 = 'qe@up\rfuiR5j_@cfjX'
    var_4 = 'activKoe'
    var_5 = 'text'
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
    assert var_6.format == 'text'
    assert var_6.coerce_types is True
    assert var_6.pattern is None
    assert var_6.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = module_1.Boolean()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_8 = module_0.Jinja2Forms(directory=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    var_9 = {var_0: var_6, var_3: var_6, var_3: var_6, var_4: var_7}
    var_10 = module_2.Schema(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.fields).__module__}.{type(var_10.fields).__qualname__}' == 'builtins.dict'
    assert len(var_10.fields) == 3
    assert var_10.required == ['forms/input.html', 'qe@up\rfuiR5j_@cfjX', 'activKoe']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_11 = '0z #nj@'
    var_12 = True
    var_13 = {var_3: var_11, var_3: var_4, var_4: var_3, var_4: var_12}
    var_14 = module_0.Form(env=var_2, schema=var_10, values=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_14.env).__module__}.{type(var_14.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_14.schema).__module__}.{type(var_14.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_14.values == {'qe@up\rfuiR5j_@cfjX': 'activKoe', 'activKoe': True}
    assert var_14.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    with pytest.raises(module_5.TemplateNotFound):
        var_15 = str(var_14)

def test_case_5():
    var_0 = 'N'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        var_1.load_template_env()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'N'
    var_1 = None
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_2.env).__module__}.{type(var_2.env).__qualname__}' == 'jinja2.environment.Environment'
    var_3 = None
    var_2.create_form(var_3)

def test_case_7():
    var_0 = 'name'
    var_1 = 'age'
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
    var_3 = 'number'
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
    assert var_4.format == 'number'
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_2.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 2
    assert var_6.required == ['name', 'age']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_3.BaseLoader()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert module_3.BaseLoader.has_source_access is True
    var_8 = module_4.Environment(loader=var_7)
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
    assert var_8.autoescape is False
    assert f'{type(var_8.filters).__module__}.{type(var_8.filters).__qualname__}' == 'builtins.dict'
    assert len(var_8.filters) == 54
    assert f'{type(var_8.tests).__module__}.{type(var_8.tests).__qualname__}' == 'builtins.dict'
    assert len(var_8.tests) == 39
    assert f'{type(var_8.globals).__module__}.{type(var_8.globals).__qualname__}' == 'builtins.dict'
    assert len(var_8.globals) == 6
    assert f'{type(var_8.loader).__module__}.{type(var_8.loader).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert f'{type(var_8.cache).__module__}.{type(var_8.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_8.cache) == 0
    assert var_8.bytecode_cache is None
    assert var_8.auto_reload is True
    assert var_8.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_8.extensions == {}
    assert var_8.is_async is False
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
    var_9 = 'John'
    var_10 = '25'
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_0.Form(env=var_8, schema=var_6, values=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_12.env).__module__}.{type(var_12.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_12.schema).__module__}.{type(var_12.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_12.values == {'name': 'John', 'age': '25'}
    assert var_12.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_13 = 'Jane'
    var_14 = '30'
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = var_12.validate(var_15)
    assert var_12.values == {'name': 'Jane', 'age': '30'}
    assert var_12.data == {'name': 'Jane', 'age': '30'}
    var_17 = {var_0: var_9, var_1: var_10}
    var_18 = module_0.Form(env=var_8, schema=var_6, values=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_18.env).__module__}.{type(var_18.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_18.schema).__module__}.{type(var_18.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_18.values == {'name': 'John', 'age': '25'}
    assert var_18.errors is None
    var_19 = ''
    var_20 = {var_0: var_19}
    var_21 = var_18.validate(var_20)
    assert var_18.values is None
    assert f'{type(var_18.errors).__module__}.{type(var_18.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_18.errors) == 2
    assert var_18.data == {'name': ''}
    var_22 = {var_0: var_9, var_1: var_10}
    var_23 = module_0.Form(env=var_8, schema=var_6, values=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_23.env).__module__}.{type(var_23.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_23.schema).__module__}.{type(var_23.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_23.values == {'name': 'John', 'age': '25'}
    assert var_23.errors is None
    var_24 = {var_0: var_13, var_1: var_14}
    var_25 = var_23.validate(var_24)
    assert var_23.values == {'name': 'Jane', 'age': '30'}
    assert var_23.data == {'name': 'Jane', 'age': '30'}
    var_26 = 'name'
    var_27 = 'age'
    var_28 = 'Jane'
    var_29 = '30'
    var_30 = {var_26: var_28, var_27: var_29}
    with pytest.raises(AssertionError):
        var_23.validate(var_30)

def test_case_8():
    var_0 = module_4.Environment()
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
    var_1 = {}
    var_2 = module_2.Schema(var_1)
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = var_3.input_type_for_field(var_4)
    assert var_5 == 'text'
    var_6 = var_3.input_type_for_field(var_4)
    assert var_6 == 'text'
    var_7 = 'unsupported'
    var_8 = module_1.String(format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format == 'unsupported'
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    var_9 = var_3.input_type_for_field(var_8)
    assert var_9 == 'text'
    var_10 = 'color'
    var_11 = module_1.Boolean()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_12 = None
    var_13 = var_3.validate(var_12)
    assert f'{type(var_3.errors).__module__}.{type(var_3.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.errors) == 1
    assert var_3.data is None

def test_case_9():
    var_0 = module_4.Environment()
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
    var_1 = 'S"EYxW'
    var_2 = var_0.parse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'jinja2.nodes.Template'
    assert f'{type(var_2.body).__module__}.{type(var_2.body).__qualname__}' == 'builtins.list'
    assert len(var_2.body) == 1
    assert var_2.lineno == 1
    assert f'{type(var_2.environment).__module__}.{type(var_2.environment).__qualname__}' == 'jinja2.environment.Environment'
    assert module_6.Template.fields == ('body',)
    assert module_6.Template.attributes == ('lineno', 'environment')
    assert module_6.Template.abstract is False
    var_3 = {}
    var_4 = module_2.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.fields == {}
    assert var_4.required == []
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
    var_7 = var_5.input_type_for_field(var_6)
    assert var_7 == 'text'
    var_8 = var_5.input_type_for_field(var_6)
    assert var_8 == 'text'
    var_9 = 'unsupported'
    var_10 = module_1.String(format=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length is None
    assert var_10.min_length is None
    assert var_10.format == 'unsupported'
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = var_5.input_type_for_field(var_10)
    assert var_11 == 'text'
    var_12 = 'color'
    var_13 = module_1.Boolean()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_14 = var_5.render_fields()
    assert var_14 == ''
    var_15 = var_5.input_type_for_field(var_13)
    assert var_15 == 'text'

def test_case_10():
    var_0 = 'forms/input.html'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_3.DictLoader(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_2.mapping == {'forms/input.html': 'forms/input.html'}
    var_3 = module_4.Environment(loader=var_2)
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
    assert f'{type(var_3.loader).__module__}.{type(var_3.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
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
    var_4 = 'number'
    var_5 = module_1.String(format=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length is None
    assert var_5.min_length is None
    assert var_5.format == 'number'
    assert var_5.coerce_types is True
    assert var_5.pattern is None
    assert var_5.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = module_1.Boolean()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_7 = {var_0: var_5, var_4: var_5, var_4: var_5, var_4: var_6}
    var_8 = module_2.Schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 2
    assert var_8.required == ['forms/input.html', 'number']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_9 = module_0.Form(env=var_3, schema=var_8, values=var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_9.schema).__module__}.{type(var_9.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_9.values).__module__}.{type(var_9.values).__qualname__}' == 'builtins.dict'
    assert len(var_9.values) == 2
    assert var_9.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    with pytest.raises(module_5.TemplateNotFound):
        var_10 = str(var_9)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'forms/selycl.htmEl'
    var_1 = module_3.DictLoader(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_1.mapping == 'forms/selycl.htmEl'
    var_2 = module_4.Environment(loader=var_1)
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
    assert f'{type(var_2.loader).__module__}.{type(var_2.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.extensions == {}
    assert var_2.is_async is False
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
    var_3 = 'name'
    var_4 = 'EzZ\r\rH_DH'
    var_5 = 'number'
    var_6 = 'text'
    var_7 = module_1.String(format=var_6)
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_1.Boolean()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_9 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    var_10 = {var_3: var_7, var_8: var_7, var_8: var_7, var_4: var_8}
    var_11 = module_2.Schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.fields).__module__}.{type(var_11.fields).__qualname__}' == 'builtins.dict'
    assert len(var_11.fields) == 3
    assert f'{type(var_11.required).__module__}.{type(var_11.required).__qualname__}' == 'builtins.list'
    assert len(var_11.required) == 3
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_12 = '00 #{j@'
    var_13 = True
    var_14 = {var_3: var_12, var_5: var_5, var_4: var_3, var_4: var_13}
    var_15 = module_0.Form(env=var_2, schema=var_11, values=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_15.env).__module__}.{type(var_15.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_15.schema).__module__}.{type(var_15.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_15.values == {'name': '00 #{j@', 'EzZ\r\rH_DH': True}
    assert var_15.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_15.__html__()

def test_case_12():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'bio'
    var_3 = 'country'
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = 'number'
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
    assert var_6.format == 'number'
    assert var_6.coerce_types is True
    assert var_6.pattern is None
    assert var_6.pattern_regex is None
    var_7 = 'text'
    var_8 = module_1.String(format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format == 'text'
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    var_9 = module_1.Boolean()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_10 = 'US'
    var_11 = 'UK'
    var_12 = 'CA'
    var_13 = [var_10, var_11, var_12]
    var_14 = module_1.Choice(choices=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Choice'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.choices == [('US', 'US'), ('UK', 'UK'), ('CA', 'CA')]
    assert var_14.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_15 = {var_0: var_4, var_1: var_6, var_2: var_8, var_12: var_9, var_3: var_14}
    var_16 = module_2.Schema(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert f'{type(var_16.fields).__module__}.{type(var_16.fields).__qualname__}' == 'builtins.dict'
    assert len(var_16.fields) == 5
    assert var_16.required == ['name', 'age', 'bio', 'CA', 'country']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_17 = 'forms/input.html'
    var_18 = 'forms/textarea.html'
    var_19 = 'forms/select.html'
    var_20 = 'forms/checkbox.html'
    var_21 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_22 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_23 = '<select name="{{ field_name }}"></select>'
    var_24 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_18}
    var_25 = module_3.DictLoader(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_25.mapping == {'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">', 'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>', 'forms/select.html': '<select name="{{ field_name }}"></select>', 'forms/checkbox.html': 'forms/textarea.html'}
    var_26 = True
    var_27 = module_4.Environment(autoescape=var_26, loader=var_25)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'jinja2.environment.Environment'
    assert var_27.block_start_string == '{%'
    assert var_27.block_end_string == '%}'
    assert var_27.variable_start_string == '{{'
    assert var_27.variable_end_string == '}}'
    assert var_27.comment_start_string == '{#'
    assert var_27.comment_end_string == '#}'
    assert var_27.line_statement_prefix is None
    assert var_27.line_comment_prefix is None
    assert var_27.trim_blocks is False
    assert var_27.lstrip_blocks is False
    assert var_27.newline_sequence == '\n'
    assert var_27.keep_trailing_newline is False
    assert var_27.optimized is True
    assert var_27.finalize is None
    assert var_27.autoescape is True
    assert f'{type(var_27.filters).__module__}.{type(var_27.filters).__qualname__}' == 'builtins.dict'
    assert len(var_27.filters) == 54
    assert f'{type(var_27.tests).__module__}.{type(var_27.tests).__qualname__}' == 'builtins.dict'
    assert len(var_27.tests) == 39
    assert f'{type(var_27.globals).__module__}.{type(var_27.globals).__qualname__}' == 'builtins.dict'
    assert len(var_27.globals) == 6
    assert f'{type(var_27.loader).__module__}.{type(var_27.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_27.cache).__module__}.{type(var_27.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_27.cache) == 0
    assert var_27.bytecode_cache is None
    assert var_27.auto_reload is True
    assert var_27.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_27.extensions == {}
    assert var_27.is_async is False
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
    var_28 = 'John'
    var_29 = '30'
    var_30 = 'Developer'
    var_31 = {var_0: var_28, var_1: var_29, var_2: var_30, var_5: var_26, var_3: var_10}
    var_32 = module_0.Form(env=var_27, schema=var_16, values=var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_32.env).__module__}.{type(var_32.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_32.schema).__module__}.{type(var_32.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_32.values == {'name': 'John', 'age': '30', 'bio': 'Developer', 'country': 'US'}
    assert var_32.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_33 = var_32.__html__()
    assert len(var_27.cache) == 4
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'markupsafe.Markup'
    assert len(var_33) == 179
    assert f'{type(module_7.Markup.escape).__module__}.{type(module_7.Markup.escape).__qualname__}' == 'builtins.method'
    var_34 = str(var_33)
    var_35 = var_32.render_fields()
    assert var_35 == '<input type="text" name="name" value="John"><input type="number" name="age" value="30"><textarea name="bio">Developer</textarea>forms/textarea.html<select name="country"></select>'
    assert f'{type(module_7.annotations).__module__}.{type(module_7.annotations).__qualname__}' == '__future__._Feature'
    assert module_7.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_7.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_7.annotations.compiler_flag == 16777216

def test_case_13():
    var_0 = 'forms/select.html'
    var_1 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_2 = module_3.DictLoader(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_2.mapping == '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_3 = module_4.Environment(loader=var_2)
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
    assert f'{type(var_3.loader).__module__}.{type(var_3.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
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
    var_4 = 'name'
    var_5 = 'number'
    var_6 = 'text'
    var_7 = module_1.String(format=var_6)
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_1.Boolean()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_9 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    var_10 = {var_4: var_7, var_1: var_7, var_1: var_7, var_4: var_8}
    var_11 = module_2.Schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.fields).__module__}.{type(var_11.fields).__qualname__}' == 'builtins.dict'
    assert len(var_11.fields) == 2
    assert var_11.required == ['name', '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_12 = '0z #nj@'
    var_13 = 'Test description'
    var_14 = True
    var_15 = {var_4: var_12, var_6: var_5, var_0: var_13, var_0: var_14}
    var_16 = module_0.Form(env=var_3, schema=var_11, values=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_16.env).__module__}.{type(var_16.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_16.schema).__module__}.{type(var_16.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_16.values == {'name': '0z #nj@'}
    assert var_16.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_17 = var_16.template_for_field(var_14)
    assert var_17 == 'forms/input.html'
    var_18 = var_16.validate()
    assert var_16.values is None
    assert f'{type(var_16.errors).__module__}.{type(var_16.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_16.errors) == 1
    assert var_16.data is None
    with pytest.raises(module_5.TemplateNotFound):
        var_19 = str(var_16)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'forms/input.html'
    var_1 = 'forms/select.html'
    var_2 = module_3.DictLoader(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_2.mapping == 'forms/input.html'
    var_3 = 'qe@up\rfuiR5j_@cfjX'
    var_4 = 'text'
    var_5 = module_1.String(format=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length is None
    assert var_5.min_length is None
    assert var_5.format == 'text'
    assert var_5.coerce_types is True
    assert var_5.pattern is None
    assert var_5.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = module_1.Boolean()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_7 = module_0.Jinja2Forms(directory=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    var_8 = {var_3: var_5, var_3: var_5, var_3: var_5, var_0: var_6}
    var_9 = module_2.Schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.fields).__module__}.{type(var_9.fields).__qualname__}' == 'builtins.dict'
    assert len(var_9.fields) == 2
    assert var_9.required == ['qe@up\rfuiR5j_@cfjX', 'forms/input.html']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_10 = None
    var_11 = module_0.Form(env=var_10, schema=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.forms.Form'
    assert var_11.env is None
    assert f'{type(var_11.schema).__module__}.{type(var_11.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.values is None
    assert var_11.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_12 = None
    var_13 = var_6.validate_or_error(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert f'{type(var_13.error).__module__}.{type(var_13.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13.error) == 1
    var_11.__str__()

def test_case_15():
    var_0 = 'forms/input.html'
    var_1 = 'forms/select.html'
    var_2 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_3 = module_3.DictLoader(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_3.mapping == 'forms/input.html'
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
    var_6 = 'number'
    var_7 = 'text'
    var_8 = module_1.String(format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format == 'text'
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_1.Boolean()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_10 = module_0.Jinja2Forms(directory=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_10.env).__module__}.{type(var_10.env).__qualname__}' == 'jinja2.environment.Environment'
    var_11 = {var_5: var_8, var_2: var_8, var_2: var_8, var_2: var_9}
    var_12 = module_2.Schema(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.fields).__module__}.{type(var_12.fields).__qualname__}' == 'builtins.dict'
    assert len(var_12.fields) == 2
    assert var_12.required == ['name', '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_13 = '0z #nj@'
    var_14 = 'Test description'
    var_15 = True
    var_16 = {var_5: var_13, var_0: var_6, var_14: var_14, var_14: var_15}
    var_17 = module_0.Form(env=var_4, schema=var_12, values=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_17.env).__module__}.{type(var_17.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_17.schema).__module__}.{type(var_17.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_17.values == {'name': '0z #nj@'}
    assert var_17.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_18 = var_17.template_for_field(var_15)
    assert var_18 == 'forms/input.html'
    with pytest.raises(module_5.TemplateNotFound):
        var_19 = str(var_17)

def test_case_16():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_5 = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    var_6 = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    var_7 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_3.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'forms/input.html': "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>", 'forms/textarea.html': "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>", 'forms/select.html': "<select id='{{ field_id }}' name='{{ field_name }}'></select>", 'forms/checkbox.html': "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"}
    var_10 = module_4.Environment(loader=var_9)
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
    var_11 = 'text_field'
    var_12 = 'email_field'
    var_13 = 'number_field'
    var_14 = 'password_field'
    var_15 = 'choice_field'
    var_16 = 'boolean_field'
    var_17 = 'textarea_field'
    var_18 = 'text'
    var_19 = module_1.String(format=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.String'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.allow_blank is False
    assert var_19.trim_whitespace is True
    assert var_19.max_length is None
    assert var_19.min_length is None
    assert var_19.format == 'text'
    assert var_19.coerce_types is True
    assert var_19.pattern is None
    assert var_19.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_20 = 'email'
    var_21 = module_1.String(format=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.String'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.allow_blank is False
    assert var_21.trim_whitespace is True
    assert var_21.max_length is None
    assert var_21.min_length is None
    assert var_21.format == 'email'
    assert var_21.coerce_types is True
    assert var_21.pattern is None
    assert var_21.pattern_regex is None
    var_22 = 'number'
    var_23 = module_1.String(format=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.String'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.allow_blank is False
    assert var_23.trim_whitespace is True
    assert var_23.max_length is None
    assert var_23.min_length is None
    assert var_23.format == 'number'
    assert var_23.coerce_types is True
    assert var_23.pattern is None
    assert var_23.pattern_regex is None
    var_24 = 'password'
    var_25 = module_1.String(format=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.String'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.allow_blank is False
    assert var_25.trim_whitespace is True
    assert var_25.max_length is None
    assert var_25.min_length is None
    assert var_25.format == 'password'
    assert var_25.coerce_types is True
    assert var_25.pattern is None
    assert var_25.pattern_regex is None
    var_26 = 'a'
    var_27 = 'b'
    var_28 = 'c'
    var_29 = [var_26, var_27, var_28]
    var_30 = module_1.Choice(choices=var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Choice'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_30.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_31 = module_1.Boolean()
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_32 = module_1.String(format=var_18)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.String'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.allow_blank is False
    assert var_32.trim_whitespace is True
    assert var_32.max_length is None
    assert var_32.min_length is None
    assert var_32.format == 'text'
    assert var_32.coerce_types is True
    assert var_32.pattern is None
    assert var_32.pattern_regex is None
    var_33 = {var_11: var_19, var_12: var_21, var_13: var_23, var_14: var_25, var_15: var_30, var_16: var_31, var_17: var_32}
    var_34 = module_2.Schema(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.fields).__module__}.{type(var_34.fields).__qualname__}' == 'builtins.dict'
    assert len(var_34.fields) == 7
    assert var_34.required == ['text_field', 'email_field', 'number_field', 'password_field', 'choice_field', 'boolean_field', 'textarea_field']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_35 = module_0.Form(env=var_10, schema=var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_35.env).__module__}.{type(var_35.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_35.schema).__module__}.{type(var_35.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_35.values is None
    assert var_35.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_36 = var_34.fields[var_11]
    var_37 = 'test'
    var_38 = var_35.render_field(field_name=var_11, field=var_36, value=var_37)
    assert var_38 == "<textarea id='text-field' name='text_field'>test</textarea>"
    assert len(var_10.cache) == 1
    var_39 = var_34.fields[var_12]
    var_40 = var_35.render_field(field_name=var_12, field=var_39, value=var_26)
    assert var_40 == "<input id='email-field' name='email_field' type='email' value='a'>"
    assert len(var_10.cache) == 2
    var_41 = var_34.fields[var_13]
    var_42 = '123'
    var_43 = var_35.render_field(field_name=var_13, field=var_41, value=var_42)
    assert var_43 == "<input id='number-field' name='number_field' type='number' value='123'>"
    var_44 = var_34.fields[var_14]
    var_45 = 'secret'
    var_46 = var_35.render_field(field_name=var_14, field=var_44, value=var_45)
    assert var_46 == "<input id='password-field' name='password_field' type='password' value=''>"
    var_47 = var_34.fields[var_15]
    var_48 = var_35.render_field(field_name=var_15, field=var_47, value=var_26)
    assert var_48 == "<select id='choice-field' name='choice_field'></select>"
    assert len(var_10.cache) == 3
    var_49 = var_34.fields[var_16]
    var_50 = True
    var_51 = var_35.render_field(field_name=var_16, field=var_49, value=var_50)
    assert var_51 == "<input id='boolean-field' name='boolean_field' type='checkbox' checked>"
    assert len(var_10.cache) == 4
    var_52 = var_34.fields[var_17]
    var_53 = 'long text'
    var_54 = var_35.render_field(field_name=var_17, field=var_52, value=var_53)
    assert var_54 == "<textarea id='textarea-field' name='textarea_field'>long text</textarea>"
    var_55 = var_34.fields[var_11]
    var_56 = 'Invalid'
    var_57 = var_35.render_field(field_name=var_11, field=var_55, value=var_37, error=var_56)
    assert var_57 == "<textarea id='text-field' name='text_field'>test</textarea>"

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_4.Environment()
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
    var_1 = {}
    var_2 = module_2.Schema(var_1)
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = var_3.input_type_for_field(var_4)
    assert var_5 == 'text'
    var_6 = var_3.input_type_for_field(var_4)
    assert var_6 == 'text'
    var_7 = 'unsupported'
    var_3.render_field(field_name=var_5, field=var_7)

def test_case_18():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_1.Choice(choices=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Choice'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.choices == [('a', 'A'), ('b', 'B')]
    assert var_7.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_8 = None
    var_9 = 'test'
    var_10 = {var_9: var_7}
    var_11 = module_2.Schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.fields).__module__}.{type(var_11.fields).__qualname__}' == 'builtins.dict'
    assert len(var_11.fields) == 1
    assert var_11.required == ['test']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_12 = module_1.Boolean()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_13 = {var_9: var_12}
    var_14 = module_2.Schema(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.fields).__module__}.{type(var_14.fields).__qualname__}' == 'builtins.dict'
    assert len(var_14.fields) == 1
    assert var_14.required == ['test']
    var_15 = module_0.Form(env=var_8, schema=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.forms.Form'
    assert var_15.env is None
    assert f'{type(var_15.schema).__module__}.{type(var_15.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_15.values is None
    assert var_15.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_16 = var_15.template_for_field(var_12)
    assert var_16 == 'forms/checkbox.html'
    var_17 = 'text'
    var_18 = module_1.String(format=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.String'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.allow_blank is False
    assert var_18.trim_whitespace is True
    assert var_18.max_length is None
    assert var_18.min_length is None
    assert var_18.format == 'text'
    assert var_18.coerce_types is True
    assert var_18.pattern is None
    assert var_18.pattern_regex is None
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_19 = {var_9: var_18}
    var_20 = module_2.Schema(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.fields).__module__}.{type(var_20.fields).__qualname__}' == 'builtins.dict'
    assert len(var_20.fields) == 1
    assert var_20.required == ['test']
    var_21 = var_15.template_for_field(var_18)
    assert var_21 == 'forms/textarea.html'
    var_22 = module_1.String()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.String'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.allow_blank is False
    assert var_22.trim_whitespace is True
    assert var_22.max_length is None
    assert var_22.min_length is None
    assert var_22.format is None
    assert var_22.coerce_types is True
    assert var_22.pattern is None
    assert var_22.pattern_regex is None
    var_23 = {var_9: var_22}
    var_24 = module_2.Schema(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.fields).__module__}.{type(var_24.fields).__qualname__}' == 'builtins.dict'
    assert len(var_24.fields) == 1
    assert var_24.required == ['test']
    var_25 = module_0.Form(env=var_8, schema=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.forms.Form'
    assert var_25.env is None
    assert f'{type(var_25.schema).__module__}.{type(var_25.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_25.values is None
    assert var_25.errors is None
    var_26 = var_25.template_for_field(var_22)
    assert var_26 == 'forms/input.html'
    var_27 = module_1.Field()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Field'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert module_1.Field.errors == {}
    var_28 = module_0.Form(env=var_8, schema=var_11)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.forms.Form'
    assert var_28.env is None
    assert f'{type(var_28.schema).__module__}.{type(var_28.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_28.values is None
    assert var_28.errors is None
    var_29 = var_28.template_for_field(var_27)
    assert var_29 == 'forms/input.html'
    var_30 = {}
    var_31 = module_1.Object()
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Object'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.properties == {}
    assert var_31.pattern_properties == {}
    assert var_31.additional_properties is True
    assert var_31.property_names is None
    assert var_31.min_properties is None
    assert var_31.max_properties is None
    assert var_31.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_32 = {var_9: var_31}
    var_33 = module_2.Schema(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.fields).__module__}.{type(var_33.fields).__qualname__}' == 'builtins.dict'
    assert len(var_33.fields) == 1
    assert var_33.required == ['test']
    var_34 = module_0.Form(env=var_8, schema=var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.forms.Form'
    assert var_34.env is None
    assert f'{type(var_34.schema).__module__}.{type(var_34.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_34.values is None
    assert var_34.errors is None
    with pytest.raises(AssertionError):
        var_34.template_for_field(var_31)

def test_case_19():
    var_0 = 'forms/input.html'
    var_1 = 'forms/select.html'
    var_2 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_3 = module_3.DictLoader(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_3.mapping == 'forms/input.html'
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
    var_6 = 'activKoe'
    var_7 = module_1.String(allow_blank=var_3, format=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.default == ''
    assert f'{type(var_7.allow_blank).__module__}.{type(var_7.allow_blank).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format == 'name'
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_1.Boolean()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_9 = module_0.Jinja2Forms(directory=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    var_10 = {var_5: var_7, var_2: var_7, var_2: var_7, var_6: var_8}
    var_11 = module_2.Schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.fields).__module__}.{type(var_11.fields).__qualname__}' == 'builtins.dict'
    assert len(var_11.fields) == 3
    assert var_11.required == ['activKoe']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_12 = '0z #nj@'
    var_13 = 'Test description'
    var_14 = False
    var_15 = {var_5: var_12, var_0: var_0, var_6: var_13, var_6: var_14}
    var_16 = module_0.Form(env=var_4, schema=var_11, values=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_16.env).__module__}.{type(var_16.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_16.schema).__module__}.{type(var_16.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_16.values == {'name': '0z #nj@', 'activKoe': False}
    assert var_16.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_17 = var_16.template_for_field(var_14)
    assert var_17 == 'forms/input.html'
    with pytest.raises(TypeError):
        var_18 = str(var_16)