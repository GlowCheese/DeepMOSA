# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import typesystem.fields as module_3

def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

@pytest.mark.xfail(strict=True)
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
    var_4 = module_3.String(**var_1)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = var_3.__str__()
    assert var_5 == ''
    var_3.render_field(field_name=var_5, field=var_4)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '/some/path'
    module_0.Jinja2Forms(package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '/some/pathK('
    module_0.Jinja2Forms(directory=var_0, package=var_0)

def test_case_4():
    var_0 = '/.ome/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AttributeError):
        var_2 = var_0.env.autoescape
    assert var_2 is True

def test_case_5():
    var_0 = 'vPb(eyK&'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        var_1.load_template_env()

@pytest.mark.xfail(strict=True)
def test_case_6():
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
    var_1 = 'phone'
    var_2 = 'tn\x0b'
    var_3 = {}
    var_4 = None
    var_5 = module_0.Jinja2Forms(directory=var_1, package=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_5.env).__module__}.{type(var_5.env).__qualname__}' == 'jinja2.environment.Environment'
    var_6 = module_3.String(format=var_2, **var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.String'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.allow_blank is False
    assert var_6.trim_whitespace is True
    assert var_6.max_length is None
    assert var_6.min_length is None
    assert var_6.format == 'tn\x0b'
    assert var_6.coerce_types is True
    assert var_6.pattern is None
    assert var_6.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = {var_1: var_6}
    var_8 = module_2.Schema(var_7, **var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 1
    assert var_8.required == ['phone']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_9 = module_0.Form(env=var_0, schema=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_9.schema).__module__}.{type(var_9.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.values is None
    assert var_9.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5.create_form(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_7():
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
    var_1 = 'optional'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_3.String(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length is None
    assert var_5.min_length is None
    assert var_5.format is None
    assert var_5.coerce_types is True
    assert var_5.pattern is None
    assert var_5.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 1
    assert var_8.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_9 = module_0.Form(env=var_0, schema=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_9.schema).__module__}.{type(var_9.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.values is None
    assert var_9.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_10 = 'allow_null'
    var_11 = {var_10: var_2}
    var_12 = module_3.String(**var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.default is None
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is True
    assert var_12.read_only is False
    assert var_12.allow_blank is False
    assert var_12.trim_whitespace is True
    assert var_12.max_length is None
    assert var_12.min_length is None
    assert var_12.format is None
    assert var_12.coerce_types is True
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    var_9.render_field(field_name=var_1, field=var_12)

@pytest.mark.xfail(strict=True)
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
    var_1 = 'phone'
    var_2 = '/'
    var_3 = {}
    var_4 = module_3.String(format=var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is None
    assert var_4.format == '/'
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 1
    assert var_7.required == ['phone']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = module_0.Form(env=var_0, schema=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_8.schema).__module__}.{type(var_8.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.values is None
    assert var_8.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_9 = {}
    var_10 = '+1234567890'
    var_11 = var_8.validate(var_9)
    assert f'{type(var_8.errors).__module__}.{type(var_8.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8.errors) == 1
    assert var_8.data == {}
    var_8.render_field(field_name=var_1, field=var_1, value=var_10)

@pytest.mark.xfail(strict=True)
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
    var_1 = 'unknown'
    var_2 = {}
    var_3 = module_3.String(format=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'unknown'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_2.Schema(var_2, **var_2)
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
    var_5.render_field(field_name=var_1, field=var_3)

@pytest.mark.xfail(strict=True)
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
    var_1 = 'desJri<tion'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_3.String(format=var_2, **var_3)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 1
    assert var_7.required == ['desJri<tion']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = module_0.Form(env=var_0, schema=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_8.schema).__module__}.{type(var_8.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.values is None
    assert var_8.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_9 = {}
    var_10 = module_3.String(format=var_2, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length is None
    assert var_10.min_length is None
    assert var_10.format == 'text'
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = 'Some text'
    var_8.render_field(field_name=var_1, field=var_10, value=var_11)

@pytest.mark.xfail(strict=True)
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
    var_1 = 'phone'
    var_2 = {}
    var_3 = module_3.String(format=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'phone'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {}
    var_5 = module_2.Schema(var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.fields == {}
    assert var_5.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = module_0.Form(env=var_0, schema=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_6.env).__module__}.{type(var_6.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_6.schema).__module__}.{type(var_6.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.values is None
    assert var_6.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_7 = {var_0: var_2}
    var_8 = '+1234567890'
    var_9 = var_6.validate(var_7)
    assert f'{type(var_6.errors).__module__}.{type(var_6.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.errors) == 1
    assert f'{type(var_6.data).__module__}.{type(var_6.data).__qualname__}' == 'builtins.dict'
    assert len(var_6.data) == 1
    var_6.render_field(field_name=var_1, field=var_3, value=var_8)

@pytest.mark.xfail(strict=True)
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
    var_1 = 'desJri<tion'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_3.String(format=var_2, **var_3)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 1
    assert var_7.required == ['desJri<tion']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = module_0.Form(env=var_0, schema=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_8.schema).__module__}.{type(var_8.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.values is None
    assert var_8.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_9 = {}
    var_10 = module_3.String(format=var_2, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length is None
    assert var_10.min_length is None
    assert var_10.format == 'text'
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = None
    var_12 = var_8.template_for_field(var_11)
    assert var_12 == 'forms/input.html'
    var_13 = 'Some text'
    var_8.render_field(field_name=var_1, field=var_10, value=var_13)

@pytest.mark.xfail(strict=True)
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
    var_1 = 'name'
    var_2 = {}
    var_3 = module_3.String(**var_2)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_0, schema=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = module_3.String(**var_5)
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
    var_7.__str__()

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
    var_1 = 'phone'
    var_2 = {}
    var_3 = module_3.String(format=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'phone'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['phone']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_0, schema=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = {}
    var_9 = module_3.String(format=var_1, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length is None
    assert var_9.min_length is None
    assert var_9.format == 'phone'
    assert var_9.coerce_types is True
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    var_10 = var_7.validate(var_8)
    assert f'{type(var_7.errors).__module__}.{type(var_7.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.errors) == 1
    assert var_7.data == {}
    var_7.__html__()

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
    var_1 = 'phone'
    var_2 = 'tel'
    var_3 = {}
    var_4 = module_3.String(format=var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is None
    assert var_4.format == 'tel'
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 1
    assert var_7.required == ['phone']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = module_0.Form(env=var_0, schema=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_8.schema).__module__}.{type(var_8.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.values is None
    assert var_8.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_9 = {}
    var_10 = module_3.String(format=var_2, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length is None
    assert var_10.min_length is None
    assert var_10.format == 'tel'
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = None
    var_12 = var_8.validate(var_11)
    assert f'{type(var_8.errors).__module__}.{type(var_8.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8.errors) == 1
    assert var_8.data is None
    with pytest.raises(AssertionError):
        var_8.validate(var_9)

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
    var_1 = 'active'
    var_2 = {}
    var_3 = module_3.Boolean(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_4 = {var_1: var_3}
    var_5 = None
    var_6 = var_0.getitem(var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'jinja2.runtime.Undefined'
    assert len(var_6) == 0
    var_7 = {}
    var_8 = module_2.Schema(var_4, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 1
    assert var_8.required == ['active']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_9 = module_0.Form(env=var_0, schema=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_9.schema).__module__}.{type(var_9.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.values is None
    assert var_9.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_10 = {}
    var_11 = module_3.Boolean(**var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.coerce_types is True
    var_12 = var_11.serialize(var_5)
    var_13 = True
    var_9.render_field(field_name=var_1, field=var_11, value=var_13)

def test_case_17():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'j~>,+z\x0c 4qs=\t'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_3.Choice(choices=var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Choice'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.choices == [('a', 'A'), ('j~>,+z\x0c 4qs=\t', 'B')]
    assert var_8.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_9 = module_1.Environment()
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
    assert var_9.autoescape is False
    assert f'{type(var_9.filters).__module__}.{type(var_9.filters).__qualname__}' == 'builtins.dict'
    assert len(var_9.filters) == 54
    assert f'{type(var_9.tests).__module__}.{type(var_9.tests).__qualname__}' == 'builtins.dict'
    assert len(var_9.tests) == 39
    assert f'{type(var_9.globals).__module__}.{type(var_9.globals).__qualname__}' == 'builtins.dict'
    assert len(var_9.globals) == 6
    assert var_9.loader is None
    assert f'{type(var_9.cache).__module__}.{type(var_9.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_9.cache) == 0
    assert var_9.bytecode_cache is None
    assert var_9.auto_reload is True
    assert var_9.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_9.extensions == {}
    assert var_9.is_async is False
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
    var_10 = 'test'
    var_11 = {var_10: var_8}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.fields).__module__}.{type(var_13.fields).__qualname__}' == 'builtins.dict'
    assert len(var_13.fields) == 1
    assert var_13.required == ['test']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_14 = module_0.Form(env=var_9, schema=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_14.env).__module__}.{type(var_14.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_14.schema).__module__}.{type(var_14.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_14.values is None
    assert var_14.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_15 = var_14.template_for_field(var_8)
    assert var_15 == 'forms/select.html'

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
    var_1 = 'id'
    var_2 = 'name'
    var_3 = True
    var_4 = 'read_only'
    var_5 = {var_4: var_3}
    var_6 = module_3.Integer(**var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Integer'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is True
    assert var_6.minimum is None
    assert var_6.maximum is None
    assert var_6.exclusive_minimum is None
    assert var_6.exclusive_maximum is None
    assert var_6.multiple_of is None
    assert var_6.precision is None
    assert var_6.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_7 = {}
    var_8 = module_3.String(**var_7)
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
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = {var_1: var_6, var_2: var_8}
    var_10 = module_2.Schema(var_9, **var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is True
    assert f'{type(var_10.fields).__module__}.{type(var_10.fields).__qualname__}' == 'builtins.dict'
    assert len(var_10.fields) == 2
    assert var_10.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_11 = module_0.Form(env=var_0, schema=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_11.env).__module__}.{type(var_11.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_11.schema).__module__}.{type(var_11.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.values is None
    assert var_11.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_11.render_fields()