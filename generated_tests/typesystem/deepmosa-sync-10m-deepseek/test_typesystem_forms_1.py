# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import jinja2.filters as module_3
import typesystem.fields as module_4
import posixpath as module_5
import markupsafe as module_6

def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = "*j\t_'mh5B"
    module_0.Jinja2Forms(directory=var_0, package=var_0)

def test_case_2():
    var_0 = "*j\t_'mh5B"
    var_1 = None
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_2.env).__module__}.{type(var_2.env).__qualname__}' == 'jinja2.environment.Environment'

def test_case_3():
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
    var_4 = var_3.template_for_field(var_1)
    assert var_4 == 'forms/input.html'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '@|7&fl1Zoa6= -IPc'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_2 = module_3.do_striptags(var_0)
    assert var_2 == '@|7&fl1Zoa6= -IPc'
    assert f'{type(module_3.F).__module__}.{type(module_3.F).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.K).__module__}.{type(module_3.K).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.V).__module__}.{type(module_3.V).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.FILTERS).__module__}.{type(module_3.FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FILTERS) == 54
    var_3 = var_2.__str__()
    assert var_3 == '@|7&fl1Zoa6= -IPc'
    var_4 = None
    var_5 = var_2.__str__()
    assert var_5 == '@|7&fl1Zoa6= -IPc'
    var_1.create_form(var_4)

def test_case_5():
    var_0 = '*j\t_hB'
    var_1 = None
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_2.env).__module__}.{type(var_2.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        var_2.load_template_env()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '@|7&fl1a6= -IPc'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_1.load_template_env(package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '9'
    module_0.Jinja2Forms(package=var_0)

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
    var_1 = 'email_address'
    var_2 = {}
    var_3 = module_4.String(**var_2)
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
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
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
    assert var_6.required == ['email_address']
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
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert f'{type(var_7.errors).__module__}.{type(var_7.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.errors) == 1
    assert var_7.data is None
    with pytest.raises(AssertionError):
        var_7.validate()

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
    var_1 = True
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = module_4.Integer(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Integer'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is True
    assert var_4.minimum is None
    assert var_4.maximum is None
    assert var_4.exclusive_minimum is None
    assert var_4.exclusive_maximum is None
    assert var_4.multiple_of is None
    assert var_4.precision is None
    assert var_4.coerce_types is True
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    var_5 = {}
    var_6 = module_2.Schema(var_5, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.fields == {}
    assert var_6.required == []
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
    var_8 = var_7.validate()
    assert f'{type(var_7.errors).__module__}.{type(var_7.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.errors) == 1
    assert var_7.data is None
    module_5.realpath(var_7, strict=var_1)

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
    var_1 = 'optional'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_4.String(**var_4)
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
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
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
    var_12 = module_4.String(**var_11)
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
def test_case_11():
    var_0 = 'password'
    var_1 = {}
    var_2 = module_4.String(format=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'password'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_2.Schema(var_1, **var_1)
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
    assert var_4.env == 'password'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4.render_field(field_name=var_0, field=var_2, value=var_0)

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
    var_1 = 'status'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_4.String(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.default == 'status'
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
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
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
    assert var_7.required == []
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
    var_9 = 'default'
    var_10 = {var_9: var_1}
    var_11 = module_4.String(**var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.String'
    assert var_11.default == 'status'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.allow_blank is False
    assert var_11.trim_whitespace is True
    assert var_11.max_length is None
    assert var_11.min_length is None
    assert var_11.format is None
    assert var_11.coerce_types is True
    assert var_11.pattern is None
    assert var_11.pattern_regex is None
    var_8.render_field(field_name=var_1, field=var_11)

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
    var_1 = 'full_name'
    var_2 = 'Full Name'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_4.String(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == 'Full Name'
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length is None
    assert var_5.min_length is None
    assert var_5.format is None
    assert var_5.coerce_types is True
    assert var_5.pattern is None
    assert var_5.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
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
    assert var_8.required == ['full_name']
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
    var_10 = 'title'
    var_11 = {var_10: var_2}
    var_12 = module_4.String(**var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == 'Full Name'
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
    var_9.render_field(field_name=var_1, field=var_12)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = ''
    var_1 = {}
    var_2 = module_4.String(format=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == ''
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_2.Schema(var_1, **var_1)
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
    assert var_4.env == ''
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4.render_field(field_name=var_0, field=var_2, value=var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'password'
    var_1 = {}
    var_2 = module_4.String(format=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'password'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = module_2.Schema(var_3, **var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['password']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_0.Form(env=var_0, schema=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert var_5.env == 'password'
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values is None
    assert var_5.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5.__str__()

def test_case_16():
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
    var_2 = module_0.Form(env=var_1, schema=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_2.env).__module__}.{type(var_2.env).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_2.schema).__module__}.{type(var_2.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.values is None
    assert var_2.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_3 = var_2.render_fields()
    assert var_3 == ''

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'password'
    var_1 = {}
    var_2 = module_4.String(format=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'password'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_2.Schema(var_1, **var_1)
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
    assert var_4.env == 'password'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = var_4.__str__()
    assert var_5 == ''
    var_6 = 'secret'
    var_4.render_field(field_name=var_0, field=var_2, value=var_6)

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
    var_1 = 'birthdate'
    var_2 = 'date'
    var_3 = {}
    var_4 = module_4.String(format=var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is None
    assert var_4.format == 'date'
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = {}
    var_6 = module_2.Schema(var_5, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.fields == {}
    assert var_6.required == []
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
    var_8 = None
    var_9 = var_7.input_type_for_field(var_8)
    assert var_9 == 'text'
    var_10 = var_7.render_fields()
    assert var_10 == ''
    var_11 = '2023-01-01'
    var_12 = var_7.__html__()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'markupsafe.Markup'
    assert len(var_12) == 0
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'
    var_7.render_field(field_name=var_1, field=var_4, value=var_11)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'pwd'
    var_1 = {}
    var_2 = module_4.String(format=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'pwd'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_5.env).__module__}.{type(var_5.env).__qualname__}' == 'jinja2.environment.Environment'
    var_6 = module_2.Schema(var_3, **var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['pwd']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_3, schema=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'builtins.dict'
    assert len(var_7.env) == 1
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = var_7.validate(var_1)
    assert f'{type(var_7.errors).__module__}.{type(var_7.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.errors) == 1
    assert var_7.data == {}
    var_7.__str__()

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
    var_8 = var_4.template_for_field(var_7)
    assert var_8 == 'forms/textarea.html'

def test_case_21():
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
    var_4 = 'a'
    var_5 = 'A'
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_4.Choice(choices=var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Choice'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.choices == [('a', 'A')]
    assert var_9.coerce_types is True
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_10 = var_3.template_for_field(var_9)
    assert var_10 == 'forms/select.html'

@pytest.mark.xfail(strict=True)
def test_case_22():
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
    var_3 = module_4.Boolean(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert f'{type(module_4.NO_DEFAULT).__module__}.{type(module_4.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_4.FORMATS).__module__}.{type(module_4.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_4.FORMATS) == 7
    assert module_4.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_4.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_4.Boolean.coerce_null_values == {'', 'null', 'none'}
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
    assert var_6.required == ['active']
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
    var_9 = module_4.Boolean(**var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.coerce_types is True
    var_10 = True
    var_7.render_field(field_name=var_1, field=var_9, value=var_10)