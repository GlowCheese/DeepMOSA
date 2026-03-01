# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import jinja2.utils as module_4
import jinja2.loaders as module_5
import json.encoder as module_6

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
    var_1 = 'name'
    var_2 = module_2.String()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_1: var_2}
    var_4 = module_3.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['name']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_0.Form(env=var_0, schema=var_4, values=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_5.env).__module__}.{type(var_5.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values == {}
    assert var_5.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    with pytest.raises(TypeError):
        var_6 = str(var_5)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_4._PassArg.eval_context
    module_0.Jinja2Forms(directory=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ''
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_2 = var_1.load_template_env(directory=var_0)
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
    assert var_2.autoescape is True
    assert f'{type(var_2.filters).__module__}.{type(var_2.filters).__qualname__}' == 'builtins.dict'
    assert len(var_2.filters) == 54
    assert f'{type(var_2.tests).__module__}.{type(var_2.tests).__qualname__}' == 'builtins.dict'
    assert len(var_2.tests) == 39
    assert f'{type(var_2.globals).__module__}.{type(var_2.globals).__qualname__}' == 'builtins.dict'
    assert len(var_2.globals) == 6
    assert f'{type(var_2.loader).__module__}.{type(var_2.loader).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.extensions == {}
    assert var_2.is_async is False
    assert module_1.Environment.sandboxed is False
    assert module_1.Environment.overlayed is False
    assert module_1.Environment.linked_to is None
    assert module_1.Environment.shared is False
    assert f'{type(module_1.Environment.lexer).__module__}.{type(module_1.Environment.lexer).__qualname__}' == 'builtins.property'
    var_3 = var_0.__str__()
    assert var_3 == ''
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
    module_0.Jinja2Forms(package=var_0)

def test_case_4():
    var_0 = None
    var_1 = ''
    var_2 = module_0.Jinja2Forms(directory=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_2.env).__module__}.{type(var_2.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        var_2.load_template_env(directory=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = ''
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_1.create_form(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = ''
    module_0.Jinja2Forms(directory=var_0, package=var_0)

def test_case_7():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_2.String()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_2.Integer()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_3.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 2
    assert var_5.required == ['name', 'age']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = module_5.BaseLoader()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert module_5.BaseLoader.has_source_access is True
    var_7 = module_1.Environment(loader=var_6)
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
    assert f'{type(var_7.loader).__module__}.{type(var_7.loader).__qualname__}' == 'jinja2.loaders.BaseLoader'
    assert f'{type(var_7.cache).__module__}.{type(var_7.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_7.cache) == 0
    assert var_7.bytecode_cache is None
    assert var_7.auto_reload is True
    assert var_7.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_7.extensions == {}
    assert var_7.is_async is False
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
    var_8 = 'John'
    var_9 = module_0.Form(env=var_7, schema=var_5, values=var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_9.schema).__module__}.{type(var_9.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_9.values).__module__}.{type(var_9.values).__qualname__}' == 'builtins.dict'
    assert len(var_9.values) == 2
    assert var_9.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_10 = 'Jane'
    var_11 = 25
    var_12 = {var_0: var_10, var_1: var_11}
    var_13 = var_9.validate(var_12)
    assert var_9.values == {'name': 'Jane', 'age': 25}
    assert var_9.data == {'name': 'Jane', 'age': 25}
    var_14 = {var_0: var_8, var_1: var_11}
    var_15 = module_0.Form(env=var_7, schema=var_5, values=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_15.env).__module__}.{type(var_15.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_15.schema).__module__}.{type(var_15.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_15.values == {'name': 'John', 'age': 25}
    assert var_15.errors is None
    var_16 = ''
    var_17 = 'invalid'
    var_18 = {var_0: var_16, var_1: var_17}
    var_19 = var_15.validate(var_18)
    assert var_15.values is None
    assert f'{type(var_15.errors).__module__}.{type(var_15.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15.errors) == 2
    assert var_15.data == {'name': '', 'age': 'invalid'}
    var_20 = {var_8: var_8, var_1: var_8}
    var_21 = module_0.Form(env=var_7, schema=var_5, values=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_21.env).__module__}.{type(var_21.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_21.schema).__module__}.{type(var_21.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_21.values == {'age': 'John'}
    assert var_21.errors is None
    var_22 = {var_0: var_10, var_1: var_11}
    var_23 = var_21.validate(var_22)
    assert var_21.values == {'name': 'Jane', 'age': 25}
    assert var_21.data == {'name': 'Jane', 'age': 25}
    var_24 = 'name'
    var_25 = 'age'
    var_26 = 'Jane'
    var_27 = 25
    var_28 = {var_24: var_26, var_25: var_27}
    with pytest.raises(AssertionError):
        var_21.validate(var_28)

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
    var_1 = {}
    var_2 = module_3.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Form(env=var_0, schema=var_2, values=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_3.schema).__module__}.{type(var_3.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.values == {}
    assert var_3.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'
    var_5 = module_2.String()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = var_3.input_type_for_field(var_5)
    assert var_6 == 'text'
    var_7 = 'unknown'
    var_8 = module_2.String(format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format == 'unknown'
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    var_9 = var_3.input_type_for_field(var_8)
    assert var_9 == 'text'
    var_10 = var_3.__str__()
    assert var_10 == ''
    module_0.Jinja2Forms(directory=var_9, package=var_9)

def test_case_9():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for option in field.choices %}<option value="{{ option.value }}">{{ option.display_name }}</option>{% endfor %}</select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_5.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">', 'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>', 'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}">{% for option in field.choices %}<option value="{{ option.value }}">{{ option.display_name }}</option>{% endfor %}</select>', 'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'}
    var_10 = module_1.Environment(loader=var_9)
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
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'description'
    var_14 = 'country'
    var_15 = 'subscribe'
    var_16 = 'number'
    var_17 = module_2.String(format=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.String'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.allow_blank is False
    assert var_17.trim_whitespace is True
    assert var_17.max_length is None
    assert var_17.min_length is None
    assert var_17.format == 'number'
    assert var_17.coerce_types is True
    assert var_17.pattern is None
    assert var_17.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
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
    var_20 = 'US'
    var_21 = (var_20, var_14)
    var_22 = 'UK'
    var_23 = 'United Kingdom'
    var_24 = (var_22, var_23)
    var_25 = [var_21, var_24]
    var_26 = module_2.Choice(choices=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Choice'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.choices == [('US', 'country'), ('UK', 'United Kingdom')]
    assert var_26.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_27 = module_2.Boolean()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_28 = {var_11: var_17, var_12: var_17, var_13: var_19, var_14: var_26, var_15: var_27}
    var_29 = module_3.Schema(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.fields).__module__}.{type(var_29.fields).__qualname__}' == 'builtins.dict'
    assert len(var_29.fields) == 5
    assert var_29.required == ['name', 'age', 'description', 'country', 'subscribe']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_30 = 'John'
    var_31 = '25'
    var_32 = 'Test'
    var_33 = True
    var_34 = {var_11: var_30, var_12: var_31, var_13: var_32, var_14: var_20, var_15: var_33}
    var_35 = module_0.Form(env=var_10, schema=var_29, values=var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_35.env).__module__}.{type(var_35.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_35.schema).__module__}.{type(var_35.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_35.values == {'name': 'John', 'age': '25', 'description': 'Test', 'country': 'US', 'subscribe': True}
    assert var_35.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_36 = var_35.render_fields()
    assert var_36 == '<input id="name" name="name" type="number" value="John"><input id="age" name="age" type="number" value="25"><textarea id="description" name="description">Test</textarea><select id="country" name="country"><option value=""></option><option value=""></option></select><input id="subscribe" name="subscribe" type="checkbox" checked>'
    assert len(var_10.cache) == 4
    var_37 = 'invalid'
    var_38 = 'INVALID'
    var_39 = 'not_a_boolean'
    var_40 = {var_11: var_23, var_12: var_37, var_13: var_23, var_14: var_38, var_15: var_39}
    var_41 = var_35.validate(var_40)
    assert var_35.values is None
    assert f'{type(var_35.errors).__module__}.{type(var_35.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_35.errors) == 2
    assert var_35.data == {'name': 'United Kingdom', 'age': 'invalid', 'description': 'United Kingdom', 'country': 'INVALID', 'subscribe': 'not_a_boolean'}
    var_42 = var_35.render_fields()
    assert var_42 == '<input id="name" name="name" type="number" value="United Kingdom"><input id="age" name="age" type="number" value="invalid"><textarea id="description" name="description">United Kingdom</textarea><select id="country" name="country"><option value=""></option><option value=""></option></select><input id="subscribe" name="subscribe" type="checkbox" checked>'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'forms/input.html'
    var_1 = 'forms/select.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_6 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_7 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_5.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">', 'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>', 'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>', 'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'}
    var_10 = module_1.Environment(loader=var_9)
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
    var_11 = 'text_field'
    var_12 = 'email_field'
    var_13 = 'number_field'
    var_14 = 'password_field'
    var_15 = 'choice_field'
    var_16 = 'bool_field'
    var_17 = 'textarea_field'
    var_18 = module_2.String()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.String'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.allow_blank is False
    assert var_18.trim_whitespace is True
    assert var_18.max_length is None
    assert var_18.min_length is None
    assert var_18.format is None
    assert var_18.coerce_types is True
    assert var_18.pattern is None
    assert var_18.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_19 = 'email'
    var_20 = module_2.String(format=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.String'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.allow_blank is False
    assert var_20.trim_whitespace is True
    assert var_20.max_length is None
    assert var_20.min_length is None
    assert var_20.format == 'email'
    assert var_20.coerce_types is True
    assert var_20.pattern is None
    assert var_20.pattern_regex is None
    var_21 = 'number'
    var_22 = module_2.String(format=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.String'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.allow_blank is False
    assert var_22.trim_whitespace is True
    assert var_22.max_length is None
    assert var_22.min_length is None
    assert var_22.format == 'number'
    assert var_22.coerce_types is True
    assert var_22.pattern is None
    assert var_22.pattern_regex is None
    var_23 = 'password'
    var_24 = module_2.String(format=var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.String'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.allow_blank is False
    assert var_24.trim_whitespace is True
    assert var_24.max_length is None
    assert var_24.min_length is None
    assert var_24.format == 'password'
    assert var_24.coerce_types is True
    assert var_24.pattern is None
    assert var_24.pattern_regex is None
    var_25 = 'c'
    var_26 = [var_21, var_19, var_25]
    var_27 = module_2.Choice(choices=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Choice'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.choices == [('number', 'number'), ('email', 'email'), ('c', 'c')]
    assert var_27.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_28 = module_2.Boolean()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_29 = 'text'
    var_30 = module_2.String(format=var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.String'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.allow_blank is False
    assert var_30.trim_whitespace is True
    assert var_30.max_length is None
    assert var_30.min_length is None
    assert var_30.format == 'text'
    assert var_30.coerce_types is True
    assert var_30.pattern is None
    assert var_30.pattern_regex is None
    var_31 = {var_11: var_18, var_12: var_20, var_13: var_22, var_14: var_24, var_15: var_27, var_16: var_28, var_17: var_30}
    var_32 = module_3.Schema(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.fields).__module__}.{type(var_32.fields).__qualname__}' == 'builtins.dict'
    assert len(var_32.fields) == 7
    assert var_32.required == ['text_field', 'email_field', 'number_field', 'password_field', 'choice_field', 'bool_field', 'textarea_field']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_33 = module_0.Form(env=var_10, schema=var_32, values=var_8)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_33.env).__module__}.{type(var_33.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_33.schema).__module__}.{type(var_33.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_33.values == {}
    assert var_33.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_34 = var_32.fields[var_11]
    var_35 = 'test'
    var_36 = var_33.render_field(field_name=var_11, field=var_34, value=var_35)
    assert var_36 == '<input id="text-field" name="text_field" type="text" value="test">'
    assert len(var_10.cache) == 1
    var_37 = var_32.fields[var_12]
    var_38 = 'test@example.com'
    var_39 = var_33.render_field(field_name=var_12, field=var_37, value=var_38)
    assert var_39 == '<input id="email-field" name="email_field" type="email" value="test@example.com">'
    var_40 = var_32.fields[var_13]
    var_41 = '123'
    var_42 = var_33.render_field(field_name=var_13, field=var_40, value=var_41)
    assert var_42 == '<input id="number-field" name="number_field" type="number" value="123">'
    var_43 = var_32.fields[var_14]
    var_44 = 'secret'
    var_45 = var_33.render_field(field_name=var_14, field=var_43, value=var_44)
    assert var_45 == '<input id="password-field" name="password_field" type="password" value="">'
    var_46 = var_32.fields[var_15]
    var_47 = var_33.render_field(field_name=var_15, field=var_46, value=var_46)
    assert var_47 == '<select id="choice-field" name="choice_field"></select>'
    assert len(var_10.cache) == 2
    var_48 = True
    var_33.render_field(field_name=var_16, field=var_45, value=var_48)

def test_case_11():
    var_0 = 'forms/input.html'
    var_1 = 'forms/select.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_6 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_2, var_3: var_6}
    var_8 = module_5.DictLoader(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_8.mapping == {'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">', 'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>', 'forms/checkbox.html': 'forms/checkbox.html', 'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'}
    var_9 = module_1.Environment(loader=var_8)
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
    assert f'{type(var_9.loader).__module__}.{type(var_9.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
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
    var_10 = 'name'
    var_11 = 'age'
    var_12 = 'description'
    var_13 = 'active'
    var_14 = 'type'
    var_15 = module_2.String()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.String'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.allow_blank is False
    assert var_15.trim_whitespace is True
    assert var_15.max_length is None
    assert var_15.min_length is None
    assert var_15.format is None
    assert var_15.coerce_types is True
    assert var_15.pattern is None
    assert var_15.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_16 = 'number'
    var_17 = module_2.String(format=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.String'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.allow_blank is False
    assert var_17.trim_whitespace is True
    assert var_17.max_length is None
    assert var_17.min_length is None
    assert var_17.format == 'number'
    assert var_17.coerce_types is True
    assert var_17.pattern is None
    assert var_17.pattern_regex is None
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
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
    var_20 = module_2.Boolean()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_21 = 'A'
    var_22 = 'B'
    var_23 = 'C'
    var_24 = [var_21, var_22, var_23]
    var_25 = module_2.Choice(choices=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Choice'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.choices == [('A', 'A'), ('B', 'B'), ('C', 'C')]
    assert var_25.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_26 = {var_10: var_15, var_11: var_17, var_12: var_19, var_13: var_20, var_14: var_25}
    var_27 = module_3.Schema(var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.fields).__module__}.{type(var_27.fields).__qualname__}' == 'builtins.dict'
    assert len(var_27.fields) == 5
    assert var_27.required == ['name', 'age', 'description', 'active', 'type']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_28 = module_0.Form(env=var_9, schema=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_28.env).__module__}.{type(var_28.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_28.schema).__module__}.{type(var_28.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_28.values is None
    assert var_28.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_29 = str(var_28)

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
    var_2 = module_3.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = {}
    var_4 = module_0.Form(env=var_0, schema=var_2, values=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values == {}
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = module_2.String()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = var_4.input_type_for_field(var_5)
    assert var_6 == 'text'
    var_7 = var_4.input_type_for_field(var_5)
    assert var_7 == 'text'
    var_8 = module_2.String(format=var_6)
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
    var_9 = var_4.input_type_for_field(var_8)
    assert var_9 == 'text'
    var_10 = module_2.Boolean()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}

def test_case_13():
    var_0 = None
    var_1 = 'test'
    var_2 = module_2.Boolean()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_3 = {var_1: var_2}
    var_4 = module_3.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['test']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_0.Form(env=var_0, schema=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert var_5.env is None
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values is None
    assert var_5.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_6 = var_5.template_for_field(var_2)
    assert var_6 == 'forms/checkbox.html'
    var_7 = module_2.String()
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = {var_1: var_7}
    var_9 = module_3.Schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.fields).__module__}.{type(var_9.fields).__qualname__}' == 'builtins.dict'
    assert len(var_9.fields) == 1
    assert var_9.required == ['test']
    var_10 = module_0.Form(env=var_0, schema=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.forms.Form'
    assert var_10.env is None
    assert f'{type(var_10.schema).__module__}.{type(var_10.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_10.values is None
    assert var_10.errors is None
    var_11 = var_10.template_for_field(var_7)
    assert var_11 == 'forms/input.html'
    var_12 = 'subfield'
    var_13 = module_2.String()
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
    var_14 = {var_12: var_13}
    var_15 = module_2.Object()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Object'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.properties == {}
    assert var_15.pattern_properties == {}
    assert var_15.additional_properties is True
    assert var_15.property_names is None
    assert var_15.min_properties is None
    assert var_15.max_properties is None
    assert var_15.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_16 = {var_1: var_15}
    var_17 = module_3.Schema(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.fields).__module__}.{type(var_17.fields).__qualname__}' == 'builtins.dict'
    assert len(var_17.fields) == 1
    assert var_17.required == ['test']
    var_18 = module_0.Form(env=var_0, schema=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.forms.Form'
    assert var_18.env is None
    assert f'{type(var_18.schema).__module__}.{type(var_18.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_18.values is None
    assert var_18.errors is None
    with pytest.raises(AssertionError):
        var_18.template_for_field(var_15)

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
    var_1 = 'name'
    var_2 = module_2.String()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_1: var_2}
    var_4 = module_3.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['name']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_0.Form(env=var_0, schema=var_4, values=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_5.env).__module__}.{type(var_5.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values == {}
    assert var_5.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5.__html__()

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
    var_2 = module_3.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Form(env=var_0, schema=var_2, values=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_3.schema).__module__}.{type(var_3.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.values == {}
    assert var_3.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'
    var_5 = module_2.String()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = var_3.input_type_for_field(var_5)
    assert var_6 == 'text'
    var_7 = var_3.input_type_for_field(var_5)
    assert var_7 == 'text'
    var_8 = 'unknown'
    var_9 = module_2.String(format=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length is None
    assert var_9.min_length is None
    assert var_9.format == 'unknown'
    assert var_9.coerce_types is True
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    var_10 = var_3.input_type_for_field(var_9)
    assert var_10 == 'text'
    module_0.Jinja2Forms(directory=var_10, package=var_10)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'forms/input.html'
    var_1 = 'xp\\ZlX2bQ'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for option in field.choices %}<option value="{{ option.value }}">{{ option.display_name }}</option>{% endfor %}</select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_5.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">', 'xp\\ZlX2bQ': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>', 'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}">{% for option in field.choices %}<option value="{{ option.value }}">{{ option.display_name }}</option>{% endfor %}</select>', 'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'}
    var_10 = module_6.py_encode_basestring_ascii(var_0)
    assert var_10 == '"forms/input.html"'
    assert f'{type(module_6.ESCAPE).__module__}.{type(module_6.ESCAPE).__qualname__}' == 're.Pattern'
    assert f'{type(module_6.ESCAPE_ASCII).__module__}.{type(module_6.ESCAPE_ASCII).__qualname__}' == 're.Pattern'
    assert f'{type(module_6.HAS_UTF8).__module__}.{type(module_6.HAS_UTF8).__qualname__}' == 're.Pattern'
    assert module_6.ESCAPE_DCT == {'\\': '\\\\', '"': '\\"', '\x08': '\\b', '\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\x00': '\\u0000', '\x01': '\\u0001', '\x02': '\\u0002', '\x03': '\\u0003', '\x04': '\\u0004', '\x05': '\\u0005', '\x06': '\\u0006', '\x07': '\\u0007', '\x0b': '\\u000b', '\x0e': '\\u000e', '\x0f': '\\u000f', '\x10': '\\u0010', '\x11': '\\u0011', '\x12': '\\u0012', '\x13': '\\u0013', '\x14': '\\u0014', '\x15': '\\u0015', '\x16': '\\u0016', '\x17': '\\u0017', '\x18': '\\u0018', '\x19': '\\u0019', '\x1a': '\\u001a', '\x1b': '\\u001b', '\x1c': '\\u001c', '\x1d': '\\u001d', '\x1e': '\\u001e', '\x1f': '\\u001f'}
    assert module_6.i == 31
    assert module_6.INFINITY == pytest.approx(1e309, abs=0.01, rel=0.01)
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'description'
    var_14 = 'coystry'
    var_15 = 'subscribe'
    var_16 = None
    var_17 = module_2.String(allow_blank=var_5, format=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.String'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.default == ''
    assert var_17.allow_blank == '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    assert var_17.trim_whitespace is True
    assert var_17.max_length is None
    assert var_17.min_length is None
    assert var_17.format is None
    assert var_17.coerce_types is True
    assert var_17.pattern is None
    assert var_17.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_18 = 'UK'
    var_19 = 'bX0[&0e`]CnSR'
    var_20 = (var_18, var_19)
    var_21 = [var_20, var_20]
    var_22 = module_2.Choice(choices=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Choice'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.choices == [('UK', 'bX0[&0e`]CnSR'), ('UK', 'bX0[&0e`]CnSR')]
    assert var_22.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_23 = module_2.Boolean()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_24 = {var_11: var_17, var_12: var_17, var_13: var_17, var_14: var_22, var_15: var_23}
    var_25 = module_0.Jinja2Forms(directory=var_13)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_25.env).__module__}.{type(var_25.env).__qualname__}' == 'jinja2.environment.Environment'
    var_26 = module_3.Schema(var_24)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.fields).__module__}.{type(var_26.fields).__qualname__}' == 'builtins.dict'
    assert len(var_26.fields) == 5
    assert var_26.required == ['coystry', 'subscribe']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_27 = module_0.Form(env=var_10, schema=var_26, values=var_8)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.forms.Form'
    assert var_27.env == '"forms/input.html"'
    assert f'{type(var_27.schema).__module__}.{type(var_27.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_27.values == {}
    assert var_27.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_27.render_fields()