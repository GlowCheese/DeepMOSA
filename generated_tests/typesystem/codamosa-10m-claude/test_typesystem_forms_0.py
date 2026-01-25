# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2
import typesystem.fields as module_3
import typesystem.schemas as module_4
import ast as module_5
import markupsafe as module_6

def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'gu.Um;'
    module_0.Jinja2Forms(package=var_0)

def test_case_2():
    var_0 = 'forms/input.html'
    var_1 = {var_0: var_0}
    var_2 = module_1.DictLoader(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_2.mapping == {'forms/input.html': 'forms/input.html'}
    var_3 = module_2.Environment(loader=var_2)
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
    var_4 = module_3.String()
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
    var_5 = {var_0: var_4}
    var_6 = module_4.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['forms/input.html']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_3, schema=var_6, values=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values == {'forms/input.html': 'forms/input.html'}
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = var_7.render_fields()
    assert var_8 == 'forms/input.html'
    assert len(var_3.cache) == 1

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'Q5\x0c&eS.\rE|W/'
    module_0.Jinja2Forms(directory=var_0, package=var_0)

def test_case_4():
    var_0 = './templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_2 = 'typesystem'
    var_3 = module_0.Jinja2Forms(package=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_3.env).__module__}.{type(var_3.env).__qualname__}' == 'jinja2.environment.Environment'
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_4.env).__module__}.{type(var_4.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

def test_case_5():
    var_0 = 'gu.Um;'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_2 = module_5.iter_child_nodes(var_0)
    assert module_5.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_5.PyCF_ONLY_AST == 1024
    assert module_5.PyCF_TYPE_COMMENTS == 4096
    with pytest.raises(AssertionError):
        var_1.load_template_env()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'gu.Um;'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_1.create_form(var_1)

def test_case_7():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = 'age'
    var_3 = 100
    var_4 = module_3.String(max_length=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length == 100
    assert var_4.min_length is None
    assert var_4.format is None
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_3.String(format=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length is None
    assert var_5.min_length is None
    assert var_5.format == 'email'
    assert var_5.coerce_types is True
    assert var_5.pattern is None
    assert var_5.pattern_regex is None
    var_6 = module_3.Field()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Field'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert module_3.Field.errors == {}
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6}
    var_8 = module_4.Schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 3
    assert var_8.required == ['name', 'email', 'age']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_9 = {}
    var_10 = module_1.DictLoader(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_10.mapping == {}
    var_11 = module_2.Environment(loader=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'jinja2.environment.Environment'
    assert var_11.block_start_string == '{%'
    assert var_11.block_end_string == '%}'
    assert var_11.variable_start_string == '{{'
    assert var_11.variable_end_string == '}}'
    assert var_11.comment_start_string == '{#'
    assert var_11.comment_end_string == '#}'
    assert var_11.line_statement_prefix is None
    assert var_11.line_comment_prefix is None
    assert var_11.trim_blocks is False
    assert var_11.lstrip_blocks is False
    assert var_11.newline_sequence == '\n'
    assert var_11.keep_trailing_newline is False
    assert var_11.optimized is True
    assert var_11.finalize is None
    assert var_11.autoescape is False
    assert f'{type(var_11.filters).__module__}.{type(var_11.filters).__qualname__}' == 'builtins.dict'
    assert len(var_11.filters) == 54
    assert f'{type(var_11.tests).__module__}.{type(var_11.tests).__qualname__}' == 'builtins.dict'
    assert len(var_11.tests) == 39
    assert f'{type(var_11.globals).__module__}.{type(var_11.globals).__qualname__}' == 'builtins.dict'
    assert len(var_11.globals) == 6
    assert f'{type(var_11.loader).__module__}.{type(var_11.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_11.cache).__module__}.{type(var_11.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_11.cache) == 0
    assert var_11.bytecode_cache is None
    assert var_11.auto_reload is True
    assert var_11.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_11.extensions == {}
    assert var_11.is_async is False
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
    var_12 = None
    var_13 = module_0.Form(env=var_11, schema=var_8, values=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_13.env).__module__}.{type(var_13.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_13.schema).__module__}.{type(var_13.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_13.values is None
    assert var_13.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_14 = var_13.validate()
    assert f'{type(var_13.errors).__module__}.{type(var_13.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13.errors) == 1
    assert var_13.data is None
    with pytest.raises(AssertionError):
        var_13.validate()

def test_case_8():
    var_0 = 'name'
    var_1 = True
    var_2 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = module_4.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['name']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = {}
    var_6 = module_1.DictLoader(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_6.mapping == {}
    var_7 = module_2.Environment(loader=var_6)
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
    assert f'{type(var_7.loader).__module__}.{type(var_7.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_7.cache).__module__}.{type(var_7.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_7.cache) == 0
    assert var_7.bytecode_cache is None
    assert var_7.auto_reload is True
    assert var_7.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_7.extensions == {}
    assert var_7.is_async is False
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
    var_8 = None
    var_9 = module_0.Form(env=var_7, schema=var_4, values=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_9.schema).__module__}.{type(var_9.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.values is None
    assert var_9.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_10 = var_9.validate(var_8)
    assert f'{type(var_9.errors).__module__}.{type(var_9.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9.errors) == 1
    assert var_9.data is None

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'name'
    var_1 = True
    var_2 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = module_4.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['name']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = {}
    var_6 = module_1.DictLoader(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_6.mapping == {}
    var_7 = module_2.Environment(loader=var_6)
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
    assert f'{type(var_7.loader).__module__}.{type(var_7.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_7.cache).__module__}.{type(var_7.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_7.cache) == 0
    assert var_7.bytecode_cache is None
    assert var_7.auto_reload is True
    assert var_7.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_7.extensions == {}
    assert var_7.is_async is False
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
    var_8 = None
    var_9 = module_0.Form(env=var_7, schema=var_4, values=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_9.schema).__module__}.{type(var_9.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.values is None
    assert var_9.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_10 = var_9.validate(var_8)
    assert f'{type(var_9.errors).__module__}.{type(var_9.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9.errors) == 1
    assert var_9.data is None
    var_9.render_fields()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value or "" }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>'
    var_7 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value or "" }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_1.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value or "" }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}', 'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}', 'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>', 'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value or "" }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'}
    var_10 = True
    var_11 = module_2.Environment(autoescape=var_10, loader=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'jinja2.environment.Environment'
    assert var_11.block_start_string == '{%'
    assert var_11.block_end_string == '%}'
    assert var_11.variable_start_string == '{{'
    assert var_11.variable_end_string == '}}'
    assert var_11.comment_start_string == '{#'
    assert var_11.comment_end_string == '#}'
    assert var_11.line_statement_prefix is None
    assert var_11.line_comment_prefix is None
    assert var_11.trim_blocks is False
    assert var_11.lstrip_blocks is False
    assert var_11.newline_sequence == '\n'
    assert var_11.keep_trailing_newline is False
    assert var_11.optimized is True
    assert var_11.finalize is None
    assert var_11.autoescape is True
    assert f'{type(var_11.filters).__module__}.{type(var_11.filters).__qualname__}' == 'builtins.dict'
    assert len(var_11.filters) == 54
    assert f'{type(var_11.tests).__module__}.{type(var_11.tests).__qualname__}' == 'builtins.dict'
    assert len(var_11.tests) == 39
    assert f'{type(var_11.globals).__module__}.{type(var_11.globals).__qualname__}' == 'builtins.dict'
    assert len(var_11.globals) == 6
    assert f'{type(var_11.loader).__module__}.{type(var_11.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_11.cache).__module__}.{type(var_11.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_11.cache) == 0
    assert var_11.bytecode_cache is None
    assert var_11.auto_reload is True
    assert var_11.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_11.extensions == {}
    assert var_11.is_async is False
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
    var_12 = 'username'
    var_13 = 'password'
    var_14 = 'email'
    var_15 = 'bio'
    var_16 = 'is_active'
    var_17 = 'role'
    var_18 = 100
    var_19 = module_3.String(max_length=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.String'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.allow_blank is False
    assert var_19.trim_whitespace is True
    assert var_19.max_length == 100
    assert var_19.min_length is None
    assert var_19.format is None
    assert var_19.coerce_types is True
    assert var_19.pattern is None
    assert var_19.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_20 = module_3.String(format=var_13)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.String'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.allow_blank is False
    assert var_20.trim_whitespace is True
    assert var_20.max_length is None
    assert var_20.min_length is None
    assert var_20.format == 'password'
    assert var_20.coerce_types is True
    assert var_20.pattern is None
    assert var_20.pattern_regex is None
    var_21 = module_3.String(format=var_14)
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
    var_22 = 'text'
    var_23 = module_3.String(format=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.String'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.allow_blank is False
    assert var_23.trim_whitespace is True
    assert var_23.max_length is None
    assert var_23.min_length is None
    assert var_23.format == 'text'
    assert var_23.coerce_types is True
    assert var_23.pattern is None
    assert var_23.pattern_regex is None
    var_24 = module_3.Boolean()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.coerce_types is True
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_25 = 'admin'
    var_26 = 'Admin'
    var_27 = (var_25, var_26)
    var_28 = 'user'
    var_29 = 'User'
    var_30 = (var_28, var_29)
    var_31 = [var_27, var_30]
    var_32 = module_3.Choice(choices=var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Choice'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.choices == [('admin', 'Admin'), ('user', 'User')]
    assert var_32.coerce_types is True
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_33 = {var_12: var_19, var_13: var_20, var_14: var_21, var_15: var_23, var_16: var_24, var_17: var_32}
    var_34 = module_4.Schema(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.fields).__module__}.{type(var_34.fields).__qualname__}' == 'builtins.dict'
    assert len(var_34.fields) == 6
    assert var_34.required == ['username', 'password', 'email', 'bio', 'is_active', 'role']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_35 = None
    var_36 = module_0.Form(env=var_11, schema=var_34, values=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_36.env).__module__}.{type(var_36.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_36.schema).__module__}.{type(var_36.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_36.values is None
    assert var_36.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_37 = var_34.fields[var_12]
    var_38 = 'john_doe'
    var_39 = var_36.render_field(field_name=var_12, field=var_37, value=var_38, error=var_35)
    assert var_39 == '<input type="text" name="username" id="username" value="john_doe" required>'
    assert len(var_11.cache) == 1
    var_40 = var_34.fields[var_13]
    var_41 = 'secret123'
    var_42 = var_36.render_field(field_name=var_13, field=var_40, value=var_41, error=var_35)
    assert var_42 == '<input type="password" name="password" id="password" value="" required>'
    var_43 = var_34.fields[var_14]
    var_44 = 'test@example.com'
    var_45 = var_36.render_field(field_name=var_14, field=var_43, value=var_44, error=var_35)
    assert var_45 == '<input type="email" name="email" id="email" value="test@example.com" required>'
    var_46 = 'My bio'
    var_36.render_field(field_name=var_15, field=var_45, value=var_46, error=var_35)

def test_case_11():
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
    var_1 = 'test_choice'
    var_2 = 'test_boolean'
    var_3 = 'test_text'
    var_4 = 'test_email'
    var_5 = 'test_string'
    var_6 = 'a'
    var_7 = 'A'
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 'B'
    var_11 = (var_9, var_10)
    var_12 = [var_8, var_11]
    var_13 = module_3.Choice(choices=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Choice'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.choices == [('a', 'A'), ('b', 'B')]
    assert var_13.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_14 = module_3.Boolean()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.coerce_types is True
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_15 = 'text'
    var_16 = module_3.String(format=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.String'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.allow_blank is False
    assert var_16.trim_whitespace is True
    assert var_16.max_length is None
    assert var_16.min_length is None
    assert var_16.format == 'text'
    assert var_16.coerce_types is True
    assert var_16.pattern is None
    assert var_16.pattern_regex is None
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_17 = 'email'
    var_18 = module_3.String(format=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.String'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.allow_blank is False
    assert var_18.trim_whitespace is True
    assert var_18.max_length is None
    assert var_18.min_length is None
    assert var_18.format == 'email'
    assert var_18.coerce_types is True
    assert var_18.pattern is None
    assert var_18.pattern_regex is None
    var_19 = module_3.String()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.String'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.allow_blank is False
    assert var_19.trim_whitespace is True
    assert var_19.max_length is None
    assert var_19.min_length is None
    assert var_19.format is None
    assert var_19.coerce_types is True
    assert var_19.pattern is None
    assert var_19.pattern_regex is None
    var_20 = {var_1: var_13, var_2: var_14, var_3: var_16, var_4: var_18, var_5: var_19}
    var_21 = module_4.Schema(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.fields).__module__}.{type(var_21.fields).__qualname__}' == 'builtins.dict'
    assert len(var_21.fields) == 5
    assert var_21.required == ['test_choice', 'test_boolean', 'test_text', 'test_email', 'test_string']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_22 = module_0.Form(env=var_0, schema=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_22.env).__module__}.{type(var_22.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_22.schema).__module__}.{type(var_22.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_22.values is None
    assert var_22.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_23 = var_21.fields[var_1]
    var_24 = var_22.template_for_field(var_23)
    assert var_24 == 'forms/select.html'
    var_25 = var_21.fields[var_2]
    var_26 = var_22.template_for_field(var_25)
    assert var_26 == 'forms/checkbox.html'
    var_27 = var_21.fields[var_3]
    var_28 = var_22.template_for_field(var_27)
    assert var_28 == 'forms/textarea.html'
    var_29 = var_21.fields[var_4]
    var_30 = var_22.template_for_field(var_29)
    assert var_30 == 'forms/input.html'
    var_31 = var_21.fields[var_5]
    var_32 = var_22.template_for_field(var_31)
    assert var_32 == 'forms/input.html'
    var_33 = module_3.Object()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Object'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.properties == {}
    assert var_33.pattern_properties == {}
    assert var_33.additional_properties is True
    assert var_33.property_names is None
    assert var_33.min_properties is None
    assert var_33.max_properties is None
    assert var_33.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    with pytest.raises(AssertionError):
        var_22.template_for_field(var_33)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = '<input type="{{ inut_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<spn class="error">{{ error }}</span>{% endif %}'
    var_4 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = {var_0: var_3, var_1: var_4, var_2: var_5, var_2: var_6}
    var_8 = module_1.DictLoader(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_8.mapping == {'forms/input.html': '<input type="{{ inut_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<spn class="error">{{ error }}</span>{% endif %}', 'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}', 'forms/select.html': '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'}
    var_9 = False
    var_10 = module_2.Environment(autoescape=var_9, loader=var_8)
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
    var_11 = 'name'
    var_12 = 'active'
    var_13 = 're_only_field'
    var_14 = module_3.Boolean()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_15 = module_3.String()
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
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_16 = {var_11: var_15, var_1: var_15, var_12: var_14, var_13: var_15}
    var_17 = module_4.Schema(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.fields).__module__}.{type(var_17.fields).__qualname__}' == 'builtins.dict'
    assert len(var_17.fields) == 4
    assert var_17.required == ['name', 'forms/checkbox.html', 'active', 're_only_field']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_18 = 'John'
    var_19 = None
    var_20 = var_17.serialize(var_19)
    var_21 = {var_11: var_18, var_6: var_18, var_12: var_9}
    var_22 = module_0.Form(env=var_10, schema=var_17, values=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_22.env).__module__}.{type(var_22.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_22.schema).__module__}.{type(var_22.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_22.values == {'name': 'John', 'active': False}
    assert var_22.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_23 = var_22.render_fields()
    assert var_23 == '<input type="" name="name" value="John" required id="name" /><input type="" name="forms/checkbox.html" value="None" required id="forms/checkbox.html" /><input type="checkbox" name="active"  id="active" /><input type="" name="re_only_field" value="None" required id="re-only-field" />'
    assert len(var_10.cache) == 2
    var_24 = var_14.__or__(var_15)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Union'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.any_of).__module__}.{type(var_24.any_of).__qualname__}' == 'builtins.list'
    assert len(var_24.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    module_0.Jinja2Forms(directory=var_0, package=var_4)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = '<input type="{{ inut_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<spn class="error">{{ error }}</span>{% endif %}'
    var_3 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_4 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = {var_0: var_2, var_1: var_3, var_3: var_4, var_3: var_5}
    var_7 = module_1.DictLoader(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_7.mapping == {'forms/input.html': '<input type="{{ inut_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<spn class="error">{{ error }}</span>{% endif %}', 'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}', '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}': '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'}
    var_8 = True
    var_9 = module_2.Environment(autoescape=var_8, loader=var_7)
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
    var_10 = 'name'
    var_11 = 'active'
    var_12 = 'read_only_field'
    var_13 = 100
    var_14 = module_3.String(max_length=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.String'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.allow_blank is False
    assert var_14.trim_whitespace is True
    assert var_14.max_length == 100
    assert var_14.min_length is None
    assert var_14.format is None
    assert var_14.coerce_types is True
    assert var_14.pattern is None
    assert var_14.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_15 = module_3.String(format=var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.String'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.allow_blank is False
    assert var_15.trim_whitespace is True
    assert var_15.max_length is None
    assert var_15.min_length is None
    assert var_15.format == 'forms/checkbox.html'
    assert var_15.coerce_types is True
    assert var_15.pattern is None
    assert var_15.pattern_regex is None
    var_16 = module_3.Boolean()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.coerce_types is True
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_17 = module_3.String()
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
    var_18 = {var_10: var_14, var_1: var_15, var_11: var_16, var_12: var_17}
    var_19 = module_4.Schema(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.fields).__module__}.{type(var_19.fields).__qualname__}' == 'builtins.dict'
    assert len(var_19.fields) == 4
    assert var_19.required == ['name', 'forms/checkbox.html', 'active', 'read_only_field']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_20 = 'John'
    var_21 = 'john@example.com'
    var_22 = None
    var_23 = var_19.serialize(var_22)
    var_24 = {var_10: var_20, var_5: var_21, var_11: var_8}
    var_25 = module_0.Form(env=var_9, schema=var_19, values=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_25.env).__module__}.{type(var_25.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_25.schema).__module__}.{type(var_25.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_25.values == {'name': 'John', 'active': True}
    assert var_25.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_26 = var_25.render_fields()
    assert var_26 == '<input type="" name="name" value="John" required id="name" /><input type="" name="forms/checkbox.html" value="None" required id="forms/checkbox.html" /><input type="checkbox" name="active" checked id="active" /><input type="" name="read_only_field" value="None" required id="read-only-field" />'
    assert len(var_9.cache) == 2
    var_27 = var_16.__or__(var_15)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Union'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.any_of).__module__}.{type(var_27.any_of).__qualname__}' == 'builtins.list'
    assert len(var_27.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    module_0.Jinja2Forms(directory=var_0, package=var_3)

def test_case_14():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_1.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}', 'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}', 'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}', 'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'}
    var_10 = True
    var_11 = module_2.Environment(autoescape=var_10, loader=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'jinja2.environment.Environment'
    assert var_11.block_start_string == '{%'
    assert var_11.block_end_string == '%}'
    assert var_11.variable_start_string == '{{'
    assert var_11.variable_end_string == '}}'
    assert var_11.comment_start_string == '{#'
    assert var_11.comment_end_string == '#}'
    assert var_11.line_statement_prefix is None
    assert var_11.line_comment_prefix is None
    assert var_11.trim_blocks is False
    assert var_11.lstrip_blocks is False
    assert var_11.newline_sequence == '\n'
    assert var_11.keep_trailing_newline is False
    assert var_11.optimized is True
    assert var_11.finalize is None
    assert var_11.autoescape is True
    assert f'{type(var_11.filters).__module__}.{type(var_11.filters).__qualname__}' == 'builtins.dict'
    assert len(var_11.filters) == 54
    assert f'{type(var_11.tests).__module__}.{type(var_11.tests).__qualname__}' == 'builtins.dict'
    assert len(var_11.tests) == 39
    assert f'{type(var_11.globals).__module__}.{type(var_11.globals).__qualname__}' == 'builtins.dict'
    assert len(var_11.globals) == 6
    assert f'{type(var_11.loader).__module__}.{type(var_11.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_11.cache).__module__}.{type(var_11.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_11.cache) == 0
    assert var_11.bytecode_cache is None
    assert var_11.auto_reload is True
    assert var_11.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_11.extensions == {}
    assert var_11.is_async is False
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
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'active'
    var_15 = module_3.String(format=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.String'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.allow_blank is False
    assert var_15.trim_whitespace is True
    assert var_15.max_length is None
    assert var_15.min_length is None
    assert var_15.format == 'email'
    assert var_15.coerce_types is True
    assert var_15.pattern is None
    assert var_15.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_16 = module_3.Boolean()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.coerce_types is True
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_17 = module_3.String()
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
    var_18 = {var_12: var_15, var_13: var_15, var_14: var_16, var_1: var_17}
    var_19 = module_4.Schema(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.fields).__module__}.{type(var_19.fields).__qualname__}' == 'builtins.dict'
    assert len(var_19.fields) == 4
    assert var_19.required == ['name', 'email', 'active', 'forms/checkbox.html']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_20 = 'John'
    var_21 = 'john@example.com'
    var_22 = {var_12: var_20, var_13: var_21, var_14: var_10}
    var_23 = module_0.Form(env=var_11, schema=var_19, values=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_23.env).__module__}.{type(var_23.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_23.schema).__module__}.{type(var_23.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_23.values == {'name': 'John', 'email': 'john@example.com', 'active': True}
    assert var_23.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_24 = var_23.render_fields()
    assert var_24 == '<input type="email" name="name" value="John" required id="name" /><input type="email" name="email" value="john@example.com" required id="email" /><input type="checkbox" name="active" checked id="active" /><input type="text" name="forms/checkbox.html" value="None" required id="forms/checkbox.html" />'
    assert len(var_11.cache) == 2
    var_25 = module_0.Form(env=var_11, schema=var_19)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_25.env).__module__}.{type(var_25.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_25.schema).__module__}.{type(var_25.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_25.values is None
    assert var_25.errors is None
    var_26 = var_23.__html__()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'markupsafe.Markup'
    assert len(var_26) == 300
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'
    var_27 = ''
    var_28 = 'invalid-email'
    var_29 = False
    var_30 = {var_12: var_27, var_13: var_28, var_14: var_29}
    var_31 = var_25.validate(var_30)
    assert f'{type(var_25.errors).__module__}.{type(var_25.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_25.errors) == 3
    assert var_25.data == {'name': '', 'email': 'invalid-email', 'active': False}
    assert f'{type(module_6.annotations).__module__}.{type(module_6.annotations).__qualname__}' == '__future__._Feature'
    assert module_6.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_6.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_6.annotations.compiler_flag == 16777216
    var_32 = var_25.render_fields()
    assert var_32 == '<input type="email" name="name" value="" required id="name" /><span class="error">Must not be blank.</span><input type="email" name="email" value="invalid-email" required id="email" /><span class="error">Must be a valid email format.</span><input type="checkbox" name="active"  id="active" /><input type="text" name="forms/checkbox.html" value="None" required id="forms/checkbox.html" /><span class="error">This field is required.</span>'
    var_33 = None
    var_34 = module_0.Form(env=var_11, schema=var_19, values=var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_34.env).__module__}.{type(var_34.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_34.schema).__module__}.{type(var_34.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_34.values is None
    assert var_34.errors is None
    var_35 = {}
    var_36 = var_34.validate(var_35)
    assert f'{type(var_34.errors).__module__}.{type(var_34.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_34.errors) == 4
    assert var_34.data == {}
    var_37 = var_34.render_fields()
    assert var_37 == '<input type="email" name="name" value="None" required id="name" /><span class="error">This field is required.</span><input type="email" name="email" value="None" required id="email" /><span class="error">This field is required.</span><input type="checkbox" name="active"  id="active" /><span class="error">This field is required.</span><input type="text" name="forms/checkbox.html" value="None" required id="forms/checkbox.html" /><span class="error">This field is required.</span>'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'fBrms/dnut.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required\r% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_1.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'fBrms/dnut.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required\r% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}', 'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}', 'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}', 'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'}
    var_10 = True
    var_11 = module_2.Environment(autoescape=var_10, loader=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'jinja2.environment.Environment'
    assert var_11.block_start_string == '{%'
    assert var_11.block_end_string == '%}'
    assert var_11.variable_start_string == '{{'
    assert var_11.variable_end_string == '}}'
    assert var_11.comment_start_string == '{#'
    assert var_11.comment_end_string == '#}'
    assert var_11.line_statement_prefix is None
    assert var_11.line_comment_prefix is None
    assert var_11.trim_blocks is False
    assert var_11.lstrip_blocks is False
    assert var_11.newline_sequence == '\n'
    assert var_11.keep_trailing_newline is False
    assert var_11.optimized is True
    assert var_11.finalize is None
    assert var_11.autoescape is True
    assert f'{type(var_11.filters).__module__}.{type(var_11.filters).__qualname__}' == 'builtins.dict'
    assert len(var_11.filters) == 54
    assert f'{type(var_11.tests).__module__}.{type(var_11.tests).__qualname__}' == 'builtins.dict'
    assert len(var_11.tests) == 39
    assert f'{type(var_11.globals).__module__}.{type(var_11.globals).__qualname__}' == 'builtins.dict'
    assert len(var_11.globals) == 6
    assert f'{type(var_11.loader).__module__}.{type(var_11.loader).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert f'{type(var_11.cache).__module__}.{type(var_11.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_11.cache) == 0
    assert var_11.bytecode_cache is None
    assert var_11.auto_reload is True
    assert var_11.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_11.extensions == {}
    assert var_11.is_async is False
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
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'active'
    var_15 = 'read_only_field'
    var_16 = 100
    var_17 = module_3.String(max_length=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.String'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.allow_blank is False
    assert var_17.trim_whitespace is True
    assert var_17.max_length == 100
    assert var_17.min_length is None
    assert var_17.format is None
    assert var_17.coerce_types is True
    assert var_17.pattern is None
    assert var_17.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_18 = module_3.String(format=var_13)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.String'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.allow_blank is False
    assert var_18.trim_whitespace is True
    assert var_18.max_length is None
    assert var_18.min_length is None
    assert var_18.format == 'email'
    assert var_18.coerce_types is True
    assert var_18.pattern is None
    assert var_18.pattern_regex is None
    var_19 = module_3.Boolean()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.coerce_types is True
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_20 = module_3.String()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.String'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.allow_blank is False
    assert var_20.trim_whitespace is True
    assert var_20.max_length is None
    assert var_20.min_length is None
    assert var_20.format is None
    assert var_20.coerce_types is True
    assert var_20.pattern is None
    assert var_20.pattern_regex is None
    var_21 = {var_12: var_17, var_13: var_18, var_14: var_19, var_15: var_20}
    var_22 = module_4.Schema(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.fields).__module__}.{type(var_22.fields).__qualname__}' == 'builtins.dict'
    assert len(var_22.fields) == 4
    assert var_22.required == ['name', 'email', 'active', 'read_only_field']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_23 = 'Joh'
    var_24 = 'ju\nn@etample.com'
    var_25 = {var_12: var_23, var_13: var_24, var_14: var_10}
    var_26 = module_0.Form(env=var_11, schema=var_22, values=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_26.env).__module__}.{type(var_26.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_26.schema).__module__}.{type(var_26.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_26.values == {'name': 'Joh', 'email': 'ju\nn@etample.com', 'active': True}
    assert var_26.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_26.__str__()

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = '<input type="{{ inut_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<spn class="error">{{ error }}</span>{% endif %}'
    var_4 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = {var_0: var_3, var_1: var_4, var_2: var_5, var_2: var_6}
    var_8 = module_1.DictLoader(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_8.mapping == {'forms/input.html': '<input type="{{ inut_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<spn class="error">{{ error }}</span>{% endif %}', 'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}', 'forms/select.html': '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'}
    var_9 = True
    var_10 = module_2.Environment(autoescape=var_9, loader=var_8)
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
    assert var_10.autoescape is True
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
    var_11 = 'active'
    var_12 = module_3.String(format=var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.allow_blank is False
    assert var_12.trim_whitespace is True
    assert var_12.max_length is None
    assert var_12.min_length is None
    assert var_12.format == 'forms/checkbox.html'
    assert var_12.coerce_types is True
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_13 = module_3.Boolean()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.coerce_types is True
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_14 = module_3.String(allow_blank=var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.String'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.default == ''
    assert var_14.allow_blank is True
    assert var_14.trim_whitespace is True
    assert var_14.max_length is None
    assert var_14.min_length is None
    assert var_14.format is None
    assert var_14.coerce_types is True
    assert var_14.pattern is None
    assert var_14.pattern_regex is None
    var_15 = {var_2: var_14, var_1: var_12, var_11: var_13, var_6: var_14}
    var_16 = module_4.Schema(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert f'{type(var_16.fields).__module__}.{type(var_16.fields).__qualname__}' == 'builtins.dict'
    assert len(var_16.fields) == 4
    assert var_16.required == ['forms/checkbox.html', 'active']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_17 = module_0.Form(env=var_10, schema=var_16, values=var_1)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_17.env).__module__}.{type(var_17.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_17.schema).__module__}.{type(var_17.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_17.values == {}
    assert var_17.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_18 = var_17.render_fields()
    assert var_18 == '<input type="" name="forms/select.html" value="None"  id="forms/select.html" /><input type="" name="forms/checkbox.html" value="None" required id="forms/checkbox.html" /><input type="checkbox" name="active"  id="active" /><input type="" name="&lt;textarea name=&#34;{{ field_name }}&#34; id=&#34;{{ field_id }}&#34; {% if required %}required{% endif %}&gt;{{ value }}&lt;/textarea&gt;{% if error %}&lt;span class=&#34;error&#34;&gt;{{ error }}&lt;/span&gt;{% endif %}" value="None"  id="&lt;textarea name=&#34;{{ field-name }}&#34; id=&#34;{{ field-id }}&#34; {% if required %}required{% endif %}&gt;{{ value }}&lt;/textarea&gt;{% if error %}&lt;span class=&#34;error&#34;&gt;{{ error }}&lt;/span&gt;{% endif %}" />'
    assert len(var_10.cache) == 2
    module_0.Jinja2Forms(directory=var_0, package=var_4)

def test_case_17():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>'
    var_2 = {var_0: var_1}
    var_3 = module_1.DictLoader(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_3.mapping == {'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>'}
    var_4 = module_2.Environment(loader=var_3)
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
    var_5 = 'name'
    var_6 = 'id'
    var_7 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = True
    var_9 = module_3.String()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length is None
    assert var_9.min_length is None
    assert var_9.format is None
    assert var_9.coerce_types is True
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    var_10 = {var_5: var_7, var_6: var_9}
    var_11 = module_4.Schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.fields).__module__}.{type(var_11.fields).__qualname__}' == 'builtins.dict'
    assert len(var_11.fields) == 2
    assert var_11.required == ['name', 'id']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_12 = 'John'
    var_13 = '123'
    var_14 = {var_5: var_12, var_6: var_13}
    var_15 = module_0.Form(env=var_4, schema=var_11, values=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_15.env).__module__}.{type(var_15.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_15.schema).__module__}.{type(var_15.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_15.values == {'name': 'John', 'id': '123'}
    assert var_15.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_16 = {var_5: var_12}
    var_17 = var_15.validate(var_16)
    assert var_15.values is None
    assert f'{type(var_15.errors).__module__}.{type(var_15.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15.errors) == 1
    assert var_15.data == {'name': 'John'}
    var_18 = var_15.template_for_field(var_11)
    assert var_18 == 'forms/input.html'