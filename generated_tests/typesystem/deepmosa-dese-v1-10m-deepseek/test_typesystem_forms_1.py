# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0
import typesystem.fields as module_1
import jinja2.filters as module_2
import jinja2.loaders as module_3
import jinja2.environment as module_4
import typesystem.schemas as module_5
import markupsafe as module_6

def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'some_packalge'
    module_0.Jinja2Forms(directory=var_0, package=var_0)

def test_case_2():
    var_0 = "*j\t_'mh5B"
    var_1 = None
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_2.env).__module__}.{type(var_2.env).__qualname__}' == 'jinja2.environment.Environment'

def test_case_3():
    var_0 = module_1.Boolean()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_1 = module_0.Form(env=var_0, schema=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_1.schema).__module__}.{type(var_1.schema).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_1.values is None
    assert var_1.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_2 = var_1.template_for_field(var_0)
    assert var_2 == 'forms/checkbox.html'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '@|7&fl1Zoa6= -IPc'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_2 = module_2.do_striptags(var_0)
    assert var_2 == '@|7&fl1Zoa6= -IPc'
    assert f'{type(module_2.F).__module__}.{type(module_2.F).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.K).__module__}.{type(module_2.K).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.V).__module__}.{type(module_2.V).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.FILTERS).__module__}.{type(module_2.FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FILTERS) == 54
    var_3 = var_2.__str__()
    assert var_3 == '@|7&fl1Zoa6= -IPc'
    var_4 = None
    var_5 = var_2.__str__()
    assert var_5 == '@|7&fl1Zoa6= -IPc'
    var_1.create_form(var_4)

def test_case_5():
    var_0 = 'cT{Sa/m18GPK|W'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        var_1.load_template_env()

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

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'forms/input.html'
    var_1 = {var_0: var_0}
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
    var_4 = module_1.Field(title=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == 'forms/input.html'
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_5 = {var_0: var_4}
    var_6 = module_5.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['forms/input.html']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_3, schema=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_7.validate(var_1)

def test_case_9():
    var_0 = 'forms/input.html'
    var_1 = {var_0: var_0}
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
    var_4 = module_1.Field()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_5 = {var_0: var_4}
    var_6 = module_5.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['forms/input.html']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_3, schema=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = var_7.__html__()
    assert len(var_3.cache) == 1
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'markupsafe.Markup'
    assert len(var_8) == 16
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'

def test_case_10():
    var_0 = {}
    var_1 = module_3.DictLoader(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_1.mapping == {}
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
    var_3 = {}
    var_4 = module_5.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.fields == {}
    assert var_4.required == []
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_0.Form(env=var_2, schema=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_5.env).__module__}.{type(var_5.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values is None
    assert var_5.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_6 = var_5.__html__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'markupsafe.Markup'
    assert len(var_6) == 0
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'
    var_7 = str(var_6)
    assert var_7 == ''

def test_case_11():
    var_0 = 'forms/input.html'
    var_1 = {var_0: var_0}
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
    var_4 = module_1.Field(title=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == 'forms/input.html'
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_5 = {var_0: var_4}
    var_6 = module_5.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['forms/input.html']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_3, schema=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = 'test@example.com'
    var_9 = {var_8: var_8}
    var_10 = var_7.validate(var_9)
    assert f'{type(var_7.errors).__module__}.{type(var_7.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.errors) == 1
    assert var_7.data == {'test@example.com': 'test@example.com'}
    var_11 = var_7.__html__()
    assert len(var_3.cache) == 1
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'markupsafe.Markup'
    assert len(var_11) == 16
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'

def test_case_12():
    var_0 = 'forms/input.html'
    var_1 = {var_0: var_0}
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
    var_4 = True
    var_5 = module_1.Field(allow_null=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Field'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_6 = {var_0: var_5}
    var_7 = module_5.Schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 1
    assert var_7.required == []
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = module_0.Form(env=var_3, schema=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_8.env).__module__}.{type(var_8.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_8.schema).__module__}.{type(var_8.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.values is None
    assert var_8.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_9 = var_8.__html__()
    assert len(var_3.cache) == 1
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'markupsafe.Markup'
    assert len(var_9) == 16
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'
    var_10 = str(var_9)

def test_case_13():
    var_0 = 'forms/input.html'
    var_1 = '<input type={{ input_type }}"name="{{ field_name }}" value="{{ value }}">'
    var_2 = {var_0: var_1}
    var_3 = module_3.DictLoader(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_3.mapping == {'forms/input.html': '<input type={{ input_type }}"name="{{ field_name }}" value="{{ value }}">'}
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
    var_8 = module_5.Schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 1
    assert var_8.required == ['password']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_9 = module_0.Form(env=var_4, schema=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_9.env).__module__}.{type(var_9.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_9.schema).__module__}.{type(var_9.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.values is None
    assert var_9.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_10 = var_9.__html__()
    assert len(var_4.cache) == 1
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'markupsafe.Markup'
    assert len(var_10) == 46
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'
    var_11 = str(var_10)

def test_case_14():
    var_0 = 'forms/input.html'
    var_1 = {var_0: var_0}
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
    var_4 = module_1.Field(title=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == 'forms/input.html'
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_5 = {var_0: var_4}
    var_6 = module_5.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['forms/input.html']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_3, schema=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = var_7.__html__()
    assert len(var_3.cache) == 1
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'markupsafe.Markup'
    assert len(var_8) == 16
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'

def test_case_15():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_3.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">', 'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>', 'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>', 'forms/select.html': '<select name="{{ field_name }}"></select>'}
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
    var_11 = 'text'
    var_12 = 'bool'
    var_13 = 'choice'
    var_14 = 'regular'
    var_15 = module_1.String(format=var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.String'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.allow_blank is False
    assert var_15.trim_whitespace is True
    assert var_15.max_length is None
    assert var_15.min_length is None
    assert var_15.format == 'text'
    assert var_15.coerce_types is True
    assert var_15.pattern is None
    assert var_15.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_16 = module_1.Boolean()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_17 = 'a'
    var_18 = 'A'
    var_19 = (var_17, var_18)
    var_20 = [var_19]
    var_21 = module_1.Choice(choices=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Choice'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.choices == [('a', 'A')]
    assert var_21.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_22 = module_1.Field()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Field'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert module_1.Field.errors == {}
    var_23 = {var_11: var_15, var_12: var_16, var_13: var_21, var_14: var_22}
    var_24 = module_5.Schema(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.fields).__module__}.{type(var_24.fields).__qualname__}' == 'builtins.dict'
    assert len(var_24.fields) == 4
    assert var_24.required == ['text', 'bool', 'choice', 'regular']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_25 = module_0.Form(env=var_10, schema=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_25.env).__module__}.{type(var_25.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_25.schema).__module__}.{type(var_25.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_25.values is None
    assert var_25.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_26 = var_25.__html__()
    assert len(var_10.cache) == 4
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'markupsafe.Markup'
    assert len(var_26) == 151
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'
    var_27 = str(var_26)
    var_28 = str(var_26)
    var_29 = str(var_26)
    var_30 = str(var_26)

def test_case_16():
    var_0 = 'forms/input.html'
    var_1 = {var_0: var_0}
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
    var_4 = module_1.Field(title=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == 'forms/input.html'
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_5 = {var_0: var_4}
    var_6 = module_5.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['forms/input.html']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_3, schema=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = var_7.__str__()
    assert var_8 == 'forms/input.html'
    assert len(var_3.cache) == 1
    var_9 = 'test@example.com'
    var_10 = {var_9: var_9}
    var_11 = var_7.validate(var_10)
    assert f'{type(var_7.errors).__module__}.{type(var_7.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.errors) == 1
    assert var_7.data == {'test@example.com': 'test@example.com'}
    var_12 = var_7.__html__()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'markupsafe.Markup'
    assert len(var_12) == 16
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'

def test_case_17():
    var_0 = 'forms/input.html'
    var_1 = {var_0: var_0}
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
    var_4 = module_1.Field(title=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == 'forms/input.html'
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_5 = {var_0: var_4}
    var_6 = module_5.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['forms/input.html']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_0.Form(env=var_3, schema=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_7.env).__module__}.{type(var_7.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_7.schema).__module__}.{type(var_7.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.values is None
    assert var_7.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_8 = var_7.__str__()
    assert var_8 == 'forms/input.html'
    assert len(var_3.cache) == 1
    var_9 = 'test@example.com'
    var_10 = {var_9: var_9}
    var_11 = var_7.validate(var_10)
    assert f'{type(var_7.errors).__module__}.{type(var_7.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.errors) == 1
    assert var_7.data == {'test@example.com': 'test@example.com'}
    with pytest.raises(AssertionError):
        var_7.validate()

def test_case_18():
    var_0 = 'fo9rms/input.htul'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_2 = {var_0: var_1}
    var_3 = module_3.DictLoader(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_3.mapping == {'fo9rms/input.htul': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'}
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
    var_5 = 'id'
    var_6 = True
    var_7 = module_1.Field(read_only=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Field'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_8 = {var_5: var_7}
    var_9 = module_5.Schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.fields).__module__}.{type(var_9.fields).__qualname__}' == 'builtins.dict'
    assert len(var_9.fields) == 1
    assert var_9.required == []
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_10 = module_0.Form(env=var_4, schema=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_10.env).__module__}.{type(var_10.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_10.schema).__module__}.{type(var_10.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_10.values is None
    assert var_10.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_11 = var_10.__html__()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'markupsafe.Markup'
    assert len(var_11) == 0
    assert f'{type(module_6.Markup.escape).__module__}.{type(module_6.Markup.escape).__qualname__}' == 'builtins.method'
    var_12 = str(var_11)
    assert var_12 == ''