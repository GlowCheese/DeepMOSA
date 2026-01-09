# Check out: https://github.com/GlowCheese/deepmosa
import jinja2.environment as module_5
import jinja2.loaders as module_4
import markupsafe as module_3
import pytest
import typesystem.fields as module_1
import typesystem.forms as module_0
import typesystem.schemas as module_2


def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'gu.Um;'
    module_0.Jinja2Forms(package=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1.String()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.String'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format is None
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = {var_0: var_0}
    var_2 = module_2.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.fields).__module__}.{type(var_2.fields).__qualname__}' == 'builtins.dict'
    assert len(var_2.fields) == 1
    assert f'{type(var_2.required).__module__}.{type(var_2.required).__qualname__}' == 'builtins.list'
    assert len(var_2.required) == 1
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = None
    var_4 = module_0.Form(env=var_3, schema=var_2, values=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert var_4.env is None
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_4.render_fields()

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'gu.Um;'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_1.load_template_env(directory=var_1, package=var_1)

def test_case_4():
    var_0 = 'gu.Um;'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_2 = None
    with pytest.raises(AssertionError):
        var_1.load_template_env(directory=var_2, package=var_2)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'gu.Um;'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_1.create_form(var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
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
    var_4 = None
    var_5 = module_0.Form(env=var_4, schema=var_3, values=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert var_5.env is None
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values is None
    assert var_5.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_6 = var_5.validate()
    assert f'{type(var_5.errors).__module__}.{type(var_5.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.errors) == 1
    assert var_5.data is None
    var_5.render_fields()

@pytest.mark.xfail(strict=True)
def test_case_7():
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
    var_4 = None
    var_5 = module_0.Form(env=var_4, schema=var_3, values=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert var_5.env is None
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values is None
    assert var_5.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5.render_fields()

def test_case_8():
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
    var_4 = None
    var_5 = module_0.Form(env=var_4, schema=var_3, values=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Form'
    assert var_5.env is None
    assert f'{type(var_5.schema).__module__}.{type(var_5.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.values is None
    assert var_5.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_6 = var_5.validate(var_2)
    assert f'{type(var_5.errors).__module__}.{type(var_5.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.errors) == 1
    assert f'{type(var_5.data).__module__}.{type(var_5.data).__qualname__}' == 'builtins.dict'
    assert len(var_5.data) == 1
    with pytest.raises(AssertionError):
        var_5.validate(var_4)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_1.String()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.String'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format is None
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
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
    var_3 = None
    var_4 = module_0.Form(env=var_3, schema=var_2, values=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert var_4.env is None
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = var_4.validate()
    assert f'{type(var_4.errors).__module__}.{type(var_4.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.errors) == 1
    assert var_4.data is None
    var_6 = var_4.render_fields()
    assert var_6 == ''
    var_1.pop_assign_tracking(var_3)

@pytest.mark.xfail(strict=True)
def test_case_10():
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
    var_4 = module_0.Form(env=var_0, schema=var_3, values=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert var_4.env == 'name'
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values == {}
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_5.env).__module__}.{type(var_5.env).__qualname__}' == 'jinja2.environment.Environment'
    var_4.render_fields()

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_1.String()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.String'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format is None
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
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
    var_3 = None
    var_4 = module_0.Form(env=var_3, schema=var_2, values=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert var_4.env is None
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = var_4.validate()
    assert f'{type(var_4.errors).__module__}.{type(var_4.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.errors) == 1
    assert var_4.data is None
    var_6 = var_4.render_fields()
    assert var_6 == ''
    var_7 = var_4.__html__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'markupsafe.Markup'
    assert len(var_7) == 0
    assert f'{type(module_3.Markup.escape).__module__}.{type(module_3.Markup.escape).__qualname__}' == 'builtins.method'
    var_1.pop_assign_tracking(var_3)

@pytest.mark.xfail(strict=True)
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
    var_4 = var_3.serialize(var_1)
    var_5 = None
    var_6 = module_0.Form(env=var_5, schema=var_3, values=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.forms.Form'
    assert var_6.env is None
    assert f'{type(var_6.schema).__module__}.{type(var_6.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.values is None
    assert var_6.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_7 = var_6.validate(var_2)
    assert f'{type(var_6.errors).__module__}.{type(var_6.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.errors) == 1
    assert f'{type(var_6.data).__module__}.{type(var_6.data).__qualname__}' == 'builtins.dict'
    assert len(var_6.data) == 1
    var_6.__str__()

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = module_1.String()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.String'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format is None
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
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
    var_3 = None
    var_4 = module_0.Form(env=var_3, schema=var_2, values=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.forms.Form'
    assert var_4.env is None
    assert f'{type(var_4.schema).__module__}.{type(var_4.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.values is None
    assert var_4.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_5 = var_4.validate()
    assert f'{type(var_4.errors).__module__}.{type(var_4.errors).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.errors) == 1
    assert var_4.data is None
    var_6 = var_4.render_fields()
    assert var_6 == ''
    var_7 = var_4.template_for_field(var_5)
    assert var_7 == 'forms/input.html'
    var_8 = var_4.__html__()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'markupsafe.Markup'
    assert len(var_8) == 0
    assert f'{type(module_3.Markup.escape).__module__}.{type(module_3.Markup.escape).__qualname__}' == 'builtins.method'
    var_1.pop_assign_tracking(var_3)

def test_case_14():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" />'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} />'
    var_7 = '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_4.DictLoader(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'jinja2.loaders.DictLoader'
    assert var_9.mapping == {'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" />', 'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>', 'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} />', 'forms/select.html': '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'}
    var_10 = module_5.Environment(loader=var_9)
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
    assert module_5.BLOCK_END_STRING == '%}'
    assert module_5.BLOCK_START_STRING == '{%'
    assert module_5.COMMENT_END_STRING == '#}'
    assert module_5.COMMENT_START_STRING == '{#'
    assert f'{type(module_5.DEFAULT_FILTERS).__module__}.{type(module_5.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_FILTERS) == 54
    assert f'{type(module_5.DEFAULT_NAMESPACE).__module__}.{type(module_5.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_NAMESPACE) == 6
    assert module_5.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_5.DEFAULT_TESTS).__module__}.{type(module_5.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_TESTS) == 39
    assert module_5.KEEP_TRAILING_NEWLINE is False
    assert module_5.LINE_COMMENT_PREFIX is None
    assert module_5.LINE_STATEMENT_PREFIX is None
    assert module_5.LSTRIP_BLOCKS is False
    assert module_5.NEWLINE_SEQUENCE == '\n'
    assert module_5.TRIM_BLOCKS is False
    assert module_5.VARIABLE_END_STRING == '}}'
    assert module_5.VARIABLE_START_STRING == '{{'
    assert f'{type(module_5.missing).__module__}.{type(module_5.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_5.Environment.sandboxed is False
    assert module_5.Environment.overlayed is False
    assert module_5.Environment.linked_to is None
    assert module_5.Environment.shared is False
    assert f'{type(module_5.Environment.lexer).__module__}.{type(module_5.Environment.lexer).__qualname__}' == 'builtins.property'
    var_11 = 'name'
    var_12 = 'description'
    var_13 = 'active'
    var_14 = 'choice'
    var_15 = 'Name'
    var_16 = module_1.String()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.String'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.allow_blank is False
    assert var_16.trim_whitespace is True
    assert var_16.max_length is None
    assert var_16.min_length is None
    assert var_16.format is None
    assert var_16.coerce_types is True
    assert var_16.pattern is None
    assert var_16.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_17 = 'Description'
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
    var_20 = 'Active'
    var_21 = module_1.Boolean()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_22 = 'Choice'
    var_23 = 'option1'
    var_24 = 'Option 1'
    var_25 = (var_23, var_24)
    var_26 = 'option2'
    var_27 = 'Option 2'
    var_28 = (var_26, var_27)
    var_29 = [var_25, var_28]
    var_30 = module_1.Choice(choices=var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Choice'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.choices == [('option1', 'Option 1'), ('option2', 'Option 2')]
    assert var_30.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_31 = {var_11: var_16, var_12: var_19, var_13: var_21, var_14: var_30}
    var_32 = module_2.Schema(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.fields).__module__}.{type(var_32.fields).__qualname__}' == 'builtins.dict'
    assert len(var_32.fields) == 4
    assert var_32.required == ['name', 'description', 'active', 'choice']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_33 = module_0.Form(env=var_10, schema=var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.forms.Form'
    assert f'{type(var_33.env).__module__}.{type(var_33.env).__qualname__}' == 'jinja2.environment.Environment'
    assert f'{type(var_33.schema).__module__}.{type(var_33.schema).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_33.values is None
    assert var_33.errors is None
    assert module_0.Form.FORMAT_TO_INPUTTYPE == {'color': 'color', 'datetime': 'datetime-local', 'date': 'date', 'email': 'email', 'hidden': 'hidden', 'month': 'month', 'number': 'number', 'password': 'password', 'range': 'range', 'search': 'search', 'tel': 'tel', 'text': 'text', 'time': 'time', 'url': 'url', 'week': 'week'}
    assert f'{type(module_0.Form.is_valid).__module__}.{type(module_0.Form.is_valid).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Form.validated_data).__module__}.{type(module_0.Form.validated_data).__qualname__}' == 'builtins.property'
    var_34 = var_32.fields[var_11]
    var_35 = 'John'
    var_36 = var_33.render_field(field_name=var_11, field=var_34, value=var_35)
    assert var_36 == '<input type="text" name="name" value="John" />'
    assert len(var_10.cache) == 1
    var_37 = var_32.fields[var_12]
    var_38 = 'Some description'
    var_39 = var_33.render_field(field_name=var_12, field=var_37, value=var_38)
    assert var_39 == '<textarea name="description">Some description</textarea>'
    assert len(var_10.cache) == 2
    var_40 = var_32.fields[var_13]
    var_41 = True
    var_42 = var_33.render_field(field_name=var_13, field=var_40, value=var_41)
    assert var_42 == '<input type="checkbox" name="active" checked />'
    assert len(var_10.cache) == 3
    var_43 = var_32.fields[var_14]
    var_44 = var_33.render_field(field_name=var_14, field=var_43, value=var_23)
    assert var_44 == '<select name="choice"><option value="option1">option1</option></select>'
    assert len(var_10.cache) == 4
    var_45 = var_32.fields[var_11]
    var_46 = 'Invalid name'
    var_47 = var_33.render_field(field_name=var_11, field=var_45, value=var_35, error=var_46)
    assert var_47 == '<input type="text" name="name" value="John" />'
    var_48 = var_32.fields[var_11]
    var_49 = var_33.render_field(field_name=var_11, field=var_48)
    assert var_49 == '<input type="text" name="name" value="None" />'
    var_50 = 'Password'
    var_51 = 'password'
    var_52 = module_1.String(format=var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.String'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert var_52.allow_blank is False
    assert var_52.trim_whitespace is True
    assert var_52.max_length is None
    assert var_52.min_length is None
    assert var_52.format == 'password'
    assert var_52.coerce_types is True
    assert var_52.pattern is None
    assert var_52.pattern_regex is None
    var_53 = 'secret'
    var_54 = var_33.render_field(field_name=var_51, field=var_52, value=var_53)
    assert var_54 == '<input type="password" name="password" value="" />'
    var_55 = 'Email'
    var_56 = 'email'
    var_57 = module_1.String(format=var_56)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.String'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert var_57.allow_blank is False
    assert var_57.trim_whitespace is True
    assert var_57.max_length is None
    assert var_57.min_length is None
    assert var_57.format == 'email'
    assert var_57.coerce_types is True
    assert var_57.pattern is None
    assert var_57.pattern_regex is None
    var_58 = 'test@example.com'
    var_59 = var_33.render_field(field_name=var_56, field=var_57, value=var_58)
    assert var_59 == '<input type="email" name="email" value="test@example.com" />'
    var_60 = 'Required'
    var_61 = False
    var_62 = module_1.String()
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.fields.String'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert var_62.allow_blank is False
    assert var_62.trim_whitespace is True
    assert var_62.max_length is None
    assert var_62.min_length is None
    assert var_62.format is None
    assert var_62.coerce_types is True
    assert var_62.pattern is None
    assert var_62.pattern_regex is None
    var_63 = 'required'
    var_64 = var_33.render_field(field_name=var_63, field=var_62)
    assert var_64 == '<input type="text" name="required" value="None" />'
    var_65 = 'Optional'
    var_66 = module_1.String()
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.fields.String'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert var_66.allow_blank is False
    assert var_66.trim_whitespace is True
    assert var_66.max_length is None
    assert var_66.min_length is None
    assert var_66.format is None
    assert var_66.coerce_types is True
    assert var_66.pattern is None
    assert var_66.pattern_regex is None
    var_67 = 'optional'
    var_68 = var_33.render_field(field_name=var_67, field=var_66)
    assert var_68 == '<input type="text" name="optional" value="None" />'
    var_69 = 'Default'
    var_70 = 'default_value'
    var_71 = module_1.String()
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.String'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.allow_blank is False
    assert var_71.trim_whitespace is True
    assert var_71.max_length is None
    assert var_71.min_length is None
    assert var_71.format is None
    assert var_71.coerce_types is True
    assert var_71.pattern is None
    assert var_71.pattern_regex is None
    var_72 = 'default'
    var_73 = var_33.render_field(field_name=var_72, field=var_71)
    assert var_73 == '<input type="text" name="default" value="None" />'
    var_74 = 'Blank'
    var_75 = module_1.String(allow_blank=var_41)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.fields.String'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert var_75.default == ''
    assert var_75.allow_blank is True
    assert var_75.trim_whitespace is True
    assert var_75.max_length is None
    assert var_75.min_length is None
    assert var_75.format is None
    assert var_75.coerce_types is True
    assert var_75.pattern is None
    assert var_75.pattern_regex is None
    var_76 = 'blank'
    var_77 = var_33.render_field(field_name=var_76, field=var_75)
    assert var_77 == '<input type="text" name="blank" value="None" />'
    var_78 = 'NullableBlank'
    var_79 = module_1.String(allow_blank=var_41)
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.fields.String'
    assert var_79.title == ''
    assert var_79.description == ''
    assert var_79.allow_null is False
    assert var_79.read_only is False
    assert var_79.default == ''
    assert var_79.allow_blank is True
    assert var_79.trim_whitespace is True
    assert var_79.max_length is None
    assert var_79.min_length is None
    assert var_79.format is None
    assert var_79.coerce_types is True
    assert var_79.pattern is None
    assert var_79.pattern_regex is None
    var_80 = 'nullable_blank'
    var_81 = var_33.render_field(field_name=var_80, field=var_79)
    assert var_81 == '<input type="text" name="nullable_blank" value="None" />'
    var_82 = 'Strict'
    var_83 = module_1.String(allow_blank=var_61)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.fields.String'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert var_83.allow_blank is False
    assert var_83.trim_whitespace is True
    assert var_83.max_length is None
    assert var_83.min_length is None
    assert var_83.format is None
    assert var_83.coerce_types is True
    assert var_83.pattern is None
    assert var_83.pattern_regex is None
    var_84 = 'strict'
    var_85 = var_33.render_field(field_name=var_84, field=var_83)
    assert var_85 == '<input type="text" name="strict" value="None" />'
    var_86 = 'Custom'
    var_87 = 'custom'
    var_88 = module_1.String(format=var_87)
    assert f'{type(var_88).__module__}.{type(var_88).__qualname__}' == 'typesystem.fields.String'
    assert var_88.title == ''
    assert var_88.description == ''
    assert var_88.allow_null is False
    assert var_88.read_only is False
    assert var_88.allow_blank is False
    assert var_88.trim_whitespace is True
    assert var_88.max_length is None
    assert var_88.min_length is None
    assert var_88.format == 'custom'
    assert var_88.coerce_types is True
    assert var_88.pattern is None
    assert var_88.pattern_regex is None
    var_89 = var_33.render_field(field_name=var_87, field=var_88)
    assert var_89 == '<input type="text" name="custom" value="None" />'
    var_90 = 'Date'
    var_91 = 'date'
    var_92 = module_1.String(format=var_91)
    assert f'{type(var_92).__module__}.{type(var_92).__qualname__}' == 'typesystem.fields.String'
    assert var_92.title == ''
    assert var_92.description == ''
    assert var_92.allow_null is False
    assert var_92.read_only is False
    assert var_92.allow_blank is False
    assert var_92.trim_whitespace is True
    assert var_92.max_length is None
    assert var_92.min_length is None
    assert var_92.format == 'date'
    assert var_92.coerce_types is True
    assert var_92.pattern is None
    assert var_92.pattern_regex is None
    var_93 = '2023-01-01'
    var_94 = var_33.render_field(field_name=var_91, field=var_92, value=var_93)
    assert var_94 == '<input type="date" name="date" value="2023-01-01" />'
    var_95 = 'DateTime'
    var_96 = 'datetime'
    var_97 = module_1.String(format=var_96)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.fields.String'
    assert var_97.title == ''
    assert var_97.description == ''
    assert var_97.allow_null is False
    assert var_97.read_only is False
    assert var_97.allow_blank is False
    assert var_97.trim_whitespace is True
    assert var_97.max_length is None
    assert var_97.min_length is None
    assert var_97.format == 'datetime'
    assert var_97.coerce_types is True
    assert var_97.pattern is None
    assert var_97.pattern_regex is None
    var_98 = '2023-01-01T12:00'
    var_99 = var_33.render_field(field_name=var_96, field=var_97, value=var_98)
    assert var_99 == '<input type="datetime-local" name="datetime" value="2023-01-01T12:00" />'
    var_100 = 'NonString'
    var_101 = module_1.Boolean()
    assert f'{type(var_101).__module__}.{type(var_101).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_101.title == ''
    assert var_101.description == ''
    assert var_101.allow_null is False
    assert var_101.read_only is False
    assert var_101.coerce_types is True
    var_102 = 'non_string'
    var_103 = var_33.render_field(field_name=var_102, field=var_101)
    assert var_103 == '<input type="checkbox" name="non_string"  />'
    var_104 = 'NonStringWithFormat'
    var_105 = module_1.Boolean()
    assert f'{type(var_105).__module__}.{type(var_105).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_105.title == ''
    assert var_105.description == ''
    assert var_105.allow_null is False
    assert var_105.read_only is False
    assert var_105.coerce_types is True
    var_106 = 'non_string_with_format'
    var_107 = var_33.render_field(field_name=var_106, field=var_105)
    assert var_107 == '<input type="checkbox" name="non_string_with_format"  />'
    var_108 = 'NoFormat'
    var_109 = None
    var_110 = module_1.String(format=var_109)
    assert f'{type(var_110).__module__}.{type(var_110).__qualname__}' == 'typesystem.fields.String'
    assert var_110.title == ''
    assert var_110.description == ''
    assert var_110.allow_null is False
    assert var_110.read_only is False
    assert var_110.allow_blank is False
    assert var_110.trim_whitespace is True
    assert var_110.max_length is None
    assert var_110.min_length is None
    assert var_110.format is None
    assert var_110.coerce_types is True
    assert var_110.pattern is None
    assert var_110.pattern_regex is None
    var_111 = 'no_format'
    var_112 = var_33.render_field(field_name=var_111, field=var_110)
    assert var_112 == '<input type="text" name="no_format" value="None" />'
    var_113 = 'EmptyFormat'
    var_114 = ''
    var_115 = module_1.String(format=var_114)
    assert f'{type(var_115).__module__}.{type(var_115).__qualname__}' == 'typesystem.fields.String'
    assert var_115.title == ''
    assert var_115.description == ''
    assert var_115.allow_null is False
    assert var_115.read_only is False
    assert var_115.allow_blank is False
    assert var_115.trim_whitespace is True
    assert var_115.max_length is None
    assert var_115.min_length is None
    assert var_115.format == ''
    assert var_115.coerce_types is True
    assert var_115.pattern is None
    assert var_115.pattern_regex is None
    var_116 = 'empty_format'
    var_117 = var_33.render_field(field_name=var_116, field=var_115)
    assert var_117 == '<input type="text" name="empty_format" value="None" />'
    var_118 = 'WhitespaceFormat'
    var_119 = ' '
    var_120 = module_1.String(format=var_119)
    assert f'{type(var_120).__module__}.{type(var_120).__qualname__}' == 'typesystem.fields.String'
    assert var_120.title == ''
    assert var_120.description == ''
    assert var_120.allow_null is False
    assert var_120.read_only is False
    assert var_120.allow_blank is False
    assert var_120.trim_whitespace is True
    assert var_120.max_length is None
    assert var_120.min_length is None
    assert var_120.format == ' '
    assert var_120.coerce_types is True
    assert var_120.pattern is None
    assert var_120.pattern_regex is None
    var_121 = 'whitespace_format'
    var_122 = var_33.render_field(field_name=var_121, field=var_120)
    assert var_122 == '<input type="text" name="whitespace_format" value="None" />'
    var_123 = 'NumberFormat'
    var_124 = 123
    with pytest.raises(AssertionError):
        module_1.String(format=var_124)