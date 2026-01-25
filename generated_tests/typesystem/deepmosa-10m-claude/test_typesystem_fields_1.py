# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.fields as module_0
import platform as module_1
import enum as module_2
import uuid as module_3
import decimal as module_4
import _collections as module_5
import typesystem.base as module_6
import re as module_7

def test_case_0():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Text'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'text'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7

def test_case_1():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1

def test_case_2():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = var_0.get_default_value()

def test_case_3():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = var_0.__or__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_0.Password()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Password'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'password'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = None
    var_0.validate(var_1)

def test_case_5():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Password(**var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Password'
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_3 = var_2.serialize(var_0)
    var_4 = None
    var_5 = True
    var_6 = module_0.Array(var_4, var_5, max_items=var_4, unique_items=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.items is None
    assert var_6.additional_items is True
    assert var_6.min_items is None
    assert var_6.max_items is None
    assert var_6.unique_items is True
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_7 = var_6.validate_or_error(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1

def test_case_6():
    var_0 = module_0.Integer()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Integer'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.minimum is None
    assert var_0.maximum is None
    assert var_0.exclusive_minimum is None
    assert var_0.exclusive_maximum is None
    assert var_0.multiple_of is None
    assert var_0.precision is None
    assert var_0.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7

def test_case_7():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Choice'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.choices == []
    assert var_0.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}

def test_case_8():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}

def test_case_9():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_1.freedesktop_os_release()
    with pytest.raises(AssertionError):
        module_0.Object(pattern_properties=var_1, additional_properties=var_1, min_properties=var_0)

def test_case_10():
    var_0 = []
    var_1 = module_0.Union(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.any_of == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_0.Const(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Const'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.const is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_1.__or__(var_0)

def test_case_12():
    var_0 = module_0.Boolean()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_1.release()
    assert var_1 == '6.17.9-76061709-generic'
    var_2 = module_0.Time()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Time'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'time'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_3 = module_0.Const(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Const'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.const is None
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_2.validate(var_0)

def test_case_14():
    var_0 = module_0.DateTime()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'datetime'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7

def test_case_15():
    var_0 = module_0.Email()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Email'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'email'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7

def test_case_16():
    var_0 = module_0.IPAddress()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'ipaddress'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7

def test_case_17():
    var_0 = module_0.URL()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.URL'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'url'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7

def test_case_18():
    var_0 = None
    var_1 = True
    var_2 = module_0.String(allow_blank=var_1, format=var_0, coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.default == ''
    assert var_2.allow_blank is True
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = -2058.7786
    module_0.Object(required=var_0)

def test_case_20():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = True
    var_2 = ',$U'
    var_3 = module_0.Integer(maximum=var_1, exclusive_minimum=var_1, precision=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is True
    assert var_3.exclusive_minimum is True
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision == ',$U'
    assert var_3.coerce_types is True
    var_4 = var_3.validate_or_error(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_21():
    var_0 = None
    var_1 = module_0.Const(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Const'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.const is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_2 = module_0.Email()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Email'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'email'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = 'E+RzO'
    var_5 = True
    var_6 = var_3.__iter__()
    var_7 = module_0.Decimal(maximum=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.minimum is None
    assert var_7.maximum is None
    assert var_7.exclusive_minimum is None
    assert var_7.exclusive_maximum is None
    assert var_7.multiple_of is None
    assert var_7.precision is None
    assert var_7.coerce_types is True
    var_8 = var_7.get_default_value()
    var_9 = var_2.validate_or_error(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert f'{type(var_9.error).__module__}.{type(var_9.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9.error) == 1
    var_10 = module_0.Choice(coerce_types=var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Choice'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.choices == []
    assert var_10.coerce_types is True
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_11 = var_1.validate_or_error(var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_1.freedesktop_os_release()
    var_2 = module_0.Choice(choices=var_1, coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Choice'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.choices == [('NAME', 'NAME'), ('ID', 'ID'), ('PRETTY_NAME', 'PRETTY_NAME'), ('VERSION_ID', 'VERSION_ID'), ('VERSION', 'VERSION'), ('VERSION_CODENAME', 'VERSION_CODENAME'), ('DEBIAN_VERSION_FULL', 'DEBIAN_VERSION_FULL'), ('HOME_URL', 'HOME_URL'), ('SUPPORT_URL', 'SUPPORT_URL'), ('BUG_REPORT_URL', 'BUG_REPORT_URL')]
    assert var_2.coerce_types == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.3', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_3 = var_0.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.3', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert var_3.error is None
    var_2.validate_or_error(var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = None
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = module_2._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    var_4 = var_0.validate(var_3)
    var_5 = module_0.Choice()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Choice'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.choices == []
    assert var_5.coerce_types is True
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_6 = module_0.Field(read_only=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Field'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only == {}
    assert module_0.Field.errors == {}
    var_6.validate_or_error(var_4)

def test_case_24():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Text'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'text'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = "Eh'@\\]"
    var_2 = var_0.validate(var_1)
    assert var_2 == "Eh'@\\]"

def test_case_25():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = module_1.python_implementation()
    assert var_1 == 'CPython'
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_26():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = module_1.python_branch()
    assert var_1 == ''
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_0.Email()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Email'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'email'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = module_3.getnode()
    assert var_1 == 165959165328334
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_2 = module_0.Decimal(minimum=var_1, maximum=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum == 165959165328334
    assert var_2.maximum == 165959165328334
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    var_3 = var_2.validate(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_4.Decimal.real).__module__}.{type(module_4.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.Decimal.imag).__module__}.{type(module_4.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    module_0.Any(title=var_3, default=var_1)

def test_case_28():
    var_0 = module_1.release()
    assert var_0 == '6.17.9-76061709-generic'
    var_1 = module_0.Array()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.items is None
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items is None
    assert var_1.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}

def test_case_29():
    var_0 = None
    var_1 = module_0.Password()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Password'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'password'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = None
    with pytest.raises(AssertionError):
        module_0.Field(description=var_0, allow_null=var_2)

def test_case_30():
    var_0 = -210
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, exclusive_maximum=var_0, precision=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum == -210
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum == -210
    assert var_2.multiple_of is None
    assert var_2.precision == -210
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_31():
    var_0 = []
    var_1 = module_5._tuplegetter
    with pytest.raises(AssertionError):
        module_0.Array(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = None
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = module_2._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    var_4 = var_0.validate(var_3)
    var_5 = module_0.Choice()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Choice'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.choices == []
    assert var_5.coerce_types is True
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_6 = var_5.serialize(var_1)
    var_5.validate_or_error(var_2)

def test_case_33():
    var_0 = module_0.UUID()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.UUID'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'uuid'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7

def test_case_34():
    var_0 = module_3.getnode()
    assert var_0 == 165959165328334
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Decimal(minimum=var_0, maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum == 165959165328334
    assert var_1.maximum == 165959165328334
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_4.Decimal.real).__module__}.{type(module_4.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.Decimal.imag).__module__}.{type(module_4.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_35():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = var_0.__or__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = module_1.freedesktop_os_release()
    var_3 = module_0.Field(allow_null=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.3', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert var_3.read_only is False
    assert module_0.Field.errors == {}
    var_4 = None
    var_5 = var_0.validate_or_error(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1

def test_case_36():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Number'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of == 5
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = var_2.validate(var_0)
    assert var_3 == 5

def test_case_37():
    var_0 = None
    var_1 = module_0.Integer(maximum=var_0, exclusive_maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_38():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = [var_1]
    var_3 = module_0.String(**var_0)
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_0.Array(var_2, var_3, **var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.items).__module__}.{type(var_4.items).__qualname__}' == 'builtins.list'
    assert len(var_4.items) == 1
    assert f'{type(var_4.additional_items).__module__}.{type(var_4.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_4.min_items == 1
    assert var_4.max_items is None
    assert var_4.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}

def test_case_39():
    var_0 = None
    var_1 = module_0.Integer(maximum=var_0, exclusive_maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = module_0.Boolean()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = None
    var_0.validate(var_1)

def test_case_41():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7

def test_case_42():
    var_0 = None
    var_1 = module_0.Integer(maximum=var_0, exclusive_maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_0.__str__()
    assert var_2 == 'None'
    var_3 = module_0.Array(max_items=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.items is None
    assert var_3.additional_items is False
    assert var_3.min_items is None
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = var_1.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_43():
    var_0 = None
    var_1 = module_0.Array(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.items is None
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items is None
    assert var_1.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_3.uuid4()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'uuid.UUID'
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.UUID.bytes).__module__}.{type(module_3.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.bytes_le).__module__}.{type(module_3.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.fields).__module__}.{type(module_3.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_low).__module__}.{type(module_3.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_mid).__module__}.{type(module_3.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_hi_version).__module__}.{type(module_3.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_hi_variant).__module__}.{type(module_3.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_low).__module__}.{type(module_3.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time).__module__}.{type(module_3.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq).__module__}.{type(module_3.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.node).__module__}.{type(module_3.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.hex).__module__}.{type(module_3.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.urn).__module__}.{type(module_3.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.variant).__module__}.{type(module_3.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.version).__module__}.{type(module_3.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.int).__module__}.{type(module_3.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.UUID.is_safe).__module__}.{type(module_3.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_1.validate_or_error(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_44():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = module_0.Object()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = var_1.__or__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = None
    var_4 = var_0.serialize(var_3)
    var_5 = var_0.has_default()
    assert var_5 is False
    var_6 = var_2.validate_or_error(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = None
    var_1 = module_0.Email()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Email'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'email'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_1.release()
    assert var_2 == '6.17.9-76061709-generic'
    var_3 = False
    var_4 = module_0.Decimal(maximum=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum is None
    assert var_4.maximum is None
    assert var_4.exclusive_minimum is None
    assert var_4.exclusive_maximum is None
    assert var_4.multiple_of is None
    assert var_4.precision is None
    assert var_4.coerce_types is True
    var_5 = module_0.Array(var_0, var_1, var_3, var_0, var_0, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.items is None
    assert f'{type(var_5.additional_items).__module__}.{type(var_5.additional_items).__qualname__}' == 'typesystem.fields.Email'
    assert var_5.min_items is False
    assert var_5.max_items is None
    assert var_5.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    module_0.Decimal(minimum=var_2, maximum=var_2, multiple_of=var_2)

def test_case_46():
    var_0 = None
    var_1 = module_0.Integer(maximum=var_0, exclusive_maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = None
    with pytest.raises(AssertionError):
        module_0.Array(var_3, unique_items=var_3)

def test_case_47():
    var_0 = module_1.processor()
    assert var_0 == ''
    var_1 = True
    var_2 = ',$U'
    var_3 = module_0.Integer(maximum=var_1, exclusive_minimum=var_1, precision=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is True
    assert var_3.exclusive_minimum is True
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision == ',$U'
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_4 = var_3.validate_or_error(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_48():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = var_0.validate(var_1)
    var_3 = None
    with pytest.raises(AssertionError):
        module_0.Array(var_2, min_items=var_3)

def test_case_49():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = 1623
    var_2 = None
    var_3 = {}
    var_4 = None
    var_5 = True
    var_6 = module_0.Decimal(minimum=var_2, maximum=var_4, exclusive_maximum=var_1, precision=var_4, coerce_types=var_5, **var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.minimum is None
    assert var_6.maximum is None
    assert var_6.exclusive_minimum is None
    assert var_6.exclusive_maximum == 1623
    assert var_6.multiple_of is None
    assert var_6.precision is None
    assert var_6.coerce_types is True
    var_7 = var_0.get_default_value()
    var_8 = var_6.validate_or_error(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert f'{type(var_8.error).__module__}.{type(var_8.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8.error) == 1
    var_9 = module_0.Decimal()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.minimum is None
    assert var_9.maximum is None
    assert var_9.exclusive_minimum is None
    assert var_9.exclusive_maximum is None
    assert var_9.multiple_of is None
    assert var_9.precision is None
    assert var_9.coerce_types is True
    var_10 = var_9.serialize(var_2)
    var_11 = var_0.validate_or_error(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert f'{type(var_11.error).__module__}.{type(var_11.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11.error) == 1

def test_case_50():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = None
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = module_2._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    var_4 = var_0.validate(var_3)
    var_5 = module_0.Decimal(precision=var_1, coerce_types=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum is None
    assert var_5.maximum is None
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of is None
    assert var_5.precision is None
    assert var_5.coerce_types == {}
    var_6 = var_5.validate_or_error(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1

def test_case_51():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items is None
    assert var_2.max_items == 2
    assert var_2.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = [var_0, var_0]
    var_4 = var_2.validate(var_3)

def test_case_52():
    var_0 = None
    var_1 = module_0.Integer(maximum=var_0, exclusive_maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    with pytest.raises(AssertionError):
        module_0.Array(max_items=var_1, unique_items=var_2)

def test_case_53():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = module_0.Object()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = var_1.__or__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = None
    var_4 = module_0.Array(var_0, exact_items=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.items).__module__}.{type(var_4.items).__qualname__}' == 'typesystem.fields.Date'
    assert var_4.additional_items is False
    assert var_4.min_items is None
    assert var_4.max_items is None
    assert var_4.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_5 = var_4.validate_or_error(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1

def test_case_54():
    var_0 = None
    var_1 = module_0.Integer()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_0.Integer()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    var_3 = module_0.Any(default=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    var_4 = var_3.get_default_value()
    var_5 = module_0.Boolean()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.coerce_types is True
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_6 = module_1.system()
    assert var_6 == 'Linux'
    var_7 = module_0.Object()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is True
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = module_0.Date()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Date'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format == 'date'
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    var_9 = module_0.Date()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Date'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length is None
    assert var_9.min_length is None
    assert var_9.format == 'date'
    assert var_9.coerce_types is True
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    var_10 = module_0.Choice(choices=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Choice'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.choices == [('L', 'L'), ('i', 'i'), ('n', 'n'), ('u', 'u'), ('x', 'x')]
    assert var_10.coerce_types is True
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_11 = var_8.validate_or_error(var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert f'{type(var_11.error).__module__}.{type(var_11.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11.error) == 1
    var_12 = var_3.validate_or_error(var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None

@pytest.mark.xfail(strict=True)
def test_case_55():
    var_0 = None
    var_1 = module_0.Any()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = True
    var_3 = var_1.validate(var_0)
    module_0.Integer(maximum=var_1, exclusive_maximum=var_2, precision=var_0, multiple_of=var_0, coerce_types=var_2)

def test_case_56():
    var_0 = module_0.UUID()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.UUID'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'uuid'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = module_1.freedesktop_os_release()
    var_2 = module_0.Array(exact_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.3', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert var_2.max_items == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.3', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    with pytest.raises(module_6.ValidationError):
        var_0.validate(var_1)

def test_case_57():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.DateTime()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'datetime'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_4 = module_0.Number(coerce_types=var_0, **var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Number'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum is None
    assert var_4.maximum is None
    assert var_4.exclusive_minimum is None
    assert var_4.exclusive_maximum is None
    assert var_4.multiple_of is None
    assert var_4.precision is None
    assert var_4.coerce_types is False
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_5 = True
    var_6 = None
    var_7 = module_0.String(allow_blank=var_5, trim_whitespace=var_0, min_length=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.default == ''
    assert var_7.allow_blank is True
    assert var_7.trim_whitespace is False
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format is None
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = var_7.validate_or_error(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert f'{type(var_8.error).__module__}.{type(var_8.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8.error) == 1
    var_9 = var_7.validate_or_error(var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value == ''
    assert var_9.error is None
    var_10 = {}
    var_11 = module_0.Choice(**var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Choice'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.choices == []
    assert var_11.coerce_types is True
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_12 = var_4.validate_or_error(var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert f'{type(var_12.error).__module__}.{type(var_12.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12.error) == 1

def test_case_58():
    var_0 = None
    var_1 = module_0.Object(pattern_properties=var_0, min_properties=var_0, max_properties=var_0, required=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_59():
    var_0 = None
    var_1 = module_0.Const(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Const'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.const is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_2 = module_0.Email()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Email'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'email'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    var_3 = module_1.python_revision()
    assert var_3 == ''
    var_4 = 'E+RzO'
    var_5 = var_1.__or__(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.any_of).__module__}.{type(var_5.any_of).__qualname__}' == 'builtins.list'
    assert len(var_5.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_6 = var_1.get_default_value()
    var_7 = var_5.validate_or_error(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_60():
    var_0 = None
    var_1 = module_0.Email()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Email'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'email'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_3.getnode()
    assert var_2 == 165959165328334
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_3 = module_0.Decimal(minimum=var_0, maximum=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Decimal'
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
    var_3.serialize(var_3)

@pytest.mark.xfail(strict=True)
def test_case_61():
    var_0 = module_1.python_implementation()
    assert var_0 == 'CPython'
    var_1 = module_0.Choice(choices=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Choice'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.choices == [('C', 'C'), ('P', 'P'), ('y', 'y'), ('t', 't'), ('h', 'h'), ('o', 'o'), ('n', 'n')]
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_1.validate(var_0)

def test_case_62():
    var_0 = None
    var_1 = module_0.Email()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Email'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'email'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = 'E+RzO'
    var_4 = module_0.Decimal(maximum=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum is None
    assert var_4.maximum is None
    assert var_4.exclusive_minimum is None
    assert var_4.exclusive_maximum is None
    assert var_4.multiple_of is None
    assert var_4.precision is None
    assert var_4.coerce_types is True
    var_5 = var_4.get_default_value()
    var_6 = var_1.validate_or_error(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_7 = module_0.Decimal()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.minimum is None
    assert var_7.maximum is None
    assert var_7.exclusive_minimum is None
    assert var_7.exclusive_maximum is None
    assert var_7.multiple_of is None
    assert var_7.precision is None
    assert var_7.coerce_types is True
    with pytest.raises(AssertionError):
        module_0.Number(exclusive_minimum=var_4, exclusive_maximum=var_0, precision=var_0)

@pytest.mark.xfail(strict=True)
def test_case_63():
    var_0 = None
    var_1 = True
    var_2 = module_0.Number(exclusive_minimum=var_0, multiple_of=var_0, coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Number'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = False
    var_4 = module_1.processor()
    assert var_4 == ''
    var_5 = module_0.Number(exclusive_minimum=var_0, multiple_of=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Number'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum is None
    assert var_5.maximum is None
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of is False
    assert var_5.precision is None
    assert var_5.coerce_types is True
    module_0.Integer(multiple_of=var_4)

def test_case_64():
    var_0 = None
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Choice'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.choices == []
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_65():
    var_0 = module_0.Email()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Email'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'email'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = var_0.get_default_value()
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = var_0.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = var_2.__iter__()
    var_5 = 'E|+RzO'
    with pytest.raises(AssertionError):
        module_0.Number(maximum=var_1, exclusive_minimum=var_1, exclusive_maximum=var_5)

def test_case_66():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = var_0.validate(var_1)

def test_case_67():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = var_0.__or__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_68():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items == 1
    assert var_2.max_items is None
    assert var_2.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = []
    with pytest.raises(module_6.ValidationError):
        var_2.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_69():
    var_0 = module_1.machine()
    assert var_0 == 'x86_64'
    var_1 = module_0.Object()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_1.python_implementation()
    assert var_2 == 'CPython'
    var_3 = module_0.String(pattern=var_2, format=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'x86_64'
    assert var_3.coerce_types is True
    assert var_3.pattern == 'CPython'
    assert f'{type(var_3.pattern_regex).__module__}.{type(var_3.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_0.validate(var_2)

def test_case_70():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = module_0.Object()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = None
    var_3 = module_0.Decimal(minimum=var_2, maximum=var_2, exclusive_maximum=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Decimal'
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
    var_4 = var_0.validate_or_error(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1
    var_5 = module_3.uuid4()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'uuid.UUID'
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.UUID.bytes).__module__}.{type(module_3.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.bytes_le).__module__}.{type(module_3.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.fields).__module__}.{type(module_3.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_low).__module__}.{type(module_3.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_mid).__module__}.{type(module_3.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_hi_version).__module__}.{type(module_3.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_hi_variant).__module__}.{type(module_3.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_low).__module__}.{type(module_3.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time).__module__}.{type(module_3.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq).__module__}.{type(module_3.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.node).__module__}.{type(module_3.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.hex).__module__}.{type(module_3.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.urn).__module__}.{type(module_3.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.variant).__module__}.{type(module_3.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.version).__module__}.{type(module_3.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.int).__module__}.{type(module_3.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.UUID.is_safe).__module__}.{type(module_3.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    with pytest.raises(AssertionError):
        module_0.String(pattern=var_5, format=var_2)

def test_case_71():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = module_0.Object()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = var_1.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = module_0.Const(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Const'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.const).__module__}.{type(var_4.const).__qualname__}' == 'typesystem.base.ValidationResult'
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_5 = None
    var_6 = var_4.validate_or_error(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1

def test_case_72():
    var_0 = None
    var_1 = module_0.Integer(maximum=var_0, exclusive_maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.__or__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_73():
    var_0 = None
    var_1 = True
    var_2 = module_0.Array(max_items=var_0, exact_items=var_0, unique_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items is None
    assert var_2.max_items is None
    assert var_2.unique_items is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_74():
    var_0 = 1
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = {}
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_2, **var_3)

def test_case_75():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.properties).__module__}.{type(var_6.properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.properties) == 1
    assert var_6.pattern_properties == {}
    assert var_6.additional_properties is True
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == ['name']
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    with pytest.raises(module_6.ValidationError):
        var_6.validate(var_3)

def test_case_76():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = var_0.__or__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = module_0.Object(additional_properties=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert f'{type(var_2.additional_properties).__module__}.{type(var_2.additional_properties).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.property_names is None
    assert var_2.min_properties is None
    assert var_2.max_properties is None
    assert var_2.required == []
    var_3 = module_1.freedesktop_os_release()
    var_4 = var_2.validate_or_error(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 10

@pytest.mark.xfail(strict=True)
def test_case_77():
    var_0 = module_0.IPAddress()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'ipaddress'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = None
    var_2 = {}
    var_3 = module_0.Object(properties=var_0, additional_properties=var_0, property_names=var_0, required=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert f'{type(var_3.additional_properties).__module__}.{type(var_3.additional_properties).__qualname__}' == 'typesystem.fields.IPAddress'
    assert f'{type(var_3.property_names).__module__}.{type(var_3.property_names).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = None
    var_5 = module_0.Float(exclusive_minimum=var_1, multiple_of=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Float'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum is None
    assert var_5.maximum is None
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of is None
    assert var_5.precision is None
    assert var_5.coerce_types is True
    var_0.__or__(var_4)

def test_case_78():
    var_0 = module_1.version()
    assert var_0 == '#202511241048~1764607909~24.04~df6b2b6 SMP PREEMPT_DYNAMIC Mon D'
    var_1 = None
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_1, additional_properties=var_0, property_names=var_1, max_properties=var_1, required=var_1)

@pytest.mark.xfail(strict=True)
def test_case_79():
    var_0 = module_0.IPAddress()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'ipaddress'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = var_0.__or__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = module_0.Date()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Date'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'date'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    var_3 = None
    var_4 = module_0.Object(pattern_properties=var_3, additional_properties=var_0, property_names=var_3, min_properties=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert var_4.pattern_properties == {}
    assert f'{type(var_4.additional_properties).__module__}.{type(var_4.additional_properties).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = None
    var_1.__or__(var_5)

def test_case_80():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = None
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = module_2._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    with pytest.raises(AssertionError):
        module_0.Array(var_0, min_items=var_3)

@pytest.mark.xfail(strict=True)
def test_case_81():
    var_0 = module_0.IPAddress()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'ipaddress'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = var_0.__or__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = module_0.Date()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Date'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'date'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    var_3 = module_0.Object()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is True
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = var_3.__or__(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 3
    var_5 = module_0.Choice()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Choice'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.choices == []
    assert var_5.coerce_types is True
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_5.validate_or_error(var_2)

@pytest.mark.xfail(strict=True)
def test_case_82():
    var_0 = None
    var_1 = True
    var_2 = module_0.Field(default=var_0, allow_null=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_3 = module_0.Integer(maximum=var_0, exclusive_maximum=var_0)
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
    var_4 = None
    var_5 = module_1.libc_ver(var_0, version=var_0, chunksize=var_4)
    var_5.__ior__(var_0)

def test_case_83():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = module_0.Object()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_3.uuid4()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'uuid.UUID'
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.UUID.bytes).__module__}.{type(module_3.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.bytes_le).__module__}.{type(module_3.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.fields).__module__}.{type(module_3.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_low).__module__}.{type(module_3.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_mid).__module__}.{type(module_3.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_hi_version).__module__}.{type(module_3.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_hi_variant).__module__}.{type(module_3.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_low).__module__}.{type(module_3.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time).__module__}.{type(module_3.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq).__module__}.{type(module_3.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.node).__module__}.{type(module_3.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.hex).__module__}.{type(module_3.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.urn).__module__}.{type(module_3.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.variant).__module__}.{type(module_3.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.version).__module__}.{type(module_3.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.int).__module__}.{type(module_3.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.UUID.is_safe).__module__}.{type(module_3.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    with pytest.raises(AssertionError):
        module_0.String(format=var_2)

def test_case_84():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = False
    var_3 = var_1.get_default_value()
    var_4 = module_0.Object(max_properties=var_2, required=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert var_4.pattern_properties == {}
    assert var_4.additional_properties is True
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is False
    assert var_4.required == {}
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = var_4.validate_or_error(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value == {}
    assert var_5.error is None

@pytest.mark.xfail(strict=True)
def test_case_85():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == 3
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = 'hello'
    var_2.validate(var_3)

def test_case_86():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = None
    var_2 = module_0.Object(max_properties=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert var_2.additional_properties is True
    assert var_2.property_names is None
    assert var_2.min_properties is None
    assert var_2.max_properties is None
    assert var_2.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = None
    var_4 = var_2.validate_or_error(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1
    var_5 = module_3.uuid4()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'uuid.UUID'
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.UUID.bytes).__module__}.{type(module_3.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.bytes_le).__module__}.{type(module_3.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.fields).__module__}.{type(module_3.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_low).__module__}.{type(module_3.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_mid).__module__}.{type(module_3.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_hi_version).__module__}.{type(module_3.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_hi_variant).__module__}.{type(module_3.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_low).__module__}.{type(module_3.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time).__module__}.{type(module_3.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq).__module__}.{type(module_3.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.node).__module__}.{type(module_3.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.hex).__module__}.{type(module_3.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.urn).__module__}.{type(module_3.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.variant).__module__}.{type(module_3.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.version).__module__}.{type(module_3.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.int).__module__}.{type(module_3.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.UUID.is_safe).__module__}.{type(module_3.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    with pytest.raises(AssertionError):
        module_0.String(allow_blank=var_5, max_length=var_5, pattern=var_3)

def test_case_87():
    var_0 = module_1.version()
    assert var_0 == '#202511241048~1764607909~24.04~df6b2b6 SMP PREEMPT_DYNAMIC Mon D'
    with pytest.raises(AssertionError):
        module_0.Object(max_properties=var_0)

def test_case_88():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert var_2.additional_properties is True
    assert var_2.property_names is None
    assert var_2.min_properties == 1
    assert var_2.max_properties is None
    assert var_2.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    with pytest.raises(module_6.ValidationError):
        var_2.validate(var_1)

def test_case_89():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = module_0.String(allow_blank=var_0, min_length=var_0, pattern=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.default == ''
    assert var_3.allow_blank is True
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is True
    assert var_3.format is None
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}

def test_case_90():
    var_0 = None
    var_1 = module_0.Object(pattern_properties=var_0, min_properties=var_0, max_properties=var_0, required=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.Boolean()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.coerce_types is True
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_3 = module_3.uuid4()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'uuid.UUID'
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.UUID.bytes).__module__}.{type(module_3.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.bytes_le).__module__}.{type(module_3.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.fields).__module__}.{type(module_3.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_low).__module__}.{type(module_3.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_mid).__module__}.{type(module_3.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_hi_version).__module__}.{type(module_3.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_hi_variant).__module__}.{type(module_3.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_low).__module__}.{type(module_3.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time).__module__}.{type(module_3.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq).__module__}.{type(module_3.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.node).__module__}.{type(module_3.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.hex).__module__}.{type(module_3.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.urn).__module__}.{type(module_3.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.variant).__module__}.{type(module_3.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.version).__module__}.{type(module_3.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.int).__module__}.{type(module_3.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.UUID.is_safe).__module__}.{type(module_3.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    with pytest.raises(AssertionError):
        module_0.String(trim_whitespace=var_3, max_length=var_0, min_length=var_3, pattern=var_0, coerce_types=var_0)

def test_case_91():
    var_0 = module_0.Date()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Date'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'date'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = module_1.freedesktop_os_release()
    var_2 = None
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_1, pattern_properties=var_2, additional_properties=var_0, max_properties=var_2)

def test_case_92():
    var_0 = None
    var_1 = module_0.Array(min_items=var_0, max_items=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.items is None
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items is None
    assert var_1.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_1.machine()
    assert var_2 == 'x86_64'
    var_3 = var_2.__dir__()
    var_4 = var_1.validate_or_error(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value == ['__new__', '__repr__', '__hash__', '__str__', '__getattribute__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__iter__', '__mod__', '__rmod__', '__len__', '__getitem__', '__add__', '__mul__', '__rmul__', '__contains__', 'encode', 'replace', 'split', 'rsplit', 'join', 'capitalize', 'casefold', 'title', 'center', 'count', 'expandtabs', 'find', 'partition', 'index', 'ljust', 'lower', 'lstrip', 'rfind', 'rindex', 'rjust', 'rstrip', 'rpartition', 'splitlines', 'strip', 'swapcase', 'translate', 'upper', 'startswith', 'endswith', 'removeprefix', 'removesuffix', 'isascii', 'islower', 'isupper', 'istitle', 'isspace', 'isdecimal', 'isdigit', 'isnumeric', 'isalpha', 'isalnum', 'isidentifier', 'isprintable', 'zfill', 'format', 'format_map', '__format__', 'maketrans', '__sizeof__', '__getnewargs__', '__doc__', '__setattr__', '__delattr__', '__init__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__dir__', '__class__']
    assert var_4.error is None

def test_case_93():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_0.Array(exact_items=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.items is None
    assert var_1.additional_items is False
    assert f'{type(var_1.min_items).__module__}.{type(var_1.min_items).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_1.max_items).__module__}.{type(var_1.max_items).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = None
    var_3 = var_1.serialize(var_2)
    var_4 = var_1.serialize(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.items is None
    assert var_4.additional_items is False
    assert f'{type(var_4.min_items).__module__}.{type(var_4.min_items).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_4.max_items).__module__}.{type(var_4.max_items).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.unique_items is False

def test_case_94():
    var_0 = module_0.Boolean()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1

def test_case_95():
    var_0 = None
    var_1 = module_0.Object(pattern_properties=var_0, min_properties=var_0, max_properties=var_0, required=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    with pytest.raises(AssertionError):
        module_0.Object(required=var_2)

def test_case_96():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items is None
    assert var_2.max_items == 2
    assert var_2.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = [var_0, var_0]
    var_4 = var_2.validate(var_3)
    with pytest.raises(AssertionError):
        module_0.Object(min_properties=var_4)

def test_case_97():
    var_0 = None
    var_1 = module_0.Object(pattern_properties=var_0, min_properties=var_0, max_properties=var_0, required=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = var_1.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = var_2.__iter__()
    var_4 = var_2.__iter__()
    var_5 = module_0.Boolean()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.coerce_types is True
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_6 = module_0.Const(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Const'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.const is None
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_7 = var_1.validate_or_error(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1
    var_8 = module_0.IPAddress()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format == 'ipaddress'
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    var_9 = False
    var_10 = module_0.Array(max_items=var_0, exact_items=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Array'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.items is None
    assert var_10.additional_items is False
    assert var_10.min_items is None
    assert var_10.max_items is None
    assert var_10.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_11 = var_5.validate_or_error(var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is False
    assert var_11.error is None

def test_case_98():
    var_0 = None
    var_1 = True
    var_2 = module_0.String(allow_blank=var_1, format=var_0, coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.default == ''
    assert var_2.allow_blank is True
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == ''
    assert var_3.error is None

def test_case_99():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_1.freedesktop_os_release()
    var_2 = module_0.Array(exact_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.3', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert var_2.max_items == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.3', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = var_2.serialize(var_1)

def test_case_100():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_1.freedesktop_os_release()
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.3', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert var_2.error is None

def test_case_101():
    var_0 = module_1.python_build()
    var_1 = module_1.python_version_tuple()
    var_2 = module_5._tuplegetter
    with pytest.raises(AssertionError):
        module_0.Array(var_1, var_2)

def test_case_102():
    var_0 = module_3.getnode()
    assert var_0 == 165959165328334
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Decimal(minimum=var_0, maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum == 165959165328334
    assert var_1.maximum == 165959165328334
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.__or__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_0.Array()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.items is None
    assert var_3.additional_items is False
    assert var_3.min_items is None
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = var_2.validate_or_error(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.value).__module__}.{type(var_4.value).__qualname__}' == 'decimal.Decimal'
    assert var_4.error is None
    var_5 = var_3.serialize(var_0)
    assert var_5 == 165959165328334

def test_case_103():
    var_0 = module_3.getnode()
    assert var_0 == 165959165328334
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Decimal(minimum=var_0, maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum == 165959165328334
    assert var_1.maximum == 165959165328334
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_4.Decimal.real).__module__}.{type(module_4.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.Decimal.imag).__module__}.{type(module_4.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0.Integer(maximum=var_2, exclusive_minimum=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert f'{type(var_3.maximum).__module__}.{type(var_3.maximum).__qualname__}' == 'decimal.Decimal'
    assert f'{type(var_3.exclusive_minimum).__module__}.{type(var_3.exclusive_minimum).__qualname__}' == 'decimal.Decimal'
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_4.DefaultContext).__module__}.{type(module_4.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_4.HAVE_CONTEXTVAR is True
    assert module_4.HAVE_THREADS is True
    assert f'{type(module_4.BasicContext).__module__}.{type(module_4.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_4.ExtendedContext).__module__}.{type(module_4.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_4.MAX_PREC == 999999999999999999
    assert module_4.MAX_EMAX == 999999999999999999
    assert module_4.MIN_EMIN == -999999999999999999
    assert module_4.MIN_ETINY == -1999999999999999997
    assert module_4.ROUND_UP == 'ROUND_UP'
    assert module_4.ROUND_DOWN == 'ROUND_DOWN'
    assert module_4.ROUND_CEILING == 'ROUND_CEILING'
    assert module_4.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_4.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_4.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_4.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_4.ROUND_05UP == 'ROUND_05UP'
    var_4 = var_3.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_104():
    var_0 = module_3.getnode()
    assert var_0 == 165959165328334
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Decimal(minimum=var_0, maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum == 165959165328334
    assert var_1.maximum == 165959165328334
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_0.__xor__(var_0)
    assert var_2 == 0
    var_3 = var_1.validate(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_4.Decimal.real).__module__}.{type(module_4.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.Decimal.imag).__module__}.{type(module_4.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_1.validate(var_2)

def test_case_105():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = var_0.get_default_value()
    var_2 = module_0.Object(additional_properties=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert var_2.additional_properties is None
    assert var_2.property_names is None
    assert var_2.min_properties is None
    assert var_2.max_properties is None
    assert var_2.required == []
    var_3 = module_1.freedesktop_os_release()
    var_4 = module_0.Array()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.items is None
    assert var_4.additional_items is False
    assert var_4.min_items is None
    assert var_4.max_items is None
    assert var_4.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_5 = var_2.validate_or_error(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value == {}
    assert var_5.error is None

def test_case_106():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.items is None
    assert var_3.additional_items is False
    assert var_3.min_items is None
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

def test_case_107():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_0.Array(var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.additional_items is False
    assert var_2.min_items is None
    assert var_2.max_items is None
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = 'invalid'
    var_4 = 3
    var_5 = [var_4, var_3, var_4]
    with pytest.raises(module_6.ValidationError):
        var_2.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_108():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items is None
    assert var_2.max_items is None
    assert var_2.unique_items is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = 2
    var_4 = [var_0, var_3, var_0]
    var_2.validate(var_4)

def test_case_109():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = [var_1]
    var_3 = module_0.String(**var_0)
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.items).__module__}.{type(var_5.items).__qualname__}' == 'builtins.list'
    assert len(var_5.items) == 1
    assert f'{type(var_5.additional_items).__module__}.{type(var_5.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_5.min_items == 1
    assert var_5.max_items is None
    assert var_5.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_6 = 1
    var_7 = [var_6, var_2]
    with pytest.raises(module_6.ValidationError):
        var_5.validate(var_7)

def test_case_110():
    var_0 = module_0.Time()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Time'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'time'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = 3
    var_2 = {}
    var_3 = module_0.Array(exact_items=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.items is None
    assert var_3.additional_items is False
    assert var_3.min_items == 3
    assert var_3.max_items == 3
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5, var_1]
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

def test_case_111():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items == 1
    assert var_2.max_items is None
    assert var_2.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = [var_2]
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)

def test_case_112():
    var_0 = module_0.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_1 = module_0.Field()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = {}
    var_5 = module_0.Array(var_2, min_items=var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.items).__module__}.{type(var_5.items).__qualname__}' == 'builtins.list'
    assert len(var_5.items) == 2
    assert var_5.additional_items is False
    assert var_5.min_items == 5
    assert var_5.max_items == 2
    assert var_5.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_6 = var_5.min_items
    assert var_6 == 5

def test_case_113():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = [var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.items).__module__}.{type(var_5.items).__qualname__}' == 'builtins.list'
    assert len(var_5.items) == 1
    assert var_5.additional_items is False
    assert var_5.min_items == 1
    assert var_5.max_items == 1
    assert var_5.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7]
    with pytest.raises(module_6.ValidationError):
        var_5.validate(var_8)

def test_case_114():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = {}
    var_3 = module_0.String(**var_2)
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.items).__module__}.{type(var_6.items).__qualname__}' == 'builtins.list'
    assert len(var_6.items) == 2
    assert var_6.additional_items is False
    assert var_6.min_items == 2
    assert var_6.max_items == 2
    assert var_6.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_7 = 42
    var_8 = 'hello'
    var_9 = [var_7, var_8]
    var_10 = var_6.serialize(var_9)
    var_11 = bool(var_10 == [42, 'hello'])
    assert var_11 is True

def test_case_115():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.additional_items is False
    assert var_3.min_items is None
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.serialize(var_7)
    var_9 = bool(var_8 == [1, 2, 3])
    assert var_9 is True

def test_case_116():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_0.IPAddress()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'ipaddress'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    var_3 = [var_1]
    var_4 = True
    var_5 = module_0.Array(var_3, var_4, **var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.items).__module__}.{type(var_5.items).__qualname__}' == 'builtins.list'
    assert len(var_5.items) == 1
    assert var_5.additional_items is True
    assert var_5.min_items == 1
    assert var_5.max_items is None
    assert var_5.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_6 = 1
    var_7 = [var_6, var_4]
    var_8 = var_5.validate(var_7)
    var_9 = bool(False)

def test_case_117():
    var_0 = 7
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items == 7
    assert var_2.max_items is None
    assert var_2.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = []
    with pytest.raises(module_6.ValidationError):
        var_2.validate(var_3)

def test_case_118():
    var_0 = module_3.getnode()
    assert var_0 == 165959165328334
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Decimal(minimum=var_0, maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum == 165959165328334
    assert var_1.maximum == 165959165328334
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_0.Object(min_properties=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert var_2.additional_properties is True
    assert var_2.property_names is None
    assert var_2.min_properties == 165959165328334
    assert var_2.max_properties is None
    assert var_2.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = module_1.freedesktop_os_release()
    var_4 = var_2.validate_or_error(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_119():
    var_0 = -4
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items is None
    assert var_2.max_items == -4
    assert var_2.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = [var_0, var_0]
    var_4 = None
    var_5 = var_2.validate_or_error(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    with pytest.raises(module_6.ValidationError):
        var_2.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_120():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Number'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_2 = 'nan'
    var_3 = float(var_2)
    var_1.validate(var_3)

def test_case_121():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Number'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

def test_case_122():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Number'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

@pytest.mark.xfail(strict=True)
def test_case_123():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Number'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum == 100
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = 101
    var_2.validate(var_3)

def test_case_124():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Number'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum == 100
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = 99
    var_4 = var_2.validate(var_3)
    assert var_4 == 99

def test_case_125():
    var_0 = '0.1'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Number'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision == '0.1'
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = 3.25
    var_4 = var_2.validate(var_3)
    assert var_4 == pytest.approx(3.3, abs=0.01, rel=0.01)
    var_5 = bool(var_4 == 3.2 or var_4 == 3.3)
    assert var_5 is True

def test_case_126():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Number'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum == 10
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = 11
    var_4 = var_2.validate(var_3)
    assert var_4 == 11

def test_case_127():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Number'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4 = var_3.__or__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is True
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = None
    var_6 = var_3.validate(var_5)

def test_case_128():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Number'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4 = module_0.Array(additional_items=var_3, exact_items=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.items is None
    assert f'{type(var_4.additional_items).__module__}.{type(var_4.additional_items).__qualname__}' == 'typesystem.fields.Number'
    assert var_4.min_items is True
    assert var_4.max_items is True
    assert var_4.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_5 = None
    var_6 = module_0.Array(var_5, min_items=var_5, max_items=var_5, exact_items=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.items is None
    assert var_6.additional_items is False
    assert var_6.min_items is None
    assert var_6.max_items is None
    assert var_6.unique_items is False
    var_7 = var_6.__or__(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Union'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is True
    assert var_7.read_only is False
    assert f'{type(var_7.any_of).__module__}.{type(var_7.any_of).__qualname__}' == 'builtins.list'
    assert len(var_7.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_8 = var_7.validate_or_error(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None

def test_case_129():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    with pytest.raises(AssertionError):
        module_0.Const(var_1, **var_2)

def test_case_130():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format is None
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

def test_case_131():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is False
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = '  hello  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  hello  '

def test_case_132():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.default == ''
    assert var_2.allow_blank is True
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

def test_case_133():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(trim_whitespace=var_0, coerce_types=var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format is None
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

def test_case_134():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length == 3
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

def test_case_135():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
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
    assert var_2.pattern == '^[a-z]+$'
    assert f'{type(var_2.pattern_regex).__module__}.{type(var_2.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

@pytest.mark.xfail(strict=True)
def test_case_136():
    var_0 = '^[0-9]+$'
    var_1 = module_7.compile(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 're.Pattern'
    assert module_7.ASCII == module_7.RegexFlag.ASCII
    assert module_7.A == module_7.RegexFlag.ASCII
    assert module_7.IGNORECASE == module_7.RegexFlag.IGNORECASE
    assert module_7.I == module_7.RegexFlag.IGNORECASE
    assert module_7.LOCALE == module_7.RegexFlag.LOCALE
    assert module_7.L == module_7.RegexFlag.LOCALE
    assert module_7.UNICODE == module_7.RegexFlag.UNICODE
    assert module_7.U == module_7.RegexFlag.UNICODE
    assert module_7.MULTILINE == module_7.RegexFlag.MULTILINE
    assert module_7.M == module_7.RegexFlag.MULTILINE
    assert module_7.DOTALL == module_7.RegexFlag.DOTALL
    assert module_7.S == module_7.RegexFlag.DOTALL
    assert module_7.VERBOSE == module_7.RegexFlag.VERBOSE
    assert module_7.X == module_7.RegexFlag.VERBOSE
    assert module_7.TEMPLATE == module_7.RegexFlag.TEMPLATE
    assert module_7.T == module_7.RegexFlag.TEMPLATE
    assert module_7.DEBUG == module_7.RegexFlag.DEBUG
    assert f'{type(module_7.Pattern.pattern).__module__}.{type(module_7.Pattern.pattern).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_7.Pattern.flags).__module__}.{type(module_7.Pattern.flags).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_7.Pattern.groups).__module__}.{type(module_7.Pattern.groups).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_7.Pattern.groupindex).__module__}.{type(module_7.Pattern.groupindex).__qualname__}' == 'builtins.getset_descriptor'
    var_2 = {}
    var_3 = module_0.String(pattern=var_1, **var_2)
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
    assert var_3.pattern == '^[0-9]+$'
    assert f'{type(var_3.pattern_regex).__module__}.{type(var_3.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_137():
    var_0 = -1275.392
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Number'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of == pytest.approx(-1275.392, abs=0.01, rel=0.01)
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = var_2.validate(var_0)
    assert var_3 == pytest.approx(-1275.392, abs=0.01, rel=0.01)
    var_4 = bool(False)
    var_4.validate(var_3)

def test_case_138():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == 10
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = 'uhel'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'uhel'

@pytest.mark.xfail(strict=True)
def test_case_139():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Number'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of == pytest.approx(0.5, abs=0.01, rel=0.01)
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = 2.3
    var_2.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_140():
    var_0 = '^az]$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
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
    assert var_2.pattern == '^az]$'
    assert f'{type(var_2.pattern_regex).__module__}.{type(var_2.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = 'hello'
    var_2.validate(var_3)
    assert var_4 == 'hello'

@pytest.mark.xfail(strict=True)
def test_case_141():
    var_0 = 2455
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length == 2455
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = 'hello'
    var_2.validate(var_3)
    assert var_4 == 'hello'

@pytest.mark.xfail(strict=True)
def test_case_142():
    var_0 = module_0.UUID()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.UUID'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'uuid'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = 5
    var_2 = {}
    var_3 = module_0.Number(multiple_of=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Number'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of == 5
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4 = 49
    var_5 = var_3.get_default_value()
    var_6 = module_3.uuid4()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'uuid.UUID'
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.UUID.bytes).__module__}.{type(module_3.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.bytes_le).__module__}.{type(module_3.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.fields).__module__}.{type(module_3.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_low).__module__}.{type(module_3.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_mid).__module__}.{type(module_3.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_hi_version).__module__}.{type(module_3.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_hi_variant).__module__}.{type(module_3.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_low).__module__}.{type(module_3.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time).__module__}.{type(module_3.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq).__module__}.{type(module_3.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.node).__module__}.{type(module_3.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.hex).__module__}.{type(module_3.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.urn).__module__}.{type(module_3.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.variant).__module__}.{type(module_3.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.version).__module__}.{type(module_3.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.int).__module__}.{type(module_3.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.UUID.is_safe).__module__}.{type(module_3.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    var_3.validate(var_4)
    assert var_7 == 15

def test_case_143():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.get_default_value()
    var_3 = module_1.system()
    assert var_3 == 'Linux'
    var_4 = module_0.Object(max_properties=var_2, required=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert var_4.pattern_properties == {}
    assert var_4.additional_properties is True
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == 'Linux'
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = var_4.validate_or_error(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 5

def test_case_144():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_4 = 'none'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

def test_case_145():
    var_0 = 'option2'
    var_1 = [var_0, var_0]
    var_2 = {}
    var_3 = module_0.Choice(choices=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Choice'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.choices == [('option2', 'option2'), ('option2', 'option2')]
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_4 = var_3.validate(var_0)
    assert var_4 == 'option2'

def test_case_146():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = (var_0, var_1, var_2)
    var_4 = [var_3]
    var_5 = {}
    with pytest.raises(AssertionError):
        module_0.Choice(choices=var_4, **var_5)

@pytest.mark.xfail(strict=True)
def test_case_147():
    var_0 = None
    var_1 = module_0.Boolean(coerce_types=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.coerce_types is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_2 = module_0.Const(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Const'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.const is None
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_3 = var_1.validate_or_error(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = module_1.system()
    assert var_4 == 'Linux'
    var_5 = module_0.Object(additional_properties=var_0, required=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.properties == {}
    assert var_5.pattern_properties == {}
    assert var_5.additional_properties is None
    assert var_5.property_names is None
    assert var_5.min_properties is None
    assert var_5.max_properties is None
    assert var_5.required == 'Linux'
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = module_0.Date()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Date'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.allow_blank is False
    assert var_6.trim_whitespace is True
    assert var_6.max_length is None
    assert var_6.min_length is None
    assert var_6.format == 'date'
    assert var_6.coerce_types is True
    assert var_6.pattern is None
    assert var_6.pattern_regex is None
    var_7 = module_0.Date()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Date'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format == 'date'
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    module_0.Array(additional_items=var_4, exact_items=var_4, **var_4)

def test_case_148():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is True
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

def test_case_149():
    var_0 = '^S_'
    var_1 = module_0.Text()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Text'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'text'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert f'{type(var_4.pattern_properties).__module__}.{type(var_4.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_4.pattern_properties) == 1
    assert var_4.additional_properties is True
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = {var_0: var_0}
    var_6 = var_4.validate(var_5)

def test_case_150():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    with pytest.raises(module_6.ValidationError):
        var_1.validate(var_4)

def test_case_151():
    var_0 = ''
    var_1 = module_0.Text()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Text'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'text'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert f'{type(var_4.pattern_properties).__module__}.{type(var_4.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_4.pattern_properties) == 1
    assert var_4.additional_properties is True
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = {var_0: var_0}
    with pytest.raises(module_6.ValidationError):
        var_4.validate(var_5)

def test_case_152():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.properties).__module__}.{type(var_6.properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.properties) == 1
    assert var_6.pattern_properties == {}
    assert var_6.additional_properties is False
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = 'name'
    var_8 = 'extra'
    var_9 = {var_7: var_3, var_8: var_0}
    with pytest.raises(module_6.ValidationError):
        var_6.validate(var_9)

def test_case_153():
    var_0 = {}
    var_1 = module_0.String(**var_0)
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert f'{type(var_3.additional_properties).__module__}.{type(var_3.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = 'value2'
    var_5 = {var_4: var_4, var_4: var_4}
    var_6 = var_3.validate(var_5)

def test_case_154():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert var_2.additional_properties is True
    assert var_2.property_names is None
    assert var_2.min_properties is None
    assert var_2.max_properties == 1
    assert var_2.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    with pytest.raises(module_6.ValidationError):
        var_2.validate(var_7)

def test_case_155():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.properties).__module__}.{type(var_6.properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.properties) == 1
    assert var_6.pattern_properties == {}
    assert var_6.additional_properties is True
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == ['name']
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = {}
    with pytest.raises(module_6.ValidationError):
        var_6.validate(var_7)

def test_case_156():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.String(**var_4)
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
    var_6 = module_0.Object(properties=var_3, additional_properties=var_5, **var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.properties).__module__}.{type(var_6.properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.properties) == 1
    assert var_6.pattern_properties == {}
    assert f'{type(var_6.additional_properties).__module__}.{type(var_6.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = {var_0: var_0, var_0: var_0}
    var_8 = var_6.validate(var_7)

def test_case_157():
    var_0 = '^'
    var_1 = module_0.Text()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Text'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'text'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert f'{type(var_4.pattern_properties).__module__}.{type(var_4.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_4.pattern_properties) == 1
    assert var_4.additional_properties is True
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = {var_0: var_0}
    var_6 = var_4.validate(var_5)

def test_case_158():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == 5
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.Object(property_names=var_2, **var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is True
    assert f'{type(var_3.property_names).__module__}.{type(var_3.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = 'verylongkey'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    with pytest.raises(module_6.ValidationError):
        var_3.validate(var_6)

def test_case_159():
    var_0 = -5
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert var_2.additional_properties is True
    assert var_2.property_names is None
    assert var_2.min_properties == -5
    assert var_2.max_properties is None
    assert var_2.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)

def test_case_160():
    var_0 = 'name'
    var_1 = 'Unknown'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.default == 'Unknown'
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.properties).__module__}.{type(var_7.properties).__qualname__}' == 'builtins.dict'
    assert len(var_7.properties) == 1
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is True
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'Unknown'})
    assert var_10 is True

def test_case_161():
    var_0 = module_1.release()
    assert var_0 == '6.17.9-76061709-generic'
    var_1 = module_0.Text()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Text'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'text'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is True
    assert f'{type(var_3.property_names).__module__}.{type(var_3.property_names).__qualname__}' == 'typesystem.fields.Text'
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = 'verylongkey'
    var_5 = '6Wqkh|\tQpM'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)

def test_case_162():
    var_0 = 'option1'
    var_1 = [var_0, var_0]
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Choice(choices=var_1, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Choice'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert var_5.choices == [('option1', 'option1'), ('option1', 'option1')]
    assert var_5.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

def test_case_163():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Text'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format == 'text'
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1 = {var_0: var_0}
    var_2 = {}
    var_3 = None
    var_4 = module_0.Object(max_properties=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert var_4.pattern_properties == {}
    assert var_4.additional_properties is True
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    with pytest.raises(AssertionError):
        module_0.Object(pattern_properties=var_1, **var_2)

def test_case_164():
    var_0 = {}
    var_1 = None
    var_2 = module_0.Integer(**var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_3 = module_0.Email()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Email'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'email'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    var_4 = var_2.__or__(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = var_4.__or__(var_3)
    assert len(var_4.any_of) == 3
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.any_of).__module__}.{type(var_5.any_of).__qualname__}' == 'builtins.list'
    assert len(var_5.any_of) == 3
    var_6 = var_2.get_default_value()
    var_7 = module_1.system()
    assert var_7 == 'Linux'
    var_8 = module_0.Object(max_properties=var_6, required=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Object'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.properties == {}
    assert var_8.pattern_properties == {}
    assert var_8.additional_properties is True
    assert var_8.property_names is None
    assert var_8.min_properties is None
    assert var_8.max_properties is None
    assert var_8.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_9 = [var_2]
    var_10 = module_0.Date()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Date'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length is None
    assert var_10.min_length is None
    assert var_10.format == 'date'
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = module_0.Choice(choices=var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Choice'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.choices == [('L', 'L'), ('i', 'i'), ('n', 'n'), ('u', 'u'), ('x', 'x')]
    assert var_11.coerce_types is True
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_12 = module_0.Array(var_9, var_10, **var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Array'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.items).__module__}.{type(var_12.items).__qualname__}' == 'builtins.list'
    assert len(var_12.items) == 1
    assert f'{type(var_12.additional_items).__module__}.{type(var_12.additional_items).__qualname__}' == 'typesystem.fields.Date'
    assert var_12.min_items == 1
    assert var_12.max_items is None
    assert var_12.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_13 = var_4.validate_or_error(var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert f'{type(var_13.error).__module__}.{type(var_13.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13.error) == 1
    var_14 = var_11.validate_or_error(var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert f'{type(var_14.error).__module__}.{type(var_14.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14.error) == 1

def test_case_165():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

@pytest.mark.xfail(strict=True)
def test_case_166():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_3.validate(var_3)

def test_case_167():
    var_0 = module_0.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_1 = module_0.Field()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    var_2 = [var_0, var_1]
    var_3 = 10
    var_4 = {}
    var_5 = module_0.Array(var_2, max_items=var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.items).__module__}.{type(var_5.items).__qualname__}' == 'builtins.list'
    assert len(var_5.items) == 2
    assert var_5.additional_items is False
    assert var_5.min_items == 2
    assert var_5.max_items == 10
    assert var_5.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_6 = var_5.min_items
    assert var_6 == 2
    var_7 = var_5.max_items
    assert var_7 == 10

@pytest.mark.xfail(strict=True)
def test_case_168():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.DateTime()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'datetime'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_4 = None
    var_5 = module_0.Number(coerce_types=var_4, **var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Number'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert var_5.minimum is None
    assert var_5.maximum is None
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of is None
    assert var_5.precision is None
    assert var_5.coerce_types is None
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_6 = None
    var_7 = module_0.Object(min_properties=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is True
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = ''
    var_9 = module_0.Choice()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Choice'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.choices == []
    assert var_9.coerce_types is True
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_10 = var_7.get_default_value()
    var_11 = module_0.Const(var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Const'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.const).__module__}.{type(var_11.const).__qualname__}' == 'typesystem.fields.Number'
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_12 = module_0.String(allow_blank=var_10, trim_whitespace=var_10, pattern=var_10, coerce_types=var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.allow_blank is None
    assert var_12.trim_whitespace is None
    assert var_12.max_length is None
    assert var_12.min_length is None
    assert var_12.format is None
    assert var_12.coerce_types is None
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_13 = var_7.__or__(var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Union'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is True
    assert var_13.read_only is False
    assert f'{type(var_13.any_of).__module__}.{type(var_13.any_of).__qualname__}' == 'builtins.list'
    assert len(var_13.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_14 = module_0.Array(additional_items=var_3, min_items=var_6)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Array'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.items is None
    assert f'{type(var_14.additional_items).__module__}.{type(var_14.additional_items).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_14.min_items is None
    assert var_14.max_items is None
    assert var_14.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_15 = module_0.String(allow_blank=var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.String'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.default == ''
    assert var_15.allow_blank is True
    assert var_15.trim_whitespace is True
    assert var_15.max_length is None
    assert var_15.min_length is None
    assert var_15.format is None
    assert var_15.coerce_types is True
    assert var_15.pattern is None
    assert var_15.pattern_regex is None
    var_16 = module_0.Text()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Text'
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
    var_17 = None
    var_18 = var_11.get_default_value()
    var_19 = var_3.has_default()
    assert var_19 is False
    var_20 = var_3.validate_or_error(var_17)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_20.value is None
    assert f'{type(var_20.error).__module__}.{type(var_20.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_20.error) == 1
    var_5.validate(var_8)
    assert var_21 is None

def test_case_169():
    var_0 = ''
    var_1 = module_0.Text()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Text'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'text'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = None
    var_4 = module_0.Choice(choices=var_3, coerce_types=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Choice'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.choices == []
    assert var_4.coerce_types == ''
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_5 = var_4.validate_or_error(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = var_4.get_default_value()
    var_7 = module_0.UUID()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.UUID'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format == 'uuid'
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    var_8 = {var_0: var_1}
    var_9 = {}
    var_10 = module_0.Object(pattern_properties=var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Object'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.properties == {}
    assert f'{type(var_10.pattern_properties).__module__}.{type(var_10.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_10.pattern_properties) == 1
    assert var_10.additional_properties is True
    assert var_10.property_names is None
    assert var_10.min_properties is None
    assert var_10.max_properties is None
    assert var_10.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_11 = {var_0: var_4, var_0: var_0, var_4: var_4}
    with pytest.raises(module_6.ValidationError):
        var_10.validate(var_11)

@pytest.mark.xfail(strict=True)
def test_case_170():
    var_0 = module_0.Object()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Object'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.properties == {}
    assert var_0.pattern_properties == {}
    assert var_0.additional_properties is True
    assert var_0.property_names is None
    assert var_0.min_properties is None
    assert var_0.max_properties is None
    assert var_0.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = True
    var_2 = None
    var_3 = module_0.String(allow_blank=var_1, coerce_types=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.default == ''
    assert var_3.allow_blank is True
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format is None
    assert var_3.coerce_types is None
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3.validate(var_2)

def test_case_171():
    var_0 = True
    var_1 = 'test'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.default == 'test'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is True
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is None
    assert var_4.format is None
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = var_4.default
    assert var_5 == 'test'

@pytest.mark.xfail(strict=True)
def test_case_172():
    var_0 = None
    var_1 = module_0.Integer(maximum=var_0, exclusive_maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = 2262.21231
    var_1.validate(var_2)