# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_5
import ipaddress as module_4
import platform as module_1
import re as module_3
import uuid as module_2

import pytest
import typesystem.fields as module_0


def test_case_0():
    var_0 = module_0.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7

def test_case_1():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is True
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}

@pytest.mark.xfail(strict=True)
def test_case_2():
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
    module_0.Any(title=var_1, allow_null=var_1)

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
    var_1 = var_0.get_default_value()

def test_case_4():
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

@pytest.mark.xfail(strict=True)
def test_case_5():
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
    var_1 = module_1.machine()
    assert var_1 == 'x86_64'
    var_0.validate(var_1)

def test_case_6():
    var_0 = None
    var_1 = module_0.UUID()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.UUID'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'uuid'
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

def test_case_7():
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
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1

def test_case_8():
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

def test_case_9():
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
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_10():
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

def test_case_11():
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

def test_case_12():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}

def test_case_13():
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
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1

def test_case_14():
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

def test_case_15():
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

def test_case_16():
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

def test_case_17():
    var_0 = None
    var_1 = module_0.Time()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Time'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'time'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_0.Object(property_names=var_0, required=var_0)
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
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_18():
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

def test_case_19():
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

def test_case_20():
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

def test_case_21():
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

def test_case_22():
    var_0 = module_1.machine()
    assert var_0 == 'x86_64'
    var_1 = module_0.String(trim_whitespace=var_0, pattern=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace == 'x86_64'
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern == 'x86_64'
    assert f'{type(var_1.pattern_regex).__module__}.{type(var_1.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}

def test_case_23():
    var_0 = module_1.python_revision()
    assert var_0 == ''
    var_1 = module_0.URL()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.URL'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'url'
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

def test_case_24():
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

def test_case_25():
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

def test_case_26():
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
    var_1 = module_1.machine()
    assert var_1 == 'x86_64'
    var_2 = module_0.String(allow_blank=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.default == ''
    assert var_2.allow_blank == 'x86_64'
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'x86_64'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}

def test_case_27():
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
    var_1 = module_2.getnode()
    assert var_1 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_2 = module_0.Array(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_2.additional_items).__module__}.{type(var_2.additional_items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.min_items == 81579238197203
    assert var_2.max_items is None
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = var_0.validate(var_1)
    assert var_3 == 81579238197203

def test_case_28():
    var_0 = {}
    var_1 = None
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {}
    assert var_3.error is None
    with pytest.raises(AssertionError):
        module_0.Array(additional_items=var_1, min_items=var_1, exact_items=var_1)

def test_case_29():
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
def test_case_30():
    var_0 = None
    var_1 = module_0.URL()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.URL'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'url'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_2.getnode()
    assert var_2 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_3 = var_1.serialize(var_0)
    var_2.validate_or_error(var_0)

def test_case_31():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Integer(maximum=var_0, exclusive_maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum == 81579238197203
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum == 81579238197203
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

def test_case_32():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Integer(minimum=var_0, exclusive_maximum=var_0, precision=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum == 81579238197203
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum == 81579238197203
    assert var_1.multiple_of is None
    assert var_1.precision == 81579238197203
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.serialize(var_0)
    assert var_2 == 81579238197203
    var_3 = var_1.get_default_value()
    var_4 = var_1.validate_or_error(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = module_0.Choice(choices=var_0, coerce_types=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Choice'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.choices == []
    assert var_1.coerce_types is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_1.validate(var_0)

def test_case_34():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = None
    var_1 = module_0.Integer(minimum=var_0, exclusive_minimum=var_0, exclusive_maximum=var_0)
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
    var_1.validate(var_0)

def test_case_36():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Integer(minimum=var_0, multiple_of=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum == 81579238197203
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of == 81579238197203
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == 81579238197203
    assert var_2.error is None

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = None
    var_1 = True
    var_2 = module_0.URL()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.URL'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format == 'url'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_3 = module_0.Decimal(precision=var_0)
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
    var_4 = var_2.validate_or_error(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1
    var_5 = module_0.Integer(exclusive_maximum=var_1, multiple_of=var_1, coerce_types=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Integer'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum is None
    assert var_5.maximum is None
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is True
    assert var_5.multiple_of is True
    assert var_5.precision is None
    assert var_5.coerce_types is None
    var_6 = var_4.__repr__()
    assert var_6 == "ValidationResult(error=ValidationError(text='May not be null.', code='null'))"
    var_5.validate(var_1)

def test_case_38():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)
    var_3 = var_0.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = None
    var_1 = module_0.URL()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.URL'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'url'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_0.Decimal(precision=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Decimal'
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
    var_3 = var_1.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = module_0.Integer()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Integer'
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
    var_5 = module_1.release()
    assert var_5 == '6.17.9-76061709-generic'
    var_6 = var_1.get_default_value()
    var_7 = var_5.__lt__(var_0)
    var_8 = var_7.__hash__()
    assert var_8 == 7726259430041
    var_9 = var_1.serialize(var_7)
    var_10 = module_2.uuid4()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'uuid.UUID'
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.UUID.bytes).__module__}.{type(module_2.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.bytes_le).__module__}.{type(module_2.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.fields).__module__}.{type(module_2.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time_low).__module__}.{type(module_2.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time_mid).__module__}.{type(module_2.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time_hi_version).__module__}.{type(module_2.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.clock_seq_hi_variant).__module__}.{type(module_2.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.clock_seq_low).__module__}.{type(module_2.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time).__module__}.{type(module_2.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.clock_seq).__module__}.{type(module_2.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.node).__module__}.{type(module_2.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.hex).__module__}.{type(module_2.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.urn).__module__}.{type(module_2.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.variant).__module__}.{type(module_2.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.version).__module__}.{type(module_2.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.int).__module__}.{type(module_2.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.UUID.is_safe).__module__}.{type(module_2.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    module_0.Decimal(multiple_of=var_7, coerce_types=var_7)

def test_case_40():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
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
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == 81579238197203
    assert var_2.error is None

def test_case_41():
    var_0 = None
    var_1 = module_0.Decimal(exclusive_minimum=var_0, exclusive_maximum=var_0, multiple_of=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Decimal'
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

def test_case_42():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = var_0.serialize(var_0)
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

def test_case_43():
    var_0 = True
    var_1 = {}
    var_2 = 'nie:e]pYxwLx`'
    var_3 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is True
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_4 = module_0.String(allow_blank=var_2, max_length=var_0, min_length=var_0, **var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.default == ''
    assert var_4.allow_blank == 'nie:e]pYxwLx`'
    assert var_4.trim_whitespace is True
    assert var_4.max_length is True
    assert var_4.min_length is True
    assert var_4.format is None
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_1.processor()
    assert var_5 == ''
    var_6 = var_4.validate_or_error(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1

def test_case_44():
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
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1
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
    var_3 = var_0.get_default_value()
    var_4 = var_0.__lt__(var_3)
    var_5 = var_4.__hash__()
    assert var_5 == 7726259430041
    var_6 = var_4.__hash__()
    assert var_6 == 7726259430041
    var_7 = var_5.__hash__()
    assert var_7 == 7726259430041
    with pytest.raises(AssertionError):
        module_0.String(min_length=var_1, coerce_types=var_5)

def test_case_45():
    var_0 = None
    var_1 = module_0.Decimal(exclusive_minimum=var_0, exclusive_maximum=var_0, multiple_of=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Decimal'
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
    var_2 = var_1.serialize(var_0)

def test_case_46():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = None
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = None
    var_1 = module_0.Decimal(precision=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Decimal'
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
    var_3 = module_1.release()
    assert var_3 == '6.17.9-76061709-generic'
    var_4 = var_3.__lt__(var_0)
    var_5 = var_4.__hash__()
    assert var_5 == 7726259430041
    var_6 = var_1.validate_or_error(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_7 = module_2.uuid4()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'uuid.UUID'
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.UUID.bytes).__module__}.{type(module_2.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.bytes_le).__module__}.{type(module_2.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.fields).__module__}.{type(module_2.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time_low).__module__}.{type(module_2.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time_mid).__module__}.{type(module_2.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time_hi_version).__module__}.{type(module_2.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.clock_seq_hi_variant).__module__}.{type(module_2.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.clock_seq_low).__module__}.{type(module_2.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time).__module__}.{type(module_2.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.clock_seq).__module__}.{type(module_2.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.node).__module__}.{type(module_2.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.hex).__module__}.{type(module_2.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.urn).__module__}.{type(module_2.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.variant).__module__}.{type(module_2.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.version).__module__}.{type(module_2.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.int).__module__}.{type(module_2.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.UUID.is_safe).__module__}.{type(module_2.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    module_0.Decimal(maximum=var_3, precision=var_2, coerce_types=var_7)

def test_case_48():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Integer(exclusive_minimum=var_0, precision=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum == 81579238197203
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision == 81579238197203
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
def test_case_49():
    var_0 = module_1.python_revision()
    assert var_0 == ''
    var_1 = module_0.URL()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.URL'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'url'
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
    module_0.Decimal(minimum=var_0, precision=var_0)

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.URL()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.URL'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'url'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_2.getnode()
    assert var_2 == 81579238197203
    var_3 = var_0.__str__()
    assert var_3 == '81579238197203'
    var_4 = False
    var_5 = module_0.Integer(maximum=var_0, exclusive_minimum=var_4, multiple_of=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Integer'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum is None
    assert var_5.maximum == 81579238197203
    assert var_5.exclusive_minimum is False
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of is False
    assert var_5.precision is None
    assert var_5.coerce_types is True
    var_5.validate_or_error(var_3)

def test_case_51():
    var_0 = module_1.python_revision()
    assert var_0 == ''
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
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_52():
    var_0 = True
    with pytest.raises(AssertionError):
        module_0.String(allow_blank=var_0, format=var_0)

def test_case_53():
    var_0 = None
    var_1 = module_0.Integer(minimum=var_0, exclusive_minimum=var_0, exclusive_maximum=var_0)
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

@pytest.mark.xfail(strict=True)
def test_case_54():
    var_0 = module_1.python_revision()
    assert var_0 == ''
    var_1 = module_0.URL()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.URL'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'url'
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
    module_0.Integer(exclusive_maximum=var_0, precision=var_0, coerce_types=var_0)

@pytest.mark.xfail(strict=True)
def test_case_55():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1
    var_2 = {}
    var_3 = module_0.URL(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.URL'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'url'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    var_4 = None
    var_5 = var_3.validate_or_error(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = True
    module_0.Integer(exclusive_minimum=var_2, precision=var_4, coerce_types=var_6)

@pytest.mark.xfail(strict=True)
def test_case_56():
    var_0 = None
    module_0.Any(description=var_0)

def test_case_57():
    var_0 = False
    var_1 = None
    var_2 = module_0.String(trim_whitespace=var_0, max_length=var_0, min_length=var_1, pattern=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is False
    assert var_2.max_length is False
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}

def test_case_58():
    var_0 = -2739
    var_1 = module_0.Array(max_items=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.items is None
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items == -2739
    assert var_1.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}

def test_case_59():
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
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_60():
    var_0 = None
    var_1 = module_0.Object(max_properties=var_0)
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
    var_1.validate(var_0)

def test_case_61():
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
    var_2 = var_0.serialize(var_1)
    var_3 = module_2.SafeUUID.safe
    var_4 = module_0.Integer()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Integer'
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
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_5 = var_4.validate_or_error(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1

def test_case_62():
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
    var_1 = var_0.__or__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = var_1.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_63():
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
    var_1 = module_2.getnode()
    assert var_1 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_2 = module_0.Integer(exclusive_minimum=var_1, precision=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum == 81579238197203
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert f'{type(var_2.precision).__module__}.{type(var_2.precision).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.coerce_types is True
    var_2.validate_or_error(var_1)

@pytest.mark.xfail(strict=True)
def test_case_64():
    var_0 = None
    var_1 = module_0.URL()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.URL'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'url'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_0.Decimal(precision=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Decimal'
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
    var_3 = var_1.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = module_0.Integer()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Integer'
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
    var_5 = True
    var_6 = module_0.Field(allow_null=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Field'
    assert var_6.default is None
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is True
    assert var_6.read_only is False
    assert module_0.Field.errors == {}
    var_7 = var_3.__repr__()
    assert var_7 == "ValidationResult(error=ValidationError(text='May not be null.', code='null'))"
    var_8 = True
    var_9 = module_0.Integer(exclusive_maximum=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Integer'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.minimum is None
    assert var_9.maximum is None
    assert var_9.exclusive_minimum is None
    assert var_9.exclusive_maximum is True
    assert var_9.multiple_of is None
    assert var_9.precision is None
    assert var_9.coerce_types is True
    var_6.validate_or_error(var_0)

def test_case_65():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Integer(exclusive_minimum=var_0, precision=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum == 81579238197203
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision == 81579238197203
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    with pytest.raises(AssertionError):
        module_0.Array(var_0, var_0)

def test_case_66():
    var_0 = -97
    var_1 = module_0.Array(min_items=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.items is None
    assert var_1.additional_items is False
    assert var_1.min_items == -97
    assert var_1.max_items is None
    assert var_1.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}

def test_case_67():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = None
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = module_0.Array(var_0, min_items=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.additional_items is False
    assert var_3.min_items is None
    assert var_3.max_items is None
    assert var_3.unique_items is False

def test_case_68():
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
    var_1 = module_0.Object(additional_properties=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert f'{type(var_1.additional_properties).__module__}.{type(var_1.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_69():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    with pytest.raises(AssertionError):
        module_0.Object(additional_properties=var_0, required=var_0)

def test_case_70():
    var_0 = module_1.freedesktop_os_release()
    with pytest.raises(AssertionError):
        module_0.Object(pattern_properties=var_0, required=var_0)

def test_case_71():
    var_0 = True
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
    assert var_2.min_properties is True
    assert var_2.max_properties is None
    assert var_2.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_72():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = None
    var_2 = module_0.Object(additional_properties=var_1, property_names=var_0, max_properties=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert var_2.additional_properties is None
    assert var_2.property_names == 81579238197203
    assert var_2.min_properties is None
    assert var_2.max_properties == 81579238197203
    assert var_2.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_0.validate_or_error(var_1)

def test_case_73():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1
    var_2 = module_2.getnode()
    assert var_2 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_3 = False
    var_4 = module_0.Integer(exclusive_maximum=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Integer'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum is None
    assert var_4.maximum is None
    assert var_4.exclusive_minimum is None
    assert var_4.exclusive_maximum is False
    assert var_4.multiple_of is None
    assert var_4.precision is None
    assert var_4.coerce_types is True
    with pytest.raises(AssertionError):
        module_0.Array(exact_items=var_3, unique_items=var_2)

def test_case_74():
    var_0 = {}
    var_1 = None
    var_2 = module_0.Object(properties=var_0, additional_properties=var_1)
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_75():
    var_0 = None
    var_1 = module_0.Integer(precision=var_0, multiple_of=var_0, coerce_types=var_0)
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
    assert var_1.coerce_types is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_76():
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
    var_1 = module_1.machine()
    assert var_1 == 'x86_64'
    var_2 = module_0.Object(properties=var_0, required=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert f'{type(var_2.additional_properties).__module__}.{type(var_2.additional_properties).__qualname__}' == 'typesystem.fields.URL'
    assert var_2.property_names is None
    assert var_2.min_properties is None
    assert var_2.max_properties is None
    assert var_2.required == 'x86_64'
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1.validate_or_error(var_1)

def test_case_77():
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

def test_case_78():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    with pytest.raises(AssertionError):
        module_0.Object(min_properties=var_0, required=var_0)

def test_case_79():
    var_0 = module_1.python_revision()
    assert var_0 == ''
    with pytest.raises(AssertionError):
        module_0.Object(property_names=var_0, max_properties=var_0)

def test_case_80():
    var_0 = module_1.python_build()
    with pytest.raises(AssertionError):
        module_0.Array(var_0)

def test_case_81():
    var_0 = False
    var_1 = module_0.Boolean(coerce_types=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is False
    assert var_2.error is None

def test_case_82():
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
    var_2 = var_0.__or__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 3

@pytest.mark.xfail(strict=True)
def test_case_83():
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
    var_2 = module_0.Any(read_only=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is None
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = None
    module_3.match(var_4, var_4)

def test_case_84():
    var_0 = module_1.python_revision()
    assert var_0 == ''
    with pytest.raises(AssertionError):
        module_0.Array(max_items=var_0)

def test_case_85():
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
    var_2 = var_1.__or__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.any_of == []

@pytest.mark.xfail(strict=True)
def test_case_86():
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
    var_2 = var_1.validate(var_0)
    var_3 = module_0.Field()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert module_0.Field.errors == {}
    var_4 = None
    var_3.validation_error(var_4)

def test_case_87():
    var_0 = True
    var_1 = {}
    var_2 = 'nie:e]pYxwLx`'
    var_3 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is True
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_4 = {var_2: var_3, var_2: var_3, var_2: var_3, var_2: var_3}
    var_5 = None
    var_6 = module_0.Object(properties=var_4, pattern_properties=var_4, required=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.properties).__module__}.{type(var_6.properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.properties) == 1
    assert f'{type(var_6.pattern_properties).__module__}.{type(var_6.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.pattern_properties) == 1
    assert var_6.additional_properties is True
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = var_3.validate_or_error(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1

def test_case_88():
    var_0 = None
    var_1 = module_1.python_implementation()
    assert var_1 == 'CPython'
    var_2 = module_0.Const(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Const'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.const is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_3 = module_2.getnode()
    assert var_3 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_4 = var_2.validate_or_error(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_89():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Any(default=var_0, allow_null=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.default == 81579238197203
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null == 81579238197203
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_2.getnode()
    assert var_2 == 81579238197203
    var_3 = None
    var_4 = None
    var_5 = module_0.Integer(minimum=var_0, maximum=var_4, exclusive_minimum=var_2, exclusive_maximum=var_2, precision=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Integer'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum == 81579238197203
    assert var_5.maximum is None
    assert var_5.exclusive_minimum == 81579238197203
    assert var_5.exclusive_maximum == 81579238197203
    assert var_5.multiple_of is None
    assert var_5.precision is None
    assert var_5.coerce_types is True
    with pytest.raises(ValueError):
        module_4.ip_interface(var_3)

def test_case_90():
    var_0 = {}
    var_1 = None
    var_2 = module_0.Object(min_properties=var_1, **var_0)
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {}
    assert var_3.error is None

def test_case_91():
    var_0 = False
    var_1 = module_5._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
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
    assert var_2.min_properties is False
    assert var_2.max_properties is None
    assert var_2.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {}
    assert var_3.error is None

def test_case_92():
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
    var_1 = module_2.getnode()
    assert var_1 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_2 = module_0.Integer(minimum=var_1, multiple_of=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum == 81579238197203
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of == 81579238197203
    assert var_2.precision is None
    assert var_2.coerce_types is True
    var_3 = module_0.Number(exclusive_minimum=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Number'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum == 81579238197203
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4 = var_3.validate_or_error(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_93():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    module_0.Choice(choices=var_0)

def test_case_94():
    var_0 = {}
    var_1 = None
    var_2 = module_0.Object(additional_properties=var_1, property_names=var_1, min_properties=var_1, **var_0)
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {}
    assert var_3.error is None

def test_case_95():
    var_0 = 'key1'
    var_1 = (var_0, var_0)
    var_2 = [var_1, var_1]
    var_3 = True
    var_4 = module_0.Choice(choices=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Choice'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.choices == [('key1', 'key1'), ('key1', 'key1')]
    assert var_4.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}

@pytest.mark.xfail(strict=True)
def test_case_96():
    var_0 = module_1.python_implementation()
    assert var_0 == 'CPython'
    var_1 = module_0.URL()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.URL'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'url'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Choice'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.choices == []
    assert var_2.coerce_types is True
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_2.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_97():
    var_0 = 'k7y1'
    var_1 = (var_0, var_0)
    var_2 = 'value2'
    var_3 = [var_1, var_1]
    var_4 = True
    var_5 = module_0.Choice(choices=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Choice'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.choices == [('k7y1', 'k7y1'), ('k7y1', 'k7y1')]
    assert var_5.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_5.validate_or_error(var_2)

def test_case_98():
    var_0 = module_1.processor()
    assert var_0 == ''
    var_1 = [var_0, var_0]
    var_2 = True
    var_3 = module_0.Choice(choices=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Choice'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.choices == [('', ''), ('', '')]
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}

def test_case_99():
    var_0 = module_1.freedesktop_os_release()
    var_1 = None
    var_2 = module_0.Object(pattern_properties=var_1, required=var_1)
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.2', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert var_3.error is None

def test_case_100():
    var_0 = None
    var_1 = module_0.Object(additional_properties=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is None
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_1.freedesktop_os_release()
    var_3 = var_1.validate_or_error(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {}
    assert var_3.error is None

def test_case_101():
    var_0 = False
    var_1 = module_5._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = True
    var_3 = module_0.String(allow_blank=var_2, coerce_types=var_0)
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
    assert var_3.coerce_types is False
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = None
    var_5 = var_3.validate_or_error(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1

def test_case_102():
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
    var_1 = module_1.system()
    assert var_1 == 'Linux'
    with pytest.raises(AssertionError):
        module_0.Array(var_0, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_103():
    var_0 = {}
    var_1 = module_0.Const(var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Const'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.const == {}
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_2 = None
    var_1.validate(var_2)

def test_case_104():
    var_0 = True
    var_1 = module_5._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = True
    var_3 = module_0.String(allow_blank=var_2, coerce_types=var_0)
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
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = None
    var_5 = var_3.validate_or_error(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value == ''
    assert var_5.error is None

def test_case_105():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is False
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_3 = 'y}nE'
    var_4 = {var_3: var_2, var_3: var_2, var_3: var_2, var_3: var_2}
    var_5 = None
    var_6 = module_0.Object(properties=var_4, pattern_properties=var_4, required=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.properties).__module__}.{type(var_6.properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.properties) == 1
    assert f'{type(var_6.pattern_properties).__module__}.{type(var_6.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.pattern_properties) == 1
    assert var_6.additional_properties is True
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = var_6.validate_or_error(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1

def test_case_106():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is True
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_3 = 'y(nE'
    var_4 = {var_3: var_2, var_3: var_2, var_3: var_2}
    var_5 = None
    var_6 = module_0.Object(properties=var_4, pattern_properties=var_4, required=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.properties).__module__}.{type(var_6.properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.properties) == 1
    assert f'{type(var_6.pattern_properties).__module__}.{type(var_6.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.pattern_properties) == 1
    assert var_6.additional_properties is True
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = var_6.validate_or_error(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value == {}
    assert var_7.error is None

def test_case_107():
    var_0 = module_2.getnode()
    assert var_0 == 81579238197203
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_5._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    with pytest.raises(AssertionError):
        module_0.String(pattern=var_0, format=var_0, **var_1)

def test_case_108():
    var_0 = module_5._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = "|7L.N84: VtAbzm'x\x0c"
    var_2 = module_3.purge()
    assert module_3.ASCII == module_3.RegexFlag.ASCII
    assert module_3.A == module_3.RegexFlag.ASCII
    assert module_3.IGNORECASE == module_3.RegexFlag.IGNORECASE
    assert module_3.I == module_3.RegexFlag.IGNORECASE
    assert module_3.LOCALE == module_3.RegexFlag.LOCALE
    assert module_3.L == module_3.RegexFlag.LOCALE
    assert module_3.UNICODE == module_3.RegexFlag.UNICODE
    assert module_3.U == module_3.RegexFlag.UNICODE
    assert module_3.MULTILINE == module_3.RegexFlag.MULTILINE
    assert module_3.M == module_3.RegexFlag.MULTILINE
    assert module_3.DOTALL == module_3.RegexFlag.DOTALL
    assert module_3.S == module_3.RegexFlag.DOTALL
    assert module_3.VERBOSE == module_3.RegexFlag.VERBOSE
    assert module_3.X == module_3.RegexFlag.VERBOSE
    assert module_3.TEMPLATE == module_3.RegexFlag.TEMPLATE
    assert module_3.T == module_3.RegexFlag.TEMPLATE
    assert module_3.DEBUG == module_3.RegexFlag.DEBUG
    var_3 = 'y}nE'
    var_4 = {var_1: var_2, var_1: var_2, var_1: var_2, var_3: var_2}
    var_5 = None
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_4, pattern_properties=var_4, required=var_5)

def test_case_109():
    var_0 = None
    var_1 = False
    var_2 = module_0.Object(additional_properties=var_1, min_properties=var_0, required=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert var_2.additional_properties is False
    assert var_2.property_names is None
    assert var_2.min_properties is None
    assert var_2.max_properties is None
    assert var_2.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = module_1.freedesktop_os_release()
    var_4 = var_2.validate_or_error(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 10

def test_case_110():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is False
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_3 = 'T\nt"+w=i6\x0cd:'
    var_4 = {var_3: var_2, var_3: var_2, var_3: var_2, var_3: var_2}
    var_5 = None
    var_6 = module_0.Object(properties=var_4, pattern_properties=var_4, required=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.properties).__module__}.{type(var_6.properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.properties) == 1
    assert f'{type(var_6.pattern_properties).__module__}.{type(var_6.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.pattern_properties) == 1
    assert var_6.additional_properties is True
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = var_6.validate_or_error(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_111():
    var_0 = None
    var_1 = module_0.Object(additional_properties=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is None
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_1.freedesktop_os_release()
    var_3 = module_0.Object(property_names=var_2, required=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is True
    assert var_3.property_names == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.2', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.2', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    var_3.validate_or_error(var_2)

def test_case_112():
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
    var_1 = module_1.machine()
    assert var_1 == 'x86_64'
    with pytest.raises(AssertionError):
        module_0.String(max_length=var_1, min_length=var_1, pattern=var_1, coerce_types=var_1)

def test_case_113():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is False
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_3 = {var_2: var_2, var_2: var_2, var_2: var_2, var_2: var_2}
    var_4 = None
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_3, pattern_properties=var_3, required=var_4)

def test_case_114():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, required=var_1)
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
    assert var_2.max_properties is False
    assert var_2.required == {}
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {}
    assert var_3.error is None

def test_case_115():
    var_0 = True
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
    assert var_2.min_properties is True
    assert var_2.max_properties is None
    assert var_2.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = None
    var_4 = module_0.Object(properties=var_3, additional_properties=var_2, max_properties=var_3, required=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert var_4.pattern_properties == {}
    assert f'{type(var_4.additional_properties).__module__}.{type(var_4.additional_properties).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == []
    var_5 = var_4.validate_or_error(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value == {}
    assert var_5.error is None

@pytest.mark.xfail(strict=True)
def test_case_116():
    var_0 = False
    var_1 = module_0.Array(exact_items=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.items is None
    assert var_1.additional_items is False
    assert var_1.min_items is False
    assert var_1.max_items is False
    assert var_1.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_5._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = None
    var_2.__setitem__(var_3, var_3)

def test_case_117():
    var_0 = False
    var_1 = module_1.uname()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'platform.uname_result'
    assert len(var_1) == 6
    assert f'{type(module_1.uname_result.processor).__module__}.{type(module_1.uname_result.processor).__qualname__}' == 'functools.cached_property'
    assert module_1.uname_result.processor.attrname == 'processor'
    assert f'{type(module_1.uname_result.processor.lock).__module__}.{type(module_1.uname_result.processor.lock).__qualname__}' == '_thread.RLock'
    var_2 = module_0.Object(max_properties=var_0, required=var_1)
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
    assert var_2.max_properties is False
    assert var_2.required == ['Linux', 'cb60a55e20cb', '6.17.9-76061709-generic', '#202511241048~1764607909~24.04~df6b2b6 SMP PREEMPT_DYNAMIC Mon D', 'x86_64', '']
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = None
    var_4 = None
    var_5 = None
    var_6 = module_0.Object(properties=var_3, property_names=var_2, max_properties=var_4, required=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.properties == {}
    assert var_6.pattern_properties == {}
    assert var_6.additional_properties is True
    assert f'{type(var_6.property_names).__module__}.{type(var_6.property_names).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == []
    var_7 = var_2.validate_or_error(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_118():
    var_0 = module_0.Decimal()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Decimal'
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
    var_0.serialize(var_0)
    assert var_1 is None

@pytest.mark.xfail(strict=True)
def test_case_119():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = module_0.String()
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
    var_4 = module_0.Integer()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Integer'
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
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
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
    var_7 = 'hello'
    var_8 = 123
    var_9 = [var_7, var_8]
    var_10 = var_6.serialize(var_9)
    var_11 = module_0.String()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.String'
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
    var_12 = module_0.Array(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Array'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.items).__module__}.{type(var_12.items).__qualname__}' == 'typesystem.fields.String'
    assert var_12.additional_items is False
    assert var_12.min_items is None
    assert var_12.max_items is None
    assert var_12.unique_items is False
    module_0.Integer(minimum=var_1, maximum=var_1, exclusive_minimum=var_10, coerce_types=var_10)

@pytest.mark.xfail(strict=True)
def test_case_120():
    var_0 = module_0.Array()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Array'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.items is None
    assert var_0.additional_items is False
    assert var_0.min_items is None
    assert var_0.max_items is None
    assert var_0.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = module_0.String()
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
    var_4 = module_0.Integer()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Integer'
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
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
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
    var_7 = 'hello'
    var_8 = 123
    var_9 = [var_7, var_8]
    var_10 = var_6.serialize(var_9)
    var_11 = module_0.String()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.String'
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
    var_12 = module_0.Array(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Array'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.items).__module__}.{type(var_12.items).__qualname__}' == 'typesystem.fields.String'
    assert var_12.additional_items is False
    assert var_12.min_items is None
    assert var_12.max_items is None
    assert var_12.unique_items is False
    var_13 = 'world'
    var_14 = [var_7, var_13]
    var_15 = var_12.serialize(var_14)
    var_16 = module_0.Array()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Array'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.items is None
    assert var_16.additional_items is False
    assert var_16.min_items is None
    assert var_16.max_items is None
    assert var_16.unique_items is False
    var_17 = [var_7, var_8]
    var_18 = var_16.serialize(var_17)
    var_19 = module_0.String()
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
    var_20 = module_0.Integer()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Integer'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.minimum is None
    assert var_20.maximum is None
    assert var_20.exclusive_minimum is None
    assert var_20.exclusive_maximum is None
    assert var_20.multiple_of is None
    assert var_20.precision is None
    assert var_20.coerce_types is True
    var_21 = [var_19, var_20]
    var_22 = module_0.Array(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Array'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.items).__module__}.{type(var_22.items).__qualname__}' == 'builtins.list'
    assert len(var_22.items) == 2
    assert var_22.additional_items is False
    assert var_22.min_items == 2
    assert var_22.max_items == 2
    assert var_22.unique_items is False
    var_23 = 'extra'
    var_24 = [var_7, var_8, var_23]
    var_25 = var_22.serialize(var_24)
    var_26 = module_0.String()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.String'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.allow_blank is False
    assert var_26.trim_whitespace is True
    assert var_26.max_length is None
    assert var_26.min_length is None
    assert var_26.format is None
    assert var_26.coerce_types is True
    assert var_26.pattern is None
    assert var_26.pattern_regex is None
    var_27 = module_0.Integer()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Integer'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.minimum is None
    assert var_27.maximum is None
    assert var_27.exclusive_minimum is None
    assert var_27.exclusive_maximum is None
    assert var_27.multiple_of is None
    assert var_27.precision is None
    assert var_27.coerce_types is True
    var_28 = [var_26, var_27]
    var_29 = module_0.Array(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Array'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.items).__module__}.{type(var_29.items).__qualname__}' == 'builtins.list'
    assert len(var_29.items) == 2
    assert var_29.additional_items is False
    assert var_29.min_items == 2
    assert var_29.max_items == 2
    assert var_29.unique_items is False
    var_30 = [var_7]
    var_31 = var_29.serialize(var_30)
    var_32 = module_0.String()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.String'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.allow_blank is False
    assert var_32.trim_whitespace is True
    assert var_32.max_length is None
    assert var_32.min_length is None
    assert var_32.format is None
    assert var_32.coerce_types is True
    assert var_32.pattern is None
    assert var_32.pattern_regex is None
    var_33 = module_0.Integer()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Integer'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.minimum is None
    assert var_33.maximum is None
    assert var_33.exclusive_minimum is None
    assert var_33.exclusive_maximum is None
    assert var_33.multiple_of is None
    assert var_33.precision is None
    assert var_33.coerce_types is True
    var_34 = [var_32, var_33]
    var_35 = module_0.Array(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Array'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.items).__module__}.{type(var_35.items).__qualname__}' == 'builtins.list'
    assert len(var_35.items) == 2
    assert var_35.additional_items is False
    assert var_35.min_items == 2
    assert var_35.max_items == 2
    assert var_35.unique_items is False
    var_36 = []
    var_37 = var_35.serialize(var_36)
    var_38 = module_0.String()
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.String'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.allow_blank is False
    assert var_38.trim_whitespace is True
    assert var_38.max_length is None
    assert var_38.min_length is None
    assert var_38.format is None
    assert var_38.coerce_types is True
    assert var_38.pattern is None
    assert var_38.pattern_regex is None
    var_39 = module_0.Array(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Array'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.items).__module__}.{type(var_39.items).__qualname__}' == 'typesystem.fields.String'
    assert var_39.additional_items is False
    assert var_39.min_items is None
    assert var_39.max_items is None
    assert var_39.unique_items is False
    var_40 = []
    var_41 = var_39.serialize(var_40)
    var_42 = module_0.Array()
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.Array'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert var_42.items is None
    assert var_42.additional_items is False
    assert var_42.min_items is None
    assert var_42.max_items is None
    assert var_42.unique_items is False
    var_43 = var_42.serialize(var_17)
    var_44 = module_0.String()
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.String'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert var_44.allow_blank is False
    assert var_44.trim_whitespace is True
    assert var_44.max_length is None
    assert var_44.min_length is None
    assert var_44.format is None
    assert var_44.coerce_types is True
    assert var_44.pattern is None
    assert var_44.pattern_regex is None
    var_45 = module_0.Integer()
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Integer'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.minimum is None
    assert var_45.maximum is None
    assert var_45.exclusive_minimum is None
    assert var_45.exclusive_maximum is None
    assert var_45.multiple_of is None
    assert var_45.precision is None
    assert var_45.coerce_types is True
    var_46 = [var_44, var_45]
    var_47 = module_0.Array(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.Array'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert f'{type(var_47.items).__module__}.{type(var_47.items).__qualname__}' == 'builtins.list'
    assert len(var_47.items) == 2
    assert var_47.additional_items is False
    assert var_47.min_items == 2
    assert var_47.max_items == 2
    assert var_47.unique_items is False
    var_48 = [var_1, var_8]
    var_49 = var_47.serialize(var_48)
    var_50 = module_0.String()
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.fields.String'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert var_50.allow_blank is False
    assert var_50.trim_whitespace is True
    assert var_50.max_length is None
    assert var_50.min_length is None
    assert var_50.format is None
    assert var_50.coerce_types is True
    assert var_50.pattern is None
    assert var_50.pattern_regex is None
    var_51 = var_49.clear()
    var_52 = module_0.Array()
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Array'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert var_52.items is None
    assert var_52.additional_items is False
    assert var_52.min_items is None
    assert var_52.max_items is None
    assert var_52.unique_items is False
    var_53 = [var_1, var_8]
    var_54 = var_52.serialize(var_53)
    var_55 = module_0.String()
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.String'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.allow_blank is False
    assert var_55.trim_whitespace is True
    assert var_55.max_length is None
    assert var_55.min_length is None
    assert var_55.format is None
    assert var_55.coerce_types is True
    assert var_55.pattern is None
    assert var_55.pattern_regex is None
    var_56 = module_0.Array(var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Array'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.items).__module__}.{type(var_56.items).__qualname__}' == 'typesystem.fields.String'
    assert var_56.additional_items is False
    assert var_56.min_items is None
    assert var_56.max_items is None
    assert var_56.unique_items is False
    var_57 = [var_56]
    var_58 = module_0.Array(var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.Array'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert f'{type(var_58.items).__module__}.{type(var_58.items).__qualname__}' == 'builtins.list'
    assert len(var_58.items) == 1
    assert var_58.additional_items is False
    assert var_58.min_items == 1
    assert var_58.max_items == 1
    assert var_58.unique_items is False
    var_59 = module_0.String()
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.fields.String'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert var_59.allow_blank is False
    assert var_59.trim_whitespace is True
    assert var_59.max_length is None
    assert var_59.min_length is None
    assert var_59.format is None
    assert var_59.coerce_types is True
    assert var_59.pattern is None
    assert var_59.pattern_regex is None
    var_60 = module_0.Array(var_59)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.Array'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert f'{type(var_60.items).__module__}.{type(var_60.items).__qualname__}' == 'typesystem.fields.String'
    assert var_60.additional_items is False
    assert var_60.min_items is None
    assert var_60.max_items is None
    assert var_60.unique_items is False
    var_61 = module_0.Array(var_60)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.fields.Array'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert f'{type(var_61.items).__module__}.{type(var_61.items).__qualname__}' == 'typesystem.fields.Array'
    assert var_61.additional_items is False
    assert var_61.min_items is None
    assert var_61.max_items is None
    assert var_61.unique_items is False
    var_62 = [var_7, var_13]
    var_63 = [var_62]
    var_64 = var_61.serialize(var_63)
    var_65 = module_0.Array()
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Array'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.items is None
    assert var_65.additional_items is False
    assert var_65.min_items is None
    assert var_65.max_items is None
    assert var_65.unique_items is False
    var_66 = [var_7, var_13]
    var_67 = [var_66]
    var_68 = var_65.serialize(var_67)
    var_69 = 'key'
    var_70 = module_0.String()
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.fields.String'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert var_70.allow_blank is False
    assert var_70.trim_whitespace is True
    assert var_70.max_length is None
    assert var_70.min_length is None
    assert var_70.format is None
    assert var_70.coerce_types is True
    assert var_70.pattern is None
    assert var_70.pattern_regex is None
    var_71 = {var_69: var_70}
    var_72 = module_0.Object(properties=var_71)
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.Object'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert f'{type(var_72.properties).__module__}.{type(var_72.properties).__qualname__}' == 'builtins.dict'
    assert len(var_72.properties) == 1
    assert var_72.pattern_properties == {}
    assert var_72.additional_properties is True
    assert var_72.property_names is None
    assert var_72.min_properties is None
    assert var_72.max_properties is None
    assert var_72.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_73 = {var_69: var_41}
    var_74 = [var_73]
    var_75 = var_47.serialize(var_74)
    var_76 = module_0.String()
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.fields.String'
    assert var_76.title == ''
    assert var_76.description == ''
    assert var_76.allow_null is False
    assert var_76.read_only is False
    assert var_76.allow_blank is False
    assert var_76.trim_whitespace is True
    assert var_76.max_length is None
    assert var_76.min_length is None
    assert var_76.format is None
    assert var_76.coerce_types is True
    assert var_76.pattern is None
    assert var_76.pattern_regex is None
    var_77 = {var_69: var_76}
    var_78 = module_0.Object(properties=var_77)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.Object'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert f'{type(var_78.properties).__module__}.{type(var_78.properties).__qualname__}' == 'builtins.dict'
    assert len(var_78.properties) == 1
    assert var_78.pattern_properties == {}
    assert var_78.additional_properties is True
    assert var_78.property_names is None
    assert var_78.min_properties is None
    assert var_78.max_properties is None
    assert var_78.required == []
    var_79 = module_0.Array(var_78)
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.fields.Array'
    assert var_79.title == ''
    assert var_79.description == ''
    assert var_79.allow_null is False
    assert var_79.read_only is False
    assert f'{type(var_79.items).__module__}.{type(var_79.items).__qualname__}' == 'typesystem.fields.Object'
    assert var_79.additional_items is False
    assert var_79.min_items is None
    assert var_79.max_items is None
    assert var_79.unique_items is False
    var_80 = [var_2]
    var_81 = var_79.serialize(var_80)
    var_82 = {var_69: var_68}
    var_83 = [var_82]
    var_25.serialize(var_83)