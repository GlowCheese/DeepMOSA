# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.fields as module_0
import platform as module_1
import uuid as module_2
import re as module_3
import decimal as module_4
import typesystem.base as module_5

def test_case_0():
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
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}

def test_case_1():
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

def test_case_2():
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
    var_1 = var_0.get_default_value()

def test_case_3():
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

def test_case_4():
    var_0 = module_1.python_branch()
    assert var_0 == ''
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
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_5():
    var_0 = module_1.python_compiler()
    assert var_0 == 'GCC 14.2.0'
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
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == 'GCC 14.2.0'
    assert var_2.error is None

def test_case_6():
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
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_7():
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

def test_case_8():
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
    var_1 = module_1.python_implementation()
    assert var_1 == 'CPython'
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_9():
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
    var_1 = var_0.__or__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = None
    var_3 = var_0.serialize(var_2)
    var_4 = var_1.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_10():
    var_0 = True
    var_1 = None
    var_2 = module_0.String(allow_blank=var_0, max_length=var_1, format=var_1, coerce_types=var_1)
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
    assert var_2.coerce_types is None
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = var_2.serialize(var_2)
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

def test_case_11():
    var_0 = module_0.Float()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Float'
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

def test_case_12():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Integer(exclusive_maximum=var_0, precision=var_0, multiple_of=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum == 204875749604013
    assert var_1.multiple_of == 204875749604013
    assert var_1.precision == 204875749604013
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_13():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
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
    assert var_1.minimum == 204875749604013
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of == 204875749604013
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate(var_0)
    assert var_2 == 204875749604013

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = -284
    var_1 = module_0.Date()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Date'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'date'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
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
    var_3 = None
    var_4 = module_0.Number(minimum=var_0, exclusive_maximum=var_0, precision=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Number'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum == -284
    assert var_4.maximum is None
    assert var_4.exclusive_minimum is None
    assert var_4.exclusive_maximum == -284
    assert var_4.multiple_of is None
    assert var_4.precision is None
    assert var_4.coerce_types is True
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_5 = var_1.has_default()
    assert var_5 is False
    var_4.validate(var_1)

def test_case_15():
    var_0 = module_0.Float()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Float'
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

def test_case_16():
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

@pytest.mark.xfail(strict=True)
def test_case_17():
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
    var_1 = None
    var_0.validate(var_1)

def test_case_18():
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

def test_case_19():
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

def test_case_20():
    var_0 = module_1.machine()
    assert var_0 == 'x86_64'
    with pytest.raises(AssertionError):
        module_0.Array(max_items=var_0)

def test_case_21():
    var_0 = module_3.purge()
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

def test_case_22():
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

def test_case_23():
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

def test_case_24():
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

def test_case_25():
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
    var_1 = None
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_26():
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

def test_case_27():
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

def test_case_28():
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

def test_case_29():
    var_0 = module_1.python_version()
    assert var_0 == '3.10.19'
    var_1 = module_0.String(pattern=var_0, coerce_types=var_0)
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
    assert var_1.coerce_types == '3.10.19'
    assert var_1.pattern == '3.10.19'
    assert f'{type(var_1.pattern_regex).__module__}.{type(var_1.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}

def test_case_30():
    var_0 = -1288
    var_1 = None
    with pytest.raises(AssertionError):
        module_0.Array(var_1, max_items=var_0, exact_items=var_1, unique_items=var_1)

def test_case_31():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    with pytest.raises(AssertionError):
        module_0.Array(min_items=var_0, unique_items=var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
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
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_33():
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
    var_1 = None
    var_2 = module_1.processor()
    assert var_2 == ''
    var_3 = var_0.validate_or_error(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = module_0.Text()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Text'
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
    module_0.Any(description=var_1)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = module_0.Field(default=var_0, read_only=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_2 = 'J4e7[qW{;'
    var_3 = var_1.get_default_value()
    var_1.validation_error(var_2)

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

def test_case_36():
    var_0 = ']gree{n'
    var_1 = [var_0]
    var_2 = module_0.Choice(choices=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Choice'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.choices == [(']gree{n', ']gree{n')]
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_3 = var_2.validate(var_0)
    assert var_3 == ']gree{n'

def test_case_37():
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
    var_1 = module_1.python_branch()
    assert var_1 == ''
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = var_0.serialize(var_1)
    assert var_3 == ''

def test_case_38():
    var_0 = '#I'
    var_1 = module_0.String(allow_blank=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.default == ''
    assert var_1.allow_blank == '#I'
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

def test_case_39():
    var_0 = True
    var_1 = None
    var_2 = module_0.String(allow_blank=var_0, max_length=var_1, format=var_1, coerce_types=var_1)
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
    assert var_2.coerce_types is None
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    with pytest.raises(AssertionError):
        module_0.Field(title=var_1, default=var_1)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = None
    var_1 = module_0.Float(exclusive_maximum=var_0, precision=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Float'
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

def test_case_41():
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
    var_2 = var_0.__or__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
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
    assert var_1.maximum == 204875749604013
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum == 204875749604013
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1.validate(var_0)

def test_case_43():
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
    var_1 = True
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = None
    var_4 = var_0.validate_or_error(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_44():
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
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_45():
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
    var_1 = {}
    var_2 = None
    var_3 = module_0.Number(maximum=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Number'
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
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4 = module_0.URL(**var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.URL'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is None
    assert var_4.format == 'url'
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    var_5 = None
    var_6 = var_4.serialize(var_5)
    var_7 = module_0.IPAddress()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format == 'ipaddress'
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    var_8 = module_0.Text()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Text'
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
    var_9 = module_0.Any(default=var_2, read_only=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Any'
    assert var_9.default is None
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is None
    var_10 = var_0.serialize(var_5)
    var_11 = var_9.validate_or_error(var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    module_4.ConversionSyntax(*var_6)

@pytest.mark.xfail(strict=True)
def test_case_46():
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
    var_0.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_47():
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
    var_1.validate(var_0)

def test_case_48():
    var_0 = module_0.String()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = var_0.__or__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = [var_0]
    var_3 = var_1.__or__(var_1)
    assert len(var_1.any_of) == 4
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 4
    var_4 = var_3.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1
    var_5 = var_3.get_default_value()
    var_6 = None
    with pytest.raises(module_5.ValidationError):
        var_3.validate(var_6)

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Integer(minimum=var_0, exclusive_minimum=var_0, exclusive_maximum=var_0, coerce_types=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum == 204875749604013
    assert var_1.maximum is None
    assert var_1.exclusive_minimum == 204875749604013
    assert var_1.exclusive_maximum == 204875749604013
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types == 204875749604013
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_50():
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
    var_1 = module_1.python_revision()
    assert var_1 == ''
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    module_0.Float(minimum=var_1, precision=var_1, multiple_of=var_1, coerce_types=var_1)

def test_case_51():
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
def test_case_52():
    var_0 = None
    var_1 = module_1.python_build()
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
    var_3 = module_1.python_version_tuple()
    var_4 = 'z4X I\\@`0. '
    var_5 = module_0.Any(description=var_4, default=var_0, allow_null=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == 'z4X I\\@`0. '
    assert var_5.allow_null == ('main', 'Jan 13 2026 03:14:08')
    assert var_5.read_only is False
    var_2.validate(var_3)

def test_case_53():
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
    var_1 = 'GZKp@gX@'
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
    var_3 = {var_1: var_1, var_1: var_1, var_1: var_2, var_1: var_1, var_1: var_2, var_1: var_2, var_1: var_2}
    var_4 = module_0.Object(pattern_properties=var_3)
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
    var_5 = module_0.Field(allow_null=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Field'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null == 'GZKp@gX@'
    assert var_5.read_only is False
    assert module_0.Field.errors == {}
    var_6 = var_4.validate_or_error(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1

def test_case_54():
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

def test_case_55():
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
    var_1 = True
    var_2 = None
    var_3 = module_0.String(allow_blank=var_1, min_length=var_1, pattern=var_2)
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = var_0.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_56():
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

def test_case_57():
    var_0 = module_1.python_version_tuple()
    var_1 = module_0.Date()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Date'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'date'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
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
    with pytest.raises(AssertionError):
        module_0.String(min_length=var_0)

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
    var_1 = module_0.Boolean()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.coerce_types is True
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_2 = [var_0]
    var_3 = module_0.Array(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'builtins.list'
    assert len(var_3.items) == 1
    assert f'{type(var_3.additional_items).__module__}.{type(var_3.additional_items).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.min_items == 1
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}

@pytest.mark.xfail(strict=True)
def test_case_60():
    var_0 = None
    var_1 = module_0.Integer(minimum=var_0, maximum=var_0, precision=var_0)
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
    var_3 = module_0.Date()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Date'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'date'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    var_4 = var_3.__or__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = var_3.has_default()
    assert var_5 is False
    var_6 = var_4.serialize(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Union'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.any_of).__module__}.{type(var_6.any_of).__qualname__}' == 'builtins.list'
    assert len(var_6.any_of) == 2
    module_0.Float(maximum=var_6, precision=var_6)

@pytest.mark.xfail(strict=True)
def test_case_61():
    var_0 = None
    var_1 = False
    var_2 = module_0.Array(max_items=var_0, unique_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items is None
    assert var_2.max_items is None
    assert var_2.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = module_0.Decimal(exclusive_minimum=var_0, precision=var_0)
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
    var_3.serialize(var_4)

def test_case_62():
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
    var_1 = -2531
    var_2 = module_0.Object(min_properties=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert var_2.additional_properties is True
    assert var_2.property_names is None
    assert var_2.min_properties == -2531
    assert var_2.max_properties is None
    assert var_2.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = module_1.python_version_tuple()

def test_case_63():
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
    var_1 = module_1.python_revision()
    assert var_1 == ''
    var_2 = module_0.Object(properties=var_1)
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
    var_3 = var_0.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_64():
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
    var_2 = module_0.Float(coerce_types=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Float'
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
    assert var_2.coerce_types is None
    var_3 = module_1.python_version_tuple()
    module_0.Float(exclusive_minimum=var_3, exclusive_maximum=var_0, precision=var_3)

def test_case_65():
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
        module_0.Object(additional_properties=var_0, max_properties=var_0)

def test_case_66():
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
    var_1 = module_1.freedesktop_os_release()
    var_2 = module_0.Object(additional_properties=var_0, required=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert f'{type(var_2.additional_properties).__module__}.{type(var_2.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.property_names is None
    assert var_2.min_properties is None
    assert var_2.max_properties is None
    assert var_2.required == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.3', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 8

def test_case_67():
    var_0 = 'GZKp@gX@'
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
    var_2 = {var_0: var_0, var_0: var_0, var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert f'{type(var_3.pattern_properties).__module__}.{type(var_3.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_3.pattern_properties) == 1
    assert var_3.additional_properties is True
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = var_3.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_68():
    var_0 = 'GZKp@gX@'
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
    var_2 = {var_0: var_0, var_0: var_0, var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
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
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    with pytest.raises(module_5.ValidationError):
        var_3.validate(var_2)

def test_case_69():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Integer(coerce_types=var_0)
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
    assert var_1.coerce_types == 204875749604013
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2 = var_1.validate(var_0)
    assert var_2 == 204875749604013

def test_case_70():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Integer(exclusive_maximum=var_0, precision=var_0, multiple_of=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum == 204875749604013
    assert var_1.multiple_of == 204875749604013
    assert var_1.precision == 204875749604013
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
def test_case_71():
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
    var_2 = module_0.Float(coerce_types=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Float'
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
    assert var_2.coerce_types is None
    var_2.validate(var_2)

def test_case_72():
    var_0 = None
    var_1 = True
    var_2 = module_0.Array(exact_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items is True
    assert var_2.max_items is True
    assert var_2.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_73():
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
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_1 = None
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_74():
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
    var_2.validate_or_error(var_2)

def test_case_75():
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
    var_1 = 105
    var_2 = module_0.Object(additional_properties=var_0, property_names=var_0, max_properties=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert f'{type(var_2.additional_properties).__module__}.{type(var_2.additional_properties).__qualname__}' == 'typesystem.fields.UUID'
    assert f'{type(var_2.property_names).__module__}.{type(var_2.property_names).__qualname__}' == 'typesystem.fields.UUID'
    assert var_2.min_properties is None
    assert var_2.max_properties == 105
    assert var_2.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_0.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_76():
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
    var_0.validate_or_error(var_0)

def test_case_77():
    var_0 = module_0.Float()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Float'
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
    var_1 = module_1.processor()
    assert var_1 == ''
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_78():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_0.Const(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Const'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.const == 204875749604013
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == 204875749604013
    assert var_2.error is None

@pytest.mark.xfail(strict=True)
def test_case_79():
    var_0 = None
    var_1 = module_0.Float(minimum=var_0, exclusive_minimum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Float'
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
    var_3 = module_2.getnode()
    assert var_3 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    module_0.Integer(minimum=var_3, exclusive_minimum=var_0, exclusive_maximum=var_1, coerce_types=var_3)

def test_case_80():
    var_0 = None
    var_1 = module_0.Float(exclusive_maximum=var_0, precision=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Float'
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
    var_2 = module_1.freedesktop_os_release()
    with pytest.raises(AssertionError):
        module_0.Object(additional_properties=var_2, property_names=var_0, required=var_0)

def test_case_81():
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
    var_2 = var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
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
    var_1 = True
    var_2 = None
    var_3 = module_0.Float(maximum=var_1, exclusive_minimum=var_2, precision=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Float'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is True
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    var_4 = module_1.freedesktop_os_release()
    var_5 = module_2.getnode()
    assert var_5 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_3.validate(var_5)

def test_case_83():
    var_0 = module_2.uuid4()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'uuid.UUID'
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
    with pytest.raises(AssertionError):
        module_0.Array(additional_items=var_0)

@pytest.mark.xfail(strict=True)
def test_case_84():
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
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_0.validate(var_0)

def test_case_85():
    var_0 = module_1.freedesktop_os_release()
    var_1 = False
    var_2 = module_2.getnode()
    assert var_2 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_3 = module_0.Integer(maximum=var_2, exclusive_minimum=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum == 204875749604013
    assert var_3.exclusive_minimum is False
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_4 = var_3.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value == 204875749604013
    assert var_4.error is None

@pytest.mark.xfail(strict=True)
def test_case_86():
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
    module_0.Float(multiple_of=var_1)

def test_case_87():
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
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_1 = module_1.processor()
    assert var_1 == ''
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is False
    assert var_2.error is None

@pytest.mark.xfail(strict=True)
def test_case_88():
    var_0 = None
    var_1 = module_0.Float(exclusive_maximum=var_0, precision=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Float'
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
    var_2 = module_0.Boolean()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.coerce_types is True
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_3 = False
    var_4 = var_2.validate(var_3)
    assert var_4 is False
    module_0.URL(**var_4)

@pytest.mark.xfail(strict=True)
def test_case_89():
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
    var_2 = False
    var_3 = module_0.Boolean(coerce_types=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.coerce_types is False
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_3.validate(var_1)

def test_case_90():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = module_2.getnode()
    assert var_1 == 204875749604013
    with pytest.raises(AssertionError):
        module_0.Array(var_1)

def test_case_91():
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
    with pytest.raises(AssertionError):
        module_0.Object(min_properties=var_0)

@pytest.mark.xfail(strict=True)
def test_case_92():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = None
    var_2 = module_1.processor()
    assert var_2 == ''
    var_3 = module_0.Field()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_3.validate_or_error(var_1)

def test_case_93():
    var_0 = module_2.getnode()
    assert var_0 == 204875749604013
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = None
    with pytest.raises(AssertionError):
        module_0.String(allow_blank=var_0, trim_whitespace=var_1, pattern=var_1, format=var_0)

def test_case_94():
    var_0 = module_0.Float()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Float'
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
    var_1 = False
    var_2 = module_0.Array(var_0, min_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.Float'
    assert var_2.additional_items is False
    assert var_2.min_items is False
    assert var_2.max_items is None
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = None
    var_4 = var_2.validate_or_error(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

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
    var_3 = module_1.freedesktop_os_release()
    var_4 = module_2.uuid4()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'uuid.UUID'
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
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_0, pattern_properties=var_3, additional_properties=var_2, property_names=var_1, max_properties=var_4, required=var_4)

def test_case_96():
    var_0 = True
    var_1 = None
    var_2 = module_0.String(allow_blank=var_0, trim_whitespace=var_1, max_length=var_0, min_length=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.default == ''
    assert var_2.allow_blank is True
    assert var_2.trim_whitespace is None
    assert var_2.max_length is True
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}

def test_case_97():
    var_0 = module_1.uname()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'platform.uname_result'
    assert len(var_0) == 6
    assert f'{type(module_1.uname_result.processor).__module__}.{type(module_1.uname_result.processor).__qualname__}' == 'functools.cached_property'
    assert module_1.uname_result.processor.attrname == 'processor'
    assert f'{type(module_1.uname_result.processor.lock).__module__}.{type(module_1.uname_result.processor.lock).__qualname__}' == '_thread.RLock'
    with pytest.raises(AssertionError):
        module_0.Array(var_0, exact_items=var_0)

def test_case_98():
    var_0 = module_1.python_revision()
    assert var_0 == ''
    with pytest.raises(AssertionError):
        module_0.Array(min_items=var_0, unique_items=var_0)

def test_case_99():
    var_0 = module_1.python_compiler()
    assert var_0 == 'GCC 14.2.0'
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
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == 'GCC 14.2.0'
    assert var_2.error is None
    with pytest.raises(AssertionError):
        module_0.String(max_length=var_0, min_length=var_0)

@pytest.mark.xfail(strict=True)
def test_case_100():
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
    var_1 = None
    var_2 = module_0.Const(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Const'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.const is None
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_2.validate(var_2)

def test_case_101():
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
    var_1 = ''
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
    var_3 = {var_1: var_1, var_1: var_2, var_1: var_2, var_1: var_2}
    var_4 = module_0.Object(properties=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.properties).__module__}.{type(var_4.properties).__qualname__}' == 'builtins.dict'
    assert len(var_4.properties) == 1
    assert var_4.pattern_properties == {}
    assert var_4.additional_properties is True
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = None
    var_6 = var_4.validate_or_error(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_7 = module_0.Const(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Const'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.const).__module__}.{type(var_7.const).__qualname__}' == 'typesystem.fields.Object'
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_8 = var_7.validate_or_error(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert f'{type(var_8.error).__module__}.{type(var_8.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8.error) == 1

def test_case_102():
    var_0 = 'yl8.C95i'
    var_1 = module_0.Boolean()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
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
    var_3 = {var_0: var_0, var_0: var_0, var_0: var_2, var_0: var_0, var_0: var_2, var_0: var_2, var_0: var_0, var_0: var_0, var_0: var_2, var_0: var_2, var_0: var_2, var_0: var_2}
    var_4 = module_0.Object(pattern_properties=var_3)
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
    var_5 = var_4.validate_or_error(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    with pytest.raises(AssertionError):
        module_0.String(pattern=var_4, coerce_types=var_4)

def test_case_103():
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
    var_1 = module_0.Object(properties=var_0, additional_properties=var_0, property_names=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert f'{type(var_1.additional_properties).__module__}.{type(var_1.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_1.property_names).__module__}.{type(var_1.property_names).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = {var_0: var_0, var_0: var_1, var_0: var_0}
    var_3 = var_1.validate_or_error(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_104():
    var_0 = '#I'
    var_1 = module_0.String(allow_blank=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.default == ''
    assert var_1.allow_blank == '#I'
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
    var_2 = var_1.validate(var_0)
    assert var_2 == '#I'

def test_case_105():
    var_0 = module_0.String()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = module_1.win32_is_iot()
    assert var_1 is False
    var_2 = [var_0]
    var_3 = module_0.Array(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'builtins.list'
    assert len(var_3.items) == 1
    assert var_3.additional_items is False
    assert var_3.min_items == 1
    assert var_3.max_items == 1
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = var_3.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_106():
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
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_1 = -4925
    var_2 = module_0.Integer(maximum=var_1, multiple_of=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum == -4925
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of == -4925
    assert var_2.precision is None
    assert var_2.coerce_types is True
    var_3 = var_2.__or__(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = module_1.python_revision()
    assert var_4 == ''
    var_5 = var_3.validate_or_error(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is False
    assert var_5.error is None

@pytest.mark.xfail(strict=True)
def test_case_107():
    var_0 = None
    var_1 = module_0.Decimal(exclusive_minimum=var_0, precision=var_0)
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
    var_2 = None
    var_3 = var_1.serialize(var_2)
    var_4 = module_0.Integer(minimum=var_0, precision=var_0, multiple_of=var_2, coerce_types=var_2)
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
    assert var_4.coerce_types is None
    var_5 = var_1.__or__(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.any_of).__module__}.{type(var_5.any_of).__qualname__}' == 'builtins.list'
    assert len(var_5.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_6 = module_1.win32_edition()
    var_6.capitalize()

def test_case_108():
    var_0 = module_3.RegexFlag.LOCALE
    var_1 = module_3.RegexFlag.UNICODE
    var_2 = None
    var_3 = module_0.Integer(minimum=var_1, multiple_of=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum == module_3.RegexFlag.UNICODE
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_4 = var_3.validate_or_error(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_109():
    var_0 = module_1.python_version()
    assert var_0 == '3.10.19'
    var_1 = module_0.String(pattern=var_0, coerce_types=var_0)
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
    assert var_1.coerce_types == '3.10.19'
    assert var_1.pattern == '3.10.19'
    assert f'{type(var_1.pattern_regex).__module__}.{type(var_1.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = var_1.validate(var_0)
    assert var_2 == '3.10.19'

@pytest.mark.xfail(strict=True)
def test_case_110():
    var_0 = True
    var_1 = ':\x0b8/8I\n_:E'
    var_2 = module_0.String(max_length=var_0, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is True
    assert var_2.min_length is None
    assert var_2.format == ':\x0b8/8I\n_:E'
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
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
    var_2.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_111():
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
    var_1 = var_0.get_default_value()
    assert var_1 is None
    var_2 = 'test_value'
    var_3 = module_0.Field(default=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.default == 'test_value'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    var_4 = var_3.get_default_value()
    assert var_4 == 'test_value'
    var_5 = 42
    var_6 = module_0.Field(default=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Field'
    assert var_6.default == 42
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    var_7 = var_6.get_default_value()
    assert var_7 == 42
    var_8 = var_6.get_default_value()
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = lambda : var_12
    var_14 = module_0.Field(default=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Field'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    var_14.get_default_value()

def test_case_112():
    var_0 = module_0.String()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = module_0.Array(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.items).__module__}.{type(var_1.items).__qualname__}' == 'typesystem.fields.String'
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items is None
    assert var_1.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = module_0.String()
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
    var_5 = module_0.Array(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.items).__module__}.{type(var_5.items).__qualname__}' == 'typesystem.fields.String'
    assert var_5.additional_items is False
    assert var_5.min_items is None
    assert var_5.max_items is None
    assert var_5.unique_items is False
    var_6 = 'hello'
    var_7 = 'world'
    var_8 = [var_6, var_7]
    var_9 = var_5.serialize(var_8)
    var_10 = module_0.String()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length is None
    assert var_10.min_length is None
    assert var_10.format is None
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = module_0.Integer()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Integer'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.minimum is None
    assert var_11.maximum is None
    assert var_11.exclusive_minimum is None
    assert var_11.exclusive_maximum is None
    assert var_11.multiple_of is None
    assert var_11.precision is None
    assert var_11.coerce_types is True
    var_12 = [var_10, var_11]
    var_13 = module_0.Array(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Array'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.items).__module__}.{type(var_13.items).__qualname__}' == 'builtins.list'
    assert len(var_13.items) == 2
    assert var_13.additional_items is False
    assert var_13.min_items == 2
    assert var_13.max_items == 2
    assert var_13.unique_items is False
    var_14 = 42
    var_15 = [var_6, var_14]
    var_16 = var_13.serialize(var_15)
    var_17 = module_0.Array(var_2)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Array'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.items is None
    assert var_17.additional_items is False
    assert var_17.min_items is None
    assert var_17.max_items is None
    assert var_17.unique_items is False
    var_18 = 1
    var_19 = 'test'
    var_20 = 3.14
    var_21 = [var_18, var_19, var_20]
    var_22 = var_17.serialize(var_21)
    var_23 = module_0.Integer()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Integer'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.minimum is None
    assert var_23.maximum is None
    assert var_23.exclusive_minimum is None
    assert var_23.exclusive_maximum is None
    assert var_23.multiple_of is None
    assert var_23.precision is None
    assert var_23.coerce_types is True
    var_24 = module_0.Array(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Array'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.items).__module__}.{type(var_24.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_24.additional_items is False
    assert var_24.min_items is None
    assert var_24.max_items is None
    assert var_24.unique_items is False
    var_25 = 2
    var_26 = 3
    var_27 = [var_18, var_25, var_26]
    var_28 = var_24.serialize(var_27)
    var_29 = module_0.Decimal()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert var_29.minimum is None
    assert var_29.maximum is None
    assert var_29.exclusive_minimum is None
    assert var_29.exclusive_maximum is None
    assert var_29.multiple_of is None
    assert var_29.precision is None
    assert var_29.coerce_types is True
    var_30 = module_0.Array(var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Array'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.items).__module__}.{type(var_30.items).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_30.additional_items is False
    assert var_30.min_items is None
    assert var_30.max_items is None
    assert var_30.unique_items is False
    var_31 = module_0.Integer()
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Integer'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.minimum is None
    assert var_31.maximum is None
    assert var_31.exclusive_minimum is None
    assert var_31.exclusive_maximum is None
    assert var_31.multiple_of is None
    assert var_31.precision is None
    assert var_31.coerce_types is True
    var_32 = module_0.Array(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Array'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.items).__module__}.{type(var_32.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_32.additional_items is False
    assert var_32.min_items is None
    assert var_32.max_items is None
    assert var_32.unique_items is False
    var_33 = module_0.Array(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Array'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.items).__module__}.{type(var_33.items).__qualname__}' == 'typesystem.fields.Array'
    assert var_33.additional_items is False
    assert var_33.min_items is None
    assert var_33.max_items is None
    assert var_33.unique_items is False
    var_34 = [var_18, var_25]
    var_35 = 4
    var_36 = [var_26, var_35]
    var_37 = [var_34, var_36]
    var_38 = var_33.serialize(var_37)
    var_39 = 'name'
    var_40 = 'age'
    var_41 = module_0.String()
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.String'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.allow_blank is False
    assert var_41.trim_whitespace is True
    assert var_41.max_length is None
    assert var_41.min_length is None
    assert var_41.format is None
    assert var_41.coerce_types is True
    assert var_41.pattern is None
    assert var_41.pattern_regex is None
    var_42 = module_0.Integer()
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.Integer'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert var_42.minimum is None
    assert var_42.maximum is None
    assert var_42.exclusive_minimum is None
    assert var_42.exclusive_maximum is None
    assert var_42.multiple_of is None
    assert var_42.precision is None
    assert var_42.coerce_types is True
    var_43 = {var_39: var_41, var_40: var_42}
    var_44 = module_0.Object(properties=var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Object'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert f'{type(var_44.properties).__module__}.{type(var_44.properties).__qualname__}' == 'builtins.dict'
    assert len(var_44.properties) == 2
    assert var_44.pattern_properties == {}
    assert var_44.additional_properties is True
    assert var_44.property_names is None
    assert var_44.min_properties is None
    assert var_44.max_properties is None
    assert var_44.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_45 = module_0.Array(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Array'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert f'{type(var_45.items).__module__}.{type(var_45.items).__qualname__}' == 'typesystem.fields.Object'
    assert var_45.additional_items is False
    assert var_45.min_items is None
    assert var_45.max_items is None
    assert var_45.unique_items is False
    var_46 = 'Alice'
    var_47 = 30
    var_48 = {var_39: var_46, var_40: var_47}
    var_49 = [var_48]
    var_50 = var_45.serialize(var_49)
    var_51 = module_0.Boolean()
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert var_51.coerce_types is True
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_52 = module_0.Array(var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Array'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert f'{type(var_52.items).__module__}.{type(var_52.items).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_52.additional_items is False
    assert var_52.min_items is None
    assert var_52.max_items is None
    assert var_52.unique_items is False
    var_53 = True
    var_54 = False
    var_55 = True
    var_56 = [var_53, var_54, var_55]
    var_57 = var_52.serialize(var_56)
    var_58 = module_0.Array(var_2)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.Array'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.items is None
    assert var_58.additional_items is False
    assert var_58.min_items is None
    assert var_58.max_items is None
    assert var_58.unique_items is False
    var_59 = 'string'
    var_60 = True
    var_61 = [var_55, var_59, var_20, var_60, var_2]
    var_62 = var_58.serialize(var_61)
    var_63 = module_0.String()
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.fields.String'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert var_63.allow_blank is False
    assert var_63.trim_whitespace is True
    assert var_63.max_length is None
    assert var_63.min_length is None
    assert var_63.format is None
    assert var_63.coerce_types is True
    assert var_63.pattern is None
    assert var_63.pattern_regex is None
    var_64 = module_0.Array(var_63)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.Array'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert f'{type(var_64.items).__module__}.{type(var_64.items).__qualname__}' == 'typesystem.fields.String'
    assert var_64.additional_items is False
    assert var_64.min_items is None
    assert var_64.max_items is None
    assert var_64.unique_items is False
    var_65 = []
    var_66 = var_64.serialize(var_65)
    var_67 = module_0.String()
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.fields.String'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    assert var_67.allow_blank is False
    assert var_67.trim_whitespace is True
    assert var_67.max_length is None
    assert var_67.min_length is None
    assert var_67.format is None
    assert var_67.coerce_types is True
    assert var_67.pattern is None
    assert var_67.pattern_regex is None
    var_68 = module_0.Integer()
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.fields.Integer'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert var_68.minimum is None
    assert var_68.maximum is None
    assert var_68.exclusive_minimum is None
    assert var_68.exclusive_maximum is None
    assert var_68.multiple_of is None
    assert var_68.precision is None
    assert var_68.coerce_types is True
    var_69 = module_0.Boolean()
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert var_69.coerce_types is True
    var_70 = [var_67, var_68, var_69]
    var_71 = module_0.Array(var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Array'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert f'{type(var_71.items).__module__}.{type(var_71.items).__qualname__}' == 'builtins.list'
    assert len(var_71.items) == 3
    assert var_71.additional_items is False
    assert var_71.min_items == 3
    assert var_71.max_items == 3
    assert var_71.unique_items is False
    var_72 = True
    var_73 = [var_19, var_14, var_72]
    var_74 = var_71.serialize(var_73)
    var_75 = module_0.Decimal()
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert var_75.minimum is None
    assert var_75.maximum is None
    assert var_75.exclusive_minimum is None
    assert var_75.exclusive_maximum is None
    assert var_75.multiple_of is None
    assert var_75.precision is None
    assert var_75.coerce_types is True
    var_76 = module_0.Integer()
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.fields.Integer'
    assert var_76.title == ''
    assert var_76.description == ''
    assert var_76.allow_null is False
    assert var_76.read_only is False
    assert var_76.minimum is None
    assert var_76.maximum is None
    assert var_76.exclusive_minimum is None
    assert var_76.exclusive_maximum is None
    assert var_76.multiple_of is None
    assert var_76.precision is None
    assert var_76.coerce_types is True
    var_77 = [var_75, var_76]
    var_78 = module_0.Array(var_77)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.Array'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert f'{type(var_78.items).__module__}.{type(var_78.items).__qualname__}' == 'builtins.list'
    assert len(var_78.items) == 2
    assert var_78.additional_items is False
    assert var_78.min_items == 2
    assert var_78.max_items == 2
    assert var_78.unique_items is False

def test_case_113():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Choice'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.choices == [('red', 'red'), ('green', 'green'), ('blue', 'blue')]
    assert var_4.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_5 = var_4.choices
    var_6 = len(var_5)
    assert var_6 == 3
    var_7 = 'r'
    var_8 = 'Red'
    var_9 = (var_7, var_8)
    var_10 = 'g'
    var_11 = 'Green'
    var_12 = (var_10, var_11)
    var_13 = 'b'
    var_14 = 'Blue'
    var_15 = (var_13, var_14)
    var_16 = [var_9, var_12, var_15]
    var_17 = module_0.Choice(choices=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Choice'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.choices == [('r', 'Red'), ('g', 'Green'), ('b', 'Blue')]
    assert var_17.coerce_types is True
    var_18 = var_17.choices
    var_19 = len(var_18)
    assert var_19 == 3
    var_20 = (var_10, var_11)
    var_21 = [var_0, var_20, var_2]
    var_22 = module_0.Choice(choices=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Choice'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.choices == [('red', 'red'), ('g', 'Green'), ('blue', 'blue')]
    assert var_22.coerce_types is True
    var_23 = var_22.choices
    var_24 = len(var_23)
    assert var_24 == 3
    var_25 = []
    var_26 = module_0.Choice(choices=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Choice'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.choices == []
    assert var_26.coerce_types is True
    var_27 = var_26.choices
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = None
    var_30 = module_0.Choice(choices=var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Choice'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.choices == []
    assert var_30.coerce_types is True
    var_31 = len(var_15)
    var_32 = [var_23, var_13]
    var_33 = True
    with pytest.raises(AssertionError):
        module_0.Choice(choices=var_32)

@pytest.mark.xfail(strict=True)
def test_case_114():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Choice'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.choices == [('red', 'red'), ('green', 'green'), ('blue', 'blue')]
    assert var_4.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_5 = var_4.validate(var_0)
    assert var_5 == 'red'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'green'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'blue'
    var_8 = '1'
    var_9 = 'Option 1'
    var_10 = (var_8, var_9)
    var_11 = '2'
    var_12 = 'vZ\tn:~;\\ru=IGNqi'
    var_13 = (var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = module_0.Choice(choices=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Choice'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.choices == [('1', 'Option 1'), ('2', 'vZ\tn:~;\\ru=IGNqi')]
    assert var_15.coerce_types is True
    var_16 = var_15.validate(var_8)
    assert var_16 == '1'
    var_17 = var_15.validate(var_11)
    assert var_17 == '2'
    var_6.validate_or_error(var_17)

def test_case_115():
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
    var_1 = 'name'
    var_2 = 'age'
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
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Object(properties=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.properties).__module__}.{type(var_6.properties).__qualname__}' == 'builtins.dict'
    assert len(var_6.properties) == 2
    assert var_6.pattern_properties == {}
    assert var_6.additional_properties is True
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == []
    var_7 = var_6.properties
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = module_0.String()
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
    var_10 = module_0.Integer()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Integer'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.minimum is None
    assert var_10.maximum is None
    assert var_10.exclusive_minimum is None
    assert var_10.exclusive_maximum is None
    assert var_10.multiple_of is None
    assert var_10.precision is None
    assert var_10.coerce_types is True
    var_11 = var_6.pattern_properties
    var_12 = False
    var_13 = module_0.Object(additional_properties=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Object'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.properties == {}
    assert var_13.pattern_properties == {}
    assert var_13.additional_properties is False
    assert var_13.property_names is None
    assert var_13.min_properties is None
    assert var_13.max_properties is None
    assert var_13.required == []
    var_14 = True
    var_15 = module_0.Object(additional_properties=var_14)
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
    var_16 = module_0.String()
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
    var_17 = module_0.Object(additional_properties=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Object'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.properties == {}
    assert var_17.pattern_properties == {}
    assert f'{type(var_17.additional_properties).__module__}.{type(var_17.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_17.property_names is None
    assert var_17.min_properties is None
    assert var_17.max_properties is None
    assert var_17.required == []
    var_18 = '^[a-z]+$'
    var_19 = module_0.String(pattern=var_18)
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
    assert var_19.pattern == '^[a-z]+$'
    assert f'{type(var_19.pattern_regex).__module__}.{type(var_19.pattern_regex).__qualname__}' == 're.Pattern'
    var_20 = module_0.Object(property_names=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Object'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.properties == {}
    assert var_20.pattern_properties == {}
    assert var_20.additional_properties is True
    assert f'{type(var_20.property_names).__module__}.{type(var_20.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_20.min_properties is None
    assert var_20.max_properties is None
    assert var_20.required == []
    var_21 = 'email'
    var_22 = (var_1, var_21)
    var_23 = module_0.Object(required=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Object'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.properties == {}
    assert var_23.pattern_properties == {}
    assert var_23.additional_properties is True
    assert var_23.property_names is None
    assert var_23.min_properties is None
    assert var_23.max_properties is None
    assert var_23.required == ['name', 'email']
    var_24 = module_0.String()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.String'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.allow_blank is False
    assert var_24.trim_whitespace is True
    assert var_24.max_length is None
    assert var_24.min_length is None
    assert var_24.format is None
    assert var_24.coerce_types is True
    assert var_24.pattern is None
    assert var_24.pattern_regex is None
    var_25 = module_0.Integer()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Integer'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.minimum is None
    assert var_25.maximum is None
    assert var_25.exclusive_minimum is None
    assert var_25.exclusive_maximum is None
    assert var_25.multiple_of is None
    assert var_25.precision is None
    assert var_25.coerce_types is True
    var_26 = {var_1: var_24, var_2: var_25}
    var_27 = '^meta_'
    var_28 = module_0.String()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.String'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.allow_blank is False
    assert var_28.trim_whitespace is True
    assert var_28.max_length is None
    assert var_28.min_length is None
    assert var_28.format is None
    assert var_28.coerce_types is True
    assert var_28.pattern is None
    assert var_28.pattern_regex is None
    var_29 = {var_27: var_28}
    var_30 = [var_1]
    var_31 = module_0.String()
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.String'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.allow_blank is False
    assert var_31.trim_whitespace is True
    assert var_31.max_length is None
    assert var_31.min_length is None
    assert var_31.format is None
    assert var_31.coerce_types is True
    assert var_31.pattern is None
    assert var_31.pattern_regex is None
    var_32 = 20
    var_33 = module_0.Object(properties=var_26, pattern_properties=var_29, additional_properties=var_12, property_names=var_31, min_properties=var_14, max_properties=var_32, required=var_30)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Object'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.properties).__module__}.{type(var_33.properties).__qualname__}' == 'builtins.dict'
    assert len(var_33.properties) == 2
    assert f'{type(var_33.pattern_properties).__module__}.{type(var_33.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_33.pattern_properties) == 1
    assert var_33.additional_properties is False
    assert f'{type(var_33.property_names).__module__}.{type(var_33.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_33.min_properties is True
    assert var_33.max_properties == 20
    assert var_33.required == ['name']
    var_34 = module_0.String()
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.String'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.allow_blank is False
    assert var_34.trim_whitespace is True
    assert var_34.max_length is None
    assert var_34.min_length is None
    assert var_34.format is None
    assert var_34.coerce_types is True
    assert var_34.pattern is None
    assert var_34.pattern_regex is None
    var_35 = {var_1: var_34}
    var_36 = module_0.Object(properties=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Object'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert f'{type(var_36.properties).__module__}.{type(var_36.properties).__qualname__}' == 'builtins.dict'
    assert len(var_36.properties) == 1
    assert var_36.pattern_properties == {}
    assert var_36.additional_properties is True
    assert var_36.property_names is None
    assert var_36.min_properties is None
    assert var_36.max_properties is None
    assert var_36.required == []
    var_37 = var_36.properties
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = module_0.String()
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.String'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.allow_blank is False
    assert var_39.trim_whitespace is True
    assert var_39.max_length is None
    assert var_39.min_length is None
    assert var_39.format is None
    assert var_39.coerce_types is True
    assert var_39.pattern is None
    assert var_39.pattern_regex is None
    var_40 = None
    var_41 = module_0.Object(required=var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Object'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.properties == {}
    assert var_41.pattern_properties == {}
    assert var_41.additional_properties is True
    assert var_41.property_names is None
    assert var_41.min_properties is None
    assert var_41.max_properties is None
    assert var_41.required == []
    var_42 = 123
    var_43 = module_0.String()
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.String'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.allow_blank is False
    assert var_43.trim_whitespace is True
    assert var_43.max_length is None
    assert var_43.min_length is None
    assert var_43.format is None
    assert var_43.coerce_types is True
    assert var_43.pattern is None
    assert var_43.pattern_regex is None
    var_44 = {var_42: var_43}
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_44)

def test_case_116():
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
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.Integer()
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
    var_4 = module_0.String()
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_0.Integer()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Integer'
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
    var_6 = False
    var_7 = module_0.Object(additional_properties=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is False
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    var_8 = True
    var_9 = module_0.Object(additional_properties=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Object'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.properties == {}
    assert var_9.pattern_properties == {}
    assert var_9.additional_properties is True
    assert var_9.property_names is None
    assert var_9.min_properties is None
    assert var_9.max_properties is None
    assert var_9.required == []
    var_10 = module_0.String()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length is None
    assert var_10.min_length is None
    assert var_10.format is None
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = module_0.Object(additional_properties=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Object'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.properties == {}
    assert var_11.pattern_properties == {}
    assert f'{type(var_11.additional_properties).__module__}.{type(var_11.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_11.property_names is None
    assert var_11.min_properties is None
    assert var_11.max_properties is None
    assert var_11.required == []
    var_12 = '^[a-z]+$'
    var_13 = module_0.String(pattern=var_12)
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
    assert var_13.pattern == '^[a-z]+$'
    assert f'{type(var_13.pattern_regex).__module__}.{type(var_13.pattern_regex).__qualname__}' == 're.Pattern'
    var_14 = module_0.Object(property_names=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Object'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.properties == {}
    assert var_14.pattern_properties == {}
    assert var_14.additional_properties is True
    assert f'{type(var_14.property_names).__module__}.{type(var_14.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_14.min_properties is None
    assert var_14.max_properties is None
    assert var_14.required == []
    var_15 = 'email'
    var_16 = [var_1, var_15]
    var_17 = module_0.Object(required=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Object'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.properties == {}
    assert var_17.pattern_properties == {}
    assert var_17.additional_properties is True
    assert var_17.property_names is None
    assert var_17.min_properties is None
    assert var_17.max_properties is None
    assert var_17.required == ['name', 'email']
    var_18 = (var_1, var_15)
    var_19 = module_0.Object(required=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Object'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.properties == {}
    assert var_19.pattern_properties == {}
    assert var_19.additional_properties is True
    assert var_19.property_names is None
    assert var_19.min_properties is None
    assert var_19.max_properties is None
    assert var_19.required == ['name', 'email']
    var_20 = module_0.String()
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
    var_21 = module_0.Integer()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Integer'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.minimum is None
    assert var_21.maximum is None
    assert var_21.exclusive_minimum is None
    assert var_21.exclusive_maximum is None
    assert var_21.multiple_of is None
    assert var_21.precision is None
    assert var_21.coerce_types is True
    var_22 = {var_1: var_20, var_2: var_21}
    var_23 = module_0.String()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.String'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.allow_blank is False
    assert var_23.trim_whitespace is True
    assert var_23.max_length is None
    assert var_23.min_length is None
    assert var_23.format is None
    assert var_23.coerce_types is True
    assert var_23.pattern is None
    assert var_23.pattern_regex is None
    var_24 = {var_12: var_23}
    var_25 = [var_1]
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
    var_27 = 20
    var_28 = 'A user object'
    var_29 = module_0.Object(properties=var_22, pattern_properties=var_24, additional_properties=var_6, property_names=var_26, min_properties=var_8, max_properties=var_27, required=var_25)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Object'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.properties).__module__}.{type(var_29.properties).__qualname__}' == 'builtins.dict'
    assert len(var_29.properties) == 2
    assert f'{type(var_29.pattern_properties).__module__}.{type(var_29.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_29.pattern_properties) == 1
    assert var_29.additional_properties is False
    assert f'{type(var_29.property_names).__module__}.{type(var_29.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_29.min_properties is True
    assert var_29.max_properties == 20
    assert var_29.required == ['name']
    var_30 = module_0.String()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.String'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.allow_blank is False
    assert var_30.trim_whitespace is True
    assert var_30.max_length is None
    assert var_30.min_length is None
    assert var_30.format is None
    assert var_30.coerce_types is True
    assert var_30.pattern is None
    assert var_30.pattern_regex is None
    var_31 = {var_1: var_30}
    var_32 = module_0.Object(properties=var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Object'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.properties).__module__}.{type(var_32.properties).__qualname__}' == 'builtins.dict'
    assert len(var_32.properties) == 1
    assert var_32.pattern_properties == {}
    assert var_32.additional_properties is True
    assert var_32.property_names is None
    assert var_32.min_properties is None
    assert var_32.max_properties is None
    assert var_32.required == []
    var_33 = var_32.properties
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = module_0.String()
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.String'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.allow_blank is False
    assert var_35.trim_whitespace is True
    assert var_35.max_length is None
    assert var_35.min_length is None
    assert var_35.format is None
    assert var_35.coerce_types is True
    assert var_35.pattern is None
    assert var_35.pattern_regex is None
    var_36 = None
    var_37 = module_0.Object(required=var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Object'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.properties == {}
    assert var_37.pattern_properties == {}
    assert var_37.additional_properties is True
    assert var_37.property_names is None
    assert var_37.min_properties is None
    assert var_37.max_properties is None
    assert var_37.required == []
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
    var_39 = var_34.__repr__()
    assert var_39 == '1'

def test_case_117():
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
    var_1 = module_0.String()
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_0.Array(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.String'
    assert var_2.additional_items is False
    assert var_2.min_items is None
    assert var_2.max_items is None
    assert var_2.unique_items is False
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
    var_7 = module_0.String()
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
    var_8 = module_0.Integer()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Integer'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.minimum is None
    assert var_8.maximum is None
    assert var_8.exclusive_minimum is None
    assert var_8.exclusive_maximum is None
    assert var_8.multiple_of is None
    assert var_8.precision is None
    assert var_8.coerce_types is True
    var_9 = [var_7, var_8]
    var_10 = True
    var_11 = module_0.Array(var_9, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Array'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.items).__module__}.{type(var_11.items).__qualname__}' == 'builtins.list'
    assert len(var_11.items) == 2
    assert var_11.additional_items is True
    assert var_11.min_items == 2
    assert var_11.max_items is None
    assert var_11.unique_items is False
    var_12 = module_0.Boolean()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.coerce_types is True
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_13 = module_0.Array(var_9, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Array'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.items).__module__}.{type(var_13.items).__qualname__}' == 'builtins.list'
    assert len(var_13.items) == 2
    assert f'{type(var_13.additional_items).__module__}.{type(var_13.additional_items).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_13.min_items == 2
    assert var_13.max_items is None
    assert var_13.unique_items is False
    var_14 = module_0.String()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.String'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.allow_blank is False
    assert var_14.trim_whitespace is True
    assert var_14.max_length is None
    assert var_14.min_length is None
    assert var_14.format is None
    assert var_14.coerce_types is True
    assert var_14.pattern is None
    assert var_14.pattern_regex is None
    var_15 = module_0.Integer()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Integer'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is None
    assert var_15.maximum is None
    assert var_15.exclusive_minimum is None
    assert var_15.exclusive_maximum is None
    assert var_15.multiple_of is None
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = (var_14, var_15)
    var_17 = module_0.Array(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Array'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.items).__module__}.{type(var_17.items).__qualname__}' == 'builtins.list'
    assert len(var_17.items) == 2
    assert var_17.additional_items is False
    assert var_17.min_items == 2
    assert var_17.max_items == 2
    assert var_17.unique_items is False
    var_18 = list(var_16)
    var_19 = 5
    var_20 = module_0.Array(var_15, min_items=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Array'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.items).__module__}.{type(var_20.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_20.additional_items is False
    assert var_20.min_items == 5
    assert var_20.max_items is None
    assert var_20.unique_items is False
    var_21 = module_0.String()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.String'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.allow_blank is False
    assert var_21.trim_whitespace is True
    assert var_21.max_length is None
    assert var_21.min_length is None
    assert var_21.format is None
    assert var_21.coerce_types is True
    assert var_21.pattern is None
    assert var_21.pattern_regex is None
    var_22 = 10
    var_23 = module_0.Array(var_21, max_items=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Array'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.items).__module__}.{type(var_23.items).__qualname__}' == 'typesystem.fields.String'
    assert var_23.additional_items is False
    assert var_23.min_items is None
    assert var_23.max_items == 10
    assert var_23.unique_items is False
    var_24 = module_0.String()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.String'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.allow_blank is False
    assert var_24.trim_whitespace is True
    assert var_24.max_length is None
    assert var_24.min_length is None
    assert var_24.format is None
    assert var_24.coerce_types is True
    assert var_24.pattern is None
    assert var_24.pattern_regex is None
    var_25 = 7
    var_26 = module_0.Array(var_24, exact_items=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Array'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.items).__module__}.{type(var_26.items).__qualname__}' == 'typesystem.fields.String'
    assert var_26.additional_items is False
    assert var_26.min_items == 7
    assert var_26.max_items == 7
    assert var_26.unique_items is False
    var_27 = module_0.String()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.String'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.allow_blank is False
    assert var_27.trim_whitespace is True
    assert var_27.max_length is None
    assert var_27.min_length is None
    assert var_27.format is None
    assert var_27.coerce_types is True
    assert var_27.pattern is None
    assert var_27.pattern_regex is None
    var_28 = module_0.Array(var_27, unique_items=var_10)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Array'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.items).__module__}.{type(var_28.items).__qualname__}' == 'typesystem.fields.String'
    assert var_28.additional_items is False
    assert var_28.min_items is None
    assert var_28.max_items is None
    assert var_28.unique_items is True
    var_29 = module_0.Integer()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Integer'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert var_29.minimum is None
    assert var_29.maximum is None
    assert var_29.exclusive_minimum is None
    assert var_29.exclusive_maximum is None
    assert var_29.multiple_of is None
    assert var_29.precision is None
    assert var_29.coerce_types is True
    var_30 = False
    var_31 = module_0.Array(var_29, var_30, var_10, var_19, unique_items=var_10)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Array'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert f'{type(var_31.items).__module__}.{type(var_31.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_31.additional_items is False
    assert var_31.min_items is True
    assert var_31.max_items == 5
    assert var_31.unique_items is True
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
    var_33 = 2
    var_34 = module_0.Array(var_32, min_items=var_33, max_items=var_22, exact_items=var_19)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Array'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.items).__module__}.{type(var_34.items).__qualname__}' == 'typesystem.fields.String'
    assert var_34.additional_items is False
    assert var_34.min_items == 5
    assert var_34.max_items == 5
    assert var_34.unique_items is False
    var_35 = module_0.String()
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.String'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.allow_blank is False
    assert var_35.trim_whitespace is True
    assert var_35.max_length is None
    assert var_35.min_length is None
    assert var_35.format is None
    assert var_35.coerce_types is True
    assert var_35.pattern is None
    assert var_35.pattern_regex is None
    var_36 = module_0.Integer()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Integer'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.minimum is None
    assert var_36.maximum is None
    assert var_36.exclusive_minimum is None
    assert var_36.exclusive_maximum is None
    assert var_36.multiple_of is None
    assert var_36.precision is None
    assert var_36.coerce_types is True
    var_37 = module_0.Boolean()
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.coerce_types is True
    var_38 = [var_35, var_36, var_37]
    var_39 = module_0.Array(var_38, var_30)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Array'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.items).__module__}.{type(var_39.items).__qualname__}' == 'builtins.list'
    assert len(var_39.items) == 3
    assert var_39.additional_items is False
    assert var_39.min_items == 3
    assert var_39.max_items == 3
    assert var_39.unique_items is False
    var_40 = module_0.Array(var_38, var_10)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Array'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert f'{type(var_40.items).__module__}.{type(var_40.items).__qualname__}' == 'builtins.list'
    assert len(var_40.items) == 3
    assert var_40.additional_items is True
    assert var_40.min_items == 3
    assert var_40.max_items is None
    assert var_40.unique_items is False
    var_41 = module_0.Array(var_38, var_30, var_10, var_22)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Array'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.items).__module__}.{type(var_41.items).__qualname__}' == 'builtins.list'
    assert len(var_41.items) == 3
    assert var_41.additional_items is False
    assert var_41.min_items is True
    assert var_41.max_items == 10
    assert var_41.unique_items is False

def test_case_118():
    var_0 = module_0.String()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = [var_0]
    var_2 = module_0.Array(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'builtins.list'
    assert len(var_2.items) == 1
    assert f'{type(var_2.additional_items).__module__}.{type(var_2.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_2.min_items == 1
    assert var_2.max_items is None
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_119():
    var_0 = module_0.String()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = module_1.system()
    assert var_1 == 'Linux'
    var_2 = module_0.Boolean()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.coerce_types is True
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_3 = [var_0]
    var_4 = module_0.Array(var_3, var_2, exact_items=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.items).__module__}.{type(var_4.items).__qualname__}' == 'builtins.list'
    assert len(var_4.items) == 1
    assert f'{type(var_4.additional_items).__module__}.{type(var_4.additional_items).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_4.min_items == 'Linux'
    assert var_4.max_items == 'Linux'
    assert var_4.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_5 = var_4.validate_or_error(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1

def test_case_120():
    var_0 = True
    var_1 = None
    var_2 = module_0.String(allow_blank=var_0, max_length=var_1, format=var_1, coerce_types=var_1)
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
    assert var_2.coerce_types is None
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_121():
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
    var_1 = module_2.uuid4()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'uuid.UUID'
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
    var_2 = None
    var_3 = module_0.Const(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Const'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.const is None
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_4 = var_0.validate(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'uuid.UUID'

def test_case_122():
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
    var_1 = []
    var_2 = module_0.Array(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items == []
    assert f'{type(var_2.additional_items).__module__}.{type(var_2.additional_items).__qualname__}' == 'typesystem.fields.Date'
    assert var_2.min_items == 0
    assert var_2.max_items is None
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == []
    assert var_3.error is None

def test_case_123():
    var_0 = module_0.String()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = None
    var_2 = module_0.String(allow_blank=var_0, max_length=var_1, min_length=var_1, pattern=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.default == ''
    assert f'{type(var_2.allow_blank).__module__}.{type(var_2.allow_blank).__qualname__}' == 'typesystem.fields.String'
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == ''
    assert var_3.error is None

def test_case_124():
    var_0 = 'name'
    var_1 = module_0.String()
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
    var_3 = {var_0: var_1, var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.properties).__module__}.{type(var_4.properties).__qualname__}' == 'builtins.dict'
    assert len(var_4.properties) == 1
    assert var_4.pattern_properties == {}
    assert var_4.additional_properties is True
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = 16
    var_6 = {var_0: var_0, var_0: var_5}
    var_7 = var_4.validate(var_6)

def test_case_125():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_1.freedesktop_os_release()
    var_3 = module_0.Integer()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_4 = {var_0: var_2, var_1: var_3}
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_4)

def test_case_126():
    var_0 = 'J7`a\x0b\rjb?AA6]'
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
    var_2 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert f'{type(var_3.pattern_properties).__module__}.{type(var_3.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_3.pattern_properties) == 1
    assert var_3.additional_properties is True
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = var_3.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.value).__module__}.{type(var_4.value).__qualname__}' == 'builtins.dict'
    assert len(var_4.value) == 1
    assert var_4.error is None

def test_case_127():
    var_0 = '=#p2ytw'
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
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
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
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = {var_3: var_2, var_1: var_1}
    var_5 = var_3.validate_or_error(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 2

@pytest.mark.xfail(strict=True)
def test_case_128():
    var_0 = None
    var_1 = module_0.String(allow_blank=var_0, trim_whitespace=var_0, min_length=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is None
    assert var_1.trim_whitespace is None
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
    var_2 = module_1.python_branch()
    assert var_2 == ''
    var_3 = var_1.validate_or_error(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = module_1.processor()
    assert var_4 == ''
    var_5 = module_0.Object(required=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.properties == {}
    assert var_5.pattern_properties == {}
    assert var_5.additional_properties is True
    assert var_5.property_names is None
    assert var_5.min_properties is None
    assert var_5.max_properties is None
    assert var_5.required == ''
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4.__setitem__(var_3, var_0)

def test_case_129():
    var_0 = 'name'
    var_1 = '4rx-[V'
    var_2 = module_0.String()
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
    var_3 = module_0.Integer()
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
    var_5 = module_0.Object(properties=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.properties).__module__}.{type(var_5.properties).__qualname__}' == 'builtins.dict'
    assert len(var_5.properties) == 2
    assert var_5.pattern_properties == {}
    assert var_5.additional_properties is True
    assert var_5.property_names is None
    assert var_5.min_properties is None
    assert var_5.max_properties is None
    assert var_5.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = {var_0: var_4, var_3: var_2, var_3: var_3}
    var_7 = None
    with pytest.raises(AssertionError):
        module_0.Object(pattern_properties=var_6, property_names=var_5, min_properties=var_7, required=var_7)

def test_case_130():
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
    var_1 = None
    var_2 = False
    var_3 = module_0.String(allow_blank=var_1, min_length=var_1, coerce_types=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is None
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format is None
    assert var_3.coerce_types is False
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_1.python_branch()
    assert var_4 == ''
    var_5 = var_0.validate_or_error(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = module_1.processor()
    assert var_6 == ''
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_1, pattern_properties=var_1, property_names=var_1, required=var_5)

def test_case_131():
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
    var_1 = module_0.Object(properties=var_0, additional_properties=var_0, property_names=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert f'{type(var_1.additional_properties).__module__}.{type(var_1.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_1.property_names).__module__}.{type(var_1.property_names).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_1.freedesktop_os_release()
    var_3 = var_1.validate_or_error(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 10

def test_case_132():
    var_0 = 'sQ\nPD6WM+6rq'
    var_1 = module_0.String()
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
    var_3 = module_0.Object(properties=var_2, additional_properties=var_2, property_names=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert f'{type(var_3.additional_properties).__module__}.{type(var_3.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_3.property_names).__module__}.{type(var_3.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = {var_0: var_3, var_2: var_1, var_2: var_2}
    var_5 = var_3.validate_or_error(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 2

def test_case_133():
    var_0 = module_3.purge()
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
    var_1 = module_0.Object(properties=var_0, additional_properties=var_0, property_names=var_0)
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

def test_case_134():
    var_0 = module_0.String()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = [var_0, var_0, var_0]
    var_2 = None
    var_3 = module_0.Array(max_items=var_2)
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
    var_4 = var_3.validate_or_error(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.value).__module__}.{type(var_4.value).__qualname__}' == 'builtins.list'
    assert len(var_4.value) == 3
    assert var_4.error is None

def test_case_135():
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
    var_1 = -67.68417356122347
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_136():
    var_0 = module_0.String()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = var_0.has_default()
    assert var_1 is False
    var_2 = [var_0, var_0, var_0]
    var_3 = module_0.Array(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'builtins.list'
    assert len(var_3.items) == 3
    assert var_3.additional_items is False
    assert var_3.min_items == 3
    assert var_3.max_items == 3
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = var_3.__or__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = var_4.validate_or_error(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1

def test_case_137():
    var_0 = module_0.String()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = var_0.has_default()
    assert var_1 is False
    var_2 = [var_0, var_0, var_0]
    var_3 = module_0.Array(var_0, var_0, unique_items=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_3.additional_items).__module__}.{type(var_3.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_3.min_items is None
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.Date()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Date'
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
    var_5 = module_0.Object(max_properties=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.properties == {}
    assert var_5.pattern_properties == {}
    assert var_5.additional_properties is True
    assert var_5.property_names is None
    assert var_5.min_properties is None
    assert var_5.max_properties is False
    assert var_5.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = var_3.__or__(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Union'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.any_of).__module__}.{type(var_6.any_of).__qualname__}' == 'builtins.list'
    assert len(var_6.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_7 = var_6.validate_or_error(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1

def test_case_138():
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
    var_1 = -68.0
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == -68
    assert var_2.error is None

def test_case_139():
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
    var_1 = module_0.Object(properties=var_0, additional_properties=var_0, property_names=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert f'{type(var_1.additional_properties).__module__}.{type(var_1.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_1.property_names).__module__}.{type(var_1.property_names).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = var_1.has_default()
    assert var_2 is False
    var_3 = {}
    var_4 = None
    var_5 = module_0.Object(properties=var_3, property_names=var_0, max_properties=var_2, required=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.properties == {}
    assert var_5.pattern_properties == {}
    assert var_5.additional_properties is True
    assert f'{type(var_5.property_names).__module__}.{type(var_5.property_names).__qualname__}' == 'typesystem.fields.Integer'
    assert var_5.min_properties is None
    assert var_5.max_properties is False
    assert var_5.required == []
    var_6 = var_5.validate_or_error(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value == {}
    assert var_6.error is None

def test_case_140():
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
    var_1 = True
    var_2 = module_0.Object(properties=var_0, max_properties=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert f'{type(var_2.additional_properties).__module__}.{type(var_2.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.property_names is None
    assert var_2.min_properties is None
    assert var_2.max_properties is True
    assert var_2.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = module_1.freedesktop_os_release()
    var_4 = var_2.validate_or_error(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_141():
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
    var_1 = var_0.__or__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = module_1.python_revision()
    assert var_2 == ''
    var_3 = var_1.validate_or_error(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_142():
    var_0 = module_1.win32_is_iot()
    assert var_0 is False
    var_1 = module_1.freedesktop_os_release()
    var_2 = module_1.freedesktop_os_release()
    var_3 = module_0.URL()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_4 = module_0.Object(additional_properties=var_0, required=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert var_4.pattern_properties == {}
    assert var_4.additional_properties is False
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == {'NAME': 'Debian GNU/Linux', 'ID': 'debian', 'PRETTY_NAME': 'Debian GNU/Linux 13 (trixie)', 'VERSION_ID': '13', 'VERSION': '13 (trixie)', 'VERSION_CODENAME': 'trixie', 'DEBIAN_VERSION_FULL': '13.3', 'HOME_URL': 'https://www.debian.org/', 'SUPPORT_URL': 'https://www.debian.org/support', 'BUG_REPORT_URL': 'https://bugs.debian.org/'}
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = var_4.validate_or_error(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 10
    var_6 = var_3.validate_or_error(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1