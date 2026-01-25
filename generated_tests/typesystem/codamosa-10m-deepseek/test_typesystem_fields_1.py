# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.fields as module_0
import re as module_1
import platform as module_2
import ipaddress as module_3
import enum as module_4
import collections as module_5
import typesystem.base as module_6

def test_case_0():
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
def test_case_1():
    var_0 = module_1.RegexFlag.MULTILINE
    var_1 = module_0.Number()
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
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_2 = var_1.validate(var_0)
    assert var_2 == module_1.RegexFlag.MULTILINE
    assert module_1.RegexFlag.ASCII == module_1.RegexFlag.ASCII
    assert module_1.RegexFlag.A == module_1.RegexFlag.ASCII
    assert module_1.RegexFlag.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.RegexFlag.I == module_1.RegexFlag.IGNORECASE
    assert module_1.RegexFlag.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.RegexFlag.L == module_1.RegexFlag.LOCALE
    assert module_1.RegexFlag.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.RegexFlag.U == module_1.RegexFlag.UNICODE
    assert module_1.RegexFlag.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.RegexFlag.M == module_1.RegexFlag.MULTILINE
    assert module_1.RegexFlag.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.RegexFlag.S == module_1.RegexFlag.DOTALL
    assert module_1.RegexFlag.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.RegexFlag.X == module_1.RegexFlag.VERBOSE
    assert module_1.RegexFlag.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.RegexFlag.T == module_1.RegexFlag.TEMPLATE
    assert module_1.RegexFlag.DEBUG == module_1.RegexFlag.DEBUG
    module_0.Any(title=var_2, description=var_1)

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
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1

def test_case_3():
    var_0 = None
    var_1 = module_0.Object(properties=var_0, min_properties=var_0)
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
    var_2 = var_1.get_default_value()

def test_case_4():
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

def test_case_5():
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

def test_case_6():
    var_0 = None
    var_1 = True
    var_2 = module_0.String(allow_blank=var_1, min_length=var_0)
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
def test_case_7():
    var_0 = 'NSN.g('
    var_1 = None
    var_2 = module_0.String(trim_whitespace=var_1, max_length=var_1, coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is None
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
    var_3 = None
    var_4 = var_2.validate(var_0)
    assert var_4 == 'NSN.g('
    var_4.validate_or_error(var_3)

def test_case_8():
    var_0 = module_2.version()
    assert var_0 == '#202511241048~1764607909~24.04~df6b2b6 SMP PREEMPT_DYNAMIC Mon D'
    var_1 = module_0.DateTime()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'datetime'
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

def test_case_9():
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

def test_case_10():
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
    var_1 = var_0.validate_or_error(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert f'{type(var_1.error).__module__}.{type(var_1.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1.error) == 1

def test_case_11():
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
    var_1 = module_2.python_branch()
    assert var_1 == ''
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_12():
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
    var_1 = module_2.python_branch()
    assert var_1 == ''
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

def test_case_13():
    var_0 = -1229
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Choice'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.choices == []
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_2 = 'y?odLnO\x0b_.i`4b'
    var_3 = module_0.Integer(maximum=var_0, precision=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum == -1229
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision == 'y?odLnO\x0b_.i`4b'
    assert var_3.coerce_types is True

def test_case_14():
    var_0 = module_0.Number()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Number'
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
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 6
    var_1 = False
    var_2 = module_0.Number(minimum=var_0, maximum=var_1, exclusive_minimum=var_0, exclusive_maximum=var_1, precision=var_1, multiple_of=var_1, coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Number'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum == 6
    assert var_2.maximum is False
    assert var_2.exclusive_minimum == 6
    assert var_2.exclusive_maximum is False
    assert var_2.multiple_of is False
    assert var_2.precision is False
    assert var_2.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_2.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = module_0.Choice(choices=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Choice'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.choices == []
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_1.validate(var_1)

def test_case_17():
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

def test_case_18():
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

def test_case_19():
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

def test_case_20():
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
    var_2 = module_0.Number()
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
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = var_2.serialize(var_1)

def test_case_21():
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

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    var_2 = 3
    var_3 = module_0.Array(var_0, min_items=var_2, max_items=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.additional_items is False
    assert var_3.min_items == 3
    assert var_3.max_items == 3
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = None
    var_3.validate(var_4)

def test_case_24():
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

def test_case_25():
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

def test_case_26():
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

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = module_0.Number()
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
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_0.Number()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Number'
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
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_0.validate_or_error(var_0)

def test_case_30():
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

def test_case_31():
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
    var_1 = 'NSN.g('
    var_2 = True
    var_3 = None
    var_4 = module_0.Any(description=var_1, allow_null=var_2, read_only=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == 'NSN.g('
    assert var_4.allow_null is True
    assert var_4.read_only is None
    var_5 = None
    var_6 = module_0.Array(exact_items=var_5)
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
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}

def test_case_32():
    var_0 = module_2.win32_edition()
    var_1 = 'nGBe()1]W*{|1?['
    var_2 = module_0.Field(description=var_1, default=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == 'nGBe()1]W*{|1?['
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
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
    var_4 = var_3.validate_or_error(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_33():
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
    var_3 = module_0.Object(properties=var_1, additional_properties=var_1, max_properties=var_1, required=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is None
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_34():
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
    var_1.validate(var_0)

def test_case_35():
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
    var_1 = {}
    var_2 = None
    var_3 = module_0.Object(additional_properties=var_2, property_names=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is None
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = var_3.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = None
    var_1 = module_0.Array(var_0, max_items=var_0)
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
    var_2 = var_1.serialize(var_0)
    var_3 = False
    var_4 = module_0.Field(default=var_0, allow_null=var_1, read_only=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert f'{type(var_4.allow_null).__module__}.{type(var_4.allow_null).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.read_only is False
    assert module_0.Field.errors == {}
    var_5 = None
    var_6 = module_0.Object(property_names=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.properties == {}
    assert var_6.pattern_properties == {}
    assert var_6.additional_properties is True
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = module_2.node()
    assert var_7 == 'e4e092304153'
    var_8 = var_7.isidentifier()
    assert var_8 is True
    var_1.isdisjoint(var_5)

def test_case_37():
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
    var_1 = module_0.DateTime()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'datetime'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    var_2 = None
    var_3 = var_0.serialize(var_2)
    var_4 = None
    var_5 = module_0.Boolean(coerce_types=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.coerce_types is None
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_6 = module_2.python_branch()
    assert var_6 == ''
    var_7 = module_0.URL()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.URL'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format == 'url'
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None

def test_case_38():
    var_0 = None
    with pytest.raises(AssertionError):
        module_0.Field(description=var_0, default=var_0)

def test_case_39():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Number(minimum=var_1, maximum=var_1, coerce_types=var_1, **var_2)
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
    assert var_3.coerce_types is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4 = var_3.__or__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_3.collapse_addresses(var_2)
    assert module_3.IPV4LENGTH == 32
    assert module_3.IPV6LENGTH == 128
    var_6 = module_0.DateTime()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.allow_blank is False
    assert var_6.trim_whitespace is True
    assert var_6.max_length is None
    assert var_6.min_length is None
    assert var_6.format == 'datetime'
    assert var_6.coerce_types is True
    assert var_6.pattern is None
    assert var_6.pattern_regex is None
    var_7 = var_6.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = -1720
    var_1 = None
    var_2 = module_0.Decimal(exclusive_minimum=var_0, multiple_of=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum == -1720
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_2.validate(var_0)

def test_case_41():
    var_0 = module_1.RegexFlag.MULTILINE
    var_1 = module_0.Number()
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
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    with pytest.raises(AssertionError):
        module_0.Array(var_0, unique_items=var_0)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = module_1.RegexFlag.MULTILINE
    var_1 = module_0.Number()
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
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_2 = module_0.Array(var_1, exact_items=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.Number'
    assert var_2.additional_items is False
    assert var_2.min_items == module_1.RegexFlag.MULTILINE
    assert var_2.max_items == module_1.RegexFlag.MULTILINE
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = None
    var_2.validate(var_3)

def test_case_43():
    var_0 = 'NSN.g('
    var_1 = False
    var_2 = module_0.String(trim_whitespace=var_1, max_length=var_1, coerce_types=var_1)
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
    assert var_2.coerce_types is False
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = None
    var_4 = module_0.Any(description=var_0, allow_null=var_1, read_only=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == 'NSN.g('
    assert var_4.allow_null is False
    assert var_4.read_only is None
    var_5 = None
    var_6 = module_0.Number(exclusive_minimum=var_5, exclusive_maximum=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Number'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.minimum is None
    assert var_6.maximum is None
    assert var_6.exclusive_minimum is None
    assert var_6.exclusive_maximum is False
    assert var_6.multiple_of is None
    assert var_6.precision is None
    assert var_6.coerce_types is True
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_7 = var_6.serialize(var_0)
    assert var_7 == 'NSN.g('

def test_case_44():
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
    var_2 = True
    with pytest.raises(AssertionError):
        module_0.Number(maximum=var_0, exclusive_minimum=var_1, coerce_types=var_2)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = None
    var_1 = False
    var_2 = module_0.Object(additional_properties=var_0, max_properties=var_0, required=var_0)
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
    var_3 = module_0.Number()
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
    var_3.validate(var_1)

def test_case_46():
    var_0 = None
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
    var_2 = var_1.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = module_0.Array(var_0, var_1, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.items is None
    assert f'{type(var_3.additional_items).__module__}.{type(var_3.additional_items).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.min_items is None
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = var_1.validate_or_error(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_47():
    var_0 = module_0.Number()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Number'
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
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_1 = None
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = module_0.IPAddress()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format == 'ipaddress'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    with pytest.raises(AssertionError):
        module_0.Array(additional_items=var_1, max_items=var_1, unique_items=var_1)

@pytest.mark.xfail(strict=True)
def test_case_48():
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
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.Union(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 4
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = None
    var_4 = module_0.Object(property_names=var_3)
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
    var_5 = var_4.serialize(var_3)
    var_6 = var_4.validate_or_error(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_2.__or__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_49():
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
def test_case_50():
    var_0 = module_2.python_compiler()
    assert var_0 == 'GCC 14.2.0'
    var_1 = module_0.Number()
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
    var_1.validate(var_0)

def test_case_51():
    var_0 = None
    var_1 = True
    var_2 = module_0.String(allow_blank=var_1, min_length=var_0)
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

def test_case_52():
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
    var_1 = module_4._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = None
    var_3 = ()
    with pytest.raises(AssertionError):
        module_0.Number(minimum=var_3, exclusive_maximum=var_2, precision=var_1, multiple_of=var_2)

def test_case_53():
    var_0 = None
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
    var_2 = var_1.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    with pytest.raises(AssertionError):
        module_0.Number(maximum=var_0, exclusive_minimum=var_2)

@pytest.mark.xfail(strict=True)
def test_case_54():
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
    var_1 = "*k*|'6seJMM3"
    var_2 = '\x0c_0'
    var_3 = {var_1: var_0, var_1: var_0, var_2: var_0, var_2: var_0}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.properties == {}
    assert f'{type(var_5.pattern_properties).__module__}.{type(var_5.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_5.pattern_properties) == 2
    assert var_5.additional_properties is True
    assert var_5.property_names is None
    assert var_5.min_properties is None
    assert var_5.max_properties is None
    assert var_5.required == []
    var_6 = None
    var_0.validate(var_6)

def test_case_55():
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
    var_1 = module_1.RegexFlag.MULTILINE
    var_2 = module_0.Number()
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
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = var_2.validate(var_1)
    assert var_3 == module_1.RegexFlag.MULTILINE
    assert module_1.RegexFlag.ASCII == module_1.RegexFlag.ASCII
    assert module_1.RegexFlag.A == module_1.RegexFlag.ASCII
    assert module_1.RegexFlag.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.RegexFlag.I == module_1.RegexFlag.IGNORECASE
    assert module_1.RegexFlag.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.RegexFlag.L == module_1.RegexFlag.LOCALE
    assert module_1.RegexFlag.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.RegexFlag.U == module_1.RegexFlag.UNICODE
    assert module_1.RegexFlag.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.RegexFlag.M == module_1.RegexFlag.MULTILINE
    assert module_1.RegexFlag.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.RegexFlag.S == module_1.RegexFlag.DOTALL
    assert module_1.RegexFlag.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.RegexFlag.X == module_1.RegexFlag.VERBOSE
    assert module_1.RegexFlag.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.RegexFlag.T == module_1.RegexFlag.TEMPLATE
    assert module_1.RegexFlag.DEBUG == module_1.RegexFlag.DEBUG
    with pytest.raises(AssertionError):
        module_0.Array(unique_items=var_3)

def test_case_56():
    var_0 = None
    var_1 = True
    var_2 = module_0.String(trim_whitespace=var_0, min_length=var_1, pattern=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is None
    assert var_2.max_length is None
    assert var_2.min_length is True
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_2.python_version_tuple()

def test_case_57():
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
    var_1 = module_2.python_branch()
    assert var_1 == ''
    with pytest.raises(AssertionError):
        module_0.Number(multiple_of=var_1)

def test_case_58():
    var_0 = None
    var_1 = True
    var_2 = module_0.Object(property_names=var_0, max_properties=var_1, required=var_0)
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
    assert var_2.max_properties is True
    assert var_2.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_59():
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
    var_1 = True
    var_2 = None
    with pytest.raises(AssertionError):
        module_0.String(allow_blank=var_2, trim_whitespace=var_1, max_length=var_0, format=var_2, coerce_types=var_2)

def test_case_60():
    var_0 = module_2.python_revision()
    assert var_0 == ''
    var_1 = module_0.String(allow_blank=var_0, pattern=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank == ''
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern == ''
    assert f'{type(var_1.pattern_regex).__module__}.{type(var_1.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}

@pytest.mark.xfail(strict=True)
def test_case_61():
    var_0 = module_2.win32_is_iot()
    assert var_0 is False
    var_1 = module_2.python_branch()
    assert var_1 == ''
    var_2 = None
    var_3 = module_0.Number(minimum=var_2, coerce_types=var_2)
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
    assert var_3.coerce_types is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3.validate(var_1)

def test_case_62():
    var_0 = module_0.Number()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Number'
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
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_1 = module_2.python_revision()
    assert var_1 == ''
    var_2 = var_0.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_63():
    var_0 = None
    var_1 = module_0.Choice(choices=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Choice'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.choices == []
    assert var_1.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_1.validate(var_0)

def test_case_64():
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
    var_1 = 3
    var_2 = module_0.Array(var_0, min_items=var_1, max_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.additional_items is False
    assert var_2.min_items == 3
    assert var_2.max_items == 3
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = [var_1, var_1, var_1]
    var_4 = var_2.validate(var_3)

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

def test_case_66():
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
    var_1 = '&\\e.">e(y_%w<deS&'
    var_2 = module_2.python_build()
    var_3 = ''
    var_4 = 'g,g-I$3)*#eohPY\t'
    var_5 = {var_1: var_2, var_3: var_0, var_4: var_2}
    with pytest.raises(AssertionError):
        module_0.Object(pattern_properties=var_5)

def test_case_67():
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
    var_2 = module_0.Number(minimum=var_1)
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
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = var_0.__or__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = None
    var_5 = module_0.Number(minimum=var_1, maximum=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Number'
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
    with pytest.raises(AssertionError):
        module_0.Number(minimum=var_1, exclusive_maximum=var_5, multiple_of=var_1)

def test_case_68():
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
    var_2 = var_1.has_default()
    assert var_2 is False
    var_3 = module_0.Field()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert module_0.Field.errors == {}
    var_4 = var_1.validate(var_0)

def test_case_69():
    var_0 = module_2.python_compiler()
    assert var_0 == 'GCC 14.2.0'
    with pytest.raises(AssertionError):
        module_0.Object(min_properties=var_0)

@pytest.mark.xfail(strict=True)
def test_case_70():
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
    var_1 = False
    var_2 = None
    var_3 = module_0.Object(properties=var_2, additional_properties=var_2, property_names=var_2, min_properties=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is None
    assert var_3.property_names is None
    assert var_3.min_properties is False
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_0.__iter__()

def test_case_71():
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
    var_1 = module_0.Object(additional_properties=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert f'{type(var_1.additional_properties).__module__}.{type(var_1.additional_properties).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_1.property_names is None
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = None
    var_3 = var_0.validate_or_error(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_72():
    var_0 = None
    var_1 = True
    var_2 = module_0.String(trim_whitespace=var_0, min_length=var_1, pattern=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is None
    assert var_2.max_length is None
    assert var_2.min_length is True
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.Any(default=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    with pytest.raises(AssertionError):
        module_0.Array(var_0, var_3, var_3, var_0, var_1)

def test_case_73():
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
    var_1 = module_2.python_branch()
    assert var_1 == ''
    var_2 = None
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_2, max_properties=var_1, required=var_1)

def test_case_74():
    var_0 = None
    var_1 = module_2.python_branch()
    assert var_1 == ''
    var_2 = module_0.Object(properties=var_1, pattern_properties=var_0)
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
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_75():
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
    var_1 = module_2.python_build()
    var_2 = module_0.Object(properties=var_0, additional_properties=var_0, required=var_1)
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
    assert var_2.required == ['main', 'Jan 13 2026 03:14:08']
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = module_5.ChainMap(*var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.ChainMap'
    assert len(var_3) == 14
    assert f'{type(module_5.ChainMap.fromkeys).__module__}.{type(module_5.ChainMap.fromkeys).__qualname__}' == 'builtins.method'
    assert f'{type(module_5.ChainMap.parents).__module__}.{type(module_5.ChainMap.parents).__qualname__}' == 'builtins.property'
    var_2.validate(var_3)

def test_case_76():
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
    var_1 = module_2.version()
    assert var_1 == '#202511241048~1764607909~24.04~df6b2b6 SMP PREEMPT_DYNAMIC Mon D'
    var_2 = module_2.python_compiler()
    assert var_2 == 'GCC 14.2.0'
    var_3 = module_2.python_branch()
    assert var_3 == ''
    var_4 = None
    with pytest.raises(AssertionError):
        module_0.Object(pattern_properties=var_3, additional_properties=var_2, max_properties=var_4)

def test_case_77():
    var_0 = None
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
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_78():
    var_0 = None
    var_1 = module_0.Array(max_items=var_0)
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
    var_1.validate(var_0)

def test_case_79():
    var_0 = None
    var_1 = module_0.Object(additional_properties=var_0, max_properties=var_0, required=var_0)
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
    var_2 = module_0.Number()
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
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_3 = True
    with pytest.raises(AssertionError):
        module_0.String(pattern=var_3, format=var_3, coerce_types=var_3)

def test_case_80():
    var_0 = module_1.RegexFlag.MULTILINE
    var_1 = module_0.Number()
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
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_2 = var_1.validate(var_0)
    assert var_2 == module_1.RegexFlag.MULTILINE
    assert module_1.RegexFlag.ASCII == module_1.RegexFlag.ASCII
    assert module_1.RegexFlag.A == module_1.RegexFlag.ASCII
    assert module_1.RegexFlag.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.RegexFlag.I == module_1.RegexFlag.IGNORECASE
    assert module_1.RegexFlag.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.RegexFlag.L == module_1.RegexFlag.LOCALE
    assert module_1.RegexFlag.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.RegexFlag.U == module_1.RegexFlag.UNICODE
    assert module_1.RegexFlag.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.RegexFlag.M == module_1.RegexFlag.MULTILINE
    assert module_1.RegexFlag.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.RegexFlag.S == module_1.RegexFlag.DOTALL
    assert module_1.RegexFlag.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.RegexFlag.X == module_1.RegexFlag.VERBOSE
    assert module_1.RegexFlag.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.RegexFlag.T == module_1.RegexFlag.TEMPLATE
    assert module_1.RegexFlag.DEBUG == module_1.RegexFlag.DEBUG

@pytest.mark.xfail(strict=True)
def test_case_81():
    var_0 = '|.\t-fRCf_=~})'
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
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = module_2.python_build()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_1, required=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.properties).__module__}.{type(var_4.properties).__qualname__}' == 'builtins.dict'
    assert len(var_4.properties) == 1
    assert var_4.pattern_properties == {}
    assert f'{type(var_4.additional_properties).__module__}.{type(var_4.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == ['main', 'Jan 13 2026 03:14:08']
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_82():
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
    var_1 = module_2.uname()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'platform.uname_result'
    assert len(var_1) == 6
    assert f'{type(module_2.uname_result.processor).__module__}.{type(module_2.uname_result.processor).__qualname__}' == 'functools.cached_property'
    assert module_2.uname_result.processor.attrname == 'processor'
    assert f'{type(module_2.uname_result.processor.lock).__module__}.{type(module_2.uname_result.processor.lock).__qualname__}' == '_thread.RLock'
    var_2 = module_0.Choice(choices=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Choice'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.choices == [('Linux', 'Linux'), ('e4e092304153', 'e4e092304153'), ('6.17.9-76061709-generic', '6.17.9-76061709-generic'), ('#202511241048~1764607909~24.04~df6b2b6 SMP PREEMPT_DYNAMIC Mon D', '#202511241048~1764607909~24.04~df6b2b6 SMP PREEMPT_DYNAMIC Mon D'), ('x86_64', 'x86_64'), ('', '')]
    assert var_2.coerce_types is True
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_3 = True
    module_0.Number(exclusive_maximum=var_1, precision=var_1, coerce_types=var_3, **var_1)

def test_case_83():
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
    var_1 = module_0.DateTime()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'datetime'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    var_2 = None
    var_3 = module_0.Boolean(coerce_types=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.coerce_types is None
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_4 = module_2.python_branch()
    assert var_4 == ''
    var_5 = var_4.isdigit()
    assert var_5 is False
    var_6 = module_0.Object(required=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Object'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.properties == {}
    assert var_6.pattern_properties == {}
    assert var_6.additional_properties is True
    assert var_6.property_names is None
    assert var_6.min_properties is None
    assert var_6.max_properties is None
    assert var_6.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = var_3.validate_or_error(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1

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
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = module_2.python_branch()
    assert var_1 == ''
    var_2 = var_1.isdigit()
    assert var_2 is False
    var_3 = var_0.validate(var_1)
    assert var_3 is False

def test_case_85():
    var_0 = module_2.python_compiler()
    assert var_0 == 'GCC 14.2.0'
    var_1 = module_0.IPAddress()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.IPAddress'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'ipaddress'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    with pytest.raises(AssertionError):
        module_0.Array(var_1, max_items=var_0)

def test_case_86():
    var_0 = None
    var_1 = []
    var_2 = module_0.Union(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.any_of == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_87():
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
    var_0.validate(var_0)

def test_case_88():
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
    var_2 = module_0.Boolean(coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.coerce_types is None
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_3 = module_2.python_branch()
    assert var_3 == ''
    var_4 = var_3.isdigit()
    assert var_4 is False
    var_5 = module_0.Object(required=var_1)
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
    assert var_5.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = var_2.validate_or_error(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is False
    assert var_6.error is None

def test_case_89():
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
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == {}
    assert var_2.error is None

def test_case_90():
    var_0 = False
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
    var_2 = module_0.Any(read_only=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    var_3 = None
    var_4 = var_2.validate_or_error(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None

def test_case_91():
    var_0 = module_0.Number()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Number'
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
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_1 = None
    var_2 = module_0.Object(pattern_properties=var_1, property_names=var_1, max_properties=var_1)
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
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = module_2.python_revision()
    assert var_4 == ''
    with pytest.raises(AssertionError):
        module_0.String(min_length=var_4, pattern=var_1)

def test_case_92():
    var_0 = {}
    var_1 = None
    var_2 = module_0.Object(additional_properties=var_1, property_names=var_1)
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
    var_3 = var_2.validate(var_0)

def test_case_93():
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
    var_1 = module_2.python_branch()
    assert var_1 == ''
    var_2 = var_1.isdigit()
    assert var_2 is False
    var_3 = var_0.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1

def test_case_94():
    var_0 = None
    var_1 = True
    var_2 = module_0.String(allow_blank=var_1, min_length=var_0)
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
    var_3 = var_2.validate(var_0)
    assert var_3 == ''
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

@pytest.mark.xfail(strict=True)
def test_case_95():
    var_0 = None
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1, read_only=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_2.validate_or_error(var_0)

def test_case_96():
    var_0 = 'Q?\x0ca'
    var_1 = None
    var_2 = module_2.win32_ver(version=var_1, csd=var_1)
    var_3 = '\nWQP&'
    var_4 = '=gGV5r"n'
    var_5 = {var_0: var_2, var_3: var_2, var_4: var_2}
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_5, pattern_properties=var_1, required=var_1)

def test_case_97():
    var_0 = None
    var_1 = True
    var_2 = module_0.Float(exclusive_maximum=var_0, multiple_of=var_0, coerce_types=var_1)
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
    assert var_2.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_3 = module_0.Object(additional_properties=var_0, property_names=var_0, min_properties=var_0, required=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is None
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = module_0.Const(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Const'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.const is None
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_5 = var_3.validate_or_error(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = var_4.validate_or_error(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1

def test_case_98():
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
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

def test_case_99():
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
    var_1 = module_0.DateTime()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'datetime'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    var_2 = None
    var_3 = module_0.Object(properties=var_1, additional_properties=var_2, property_names=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert f'{type(var_3.additional_properties).__module__}.{type(var_3.additional_properties).__qualname__}' == 'typesystem.fields.DateTime'
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = var_0.validate_or_error(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

def test_case_100():
    var_0 = None
    var_1 = module_0.Object(max_properties=var_0, required=var_0)
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
    var_2 = module_0.Const(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Const'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.const).__module__}.{type(var_2.const).__qualname__}' == 'typesystem.fields.Object'
    assert module_0.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = var_2.validate_or_error(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1

@pytest.mark.xfail(strict=True)
def test_case_101():
    var_0 = True
    var_1 = None
    var_2 = module_0.String(allow_blank=var_0, coerce_types=var_1)
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
    var_2.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_102():
    var_0 = 20
    var_1 = lambda : var_0
    var_2 = module_0.Field(default=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_2.get_default_value()
    assert var_3 == 20

def test_case_103():
    var_0 = module_1.purge()
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    var_1 = module_0.Array(var_0, min_items=var_0, max_items=var_0)
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
    var_2 = [var_0, var_0, var_0]
    var_3 = var_1.validate(var_2)

def test_case_104():
    var_0 = module_2.win32_is_iot()
    assert var_0 is False
    var_1 = 3
    var_2 = None
    var_3 = module_0.Array(var_2, min_items=var_0, max_items=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.items is None
    assert var_3.additional_items is False
    assert var_3.min_items is False
    assert var_3.max_items == 3
    assert var_3.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = [var_1, var_1, var_1]
    var_5 = var_3.validate(var_4)

def test_case_105():
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
    var_1 = 43
    var_2 = 3
    var_3 = module_0.Array(var_0, min_items=var_1, max_items=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.additional_items is False
    assert var_3.min_items == 43
    assert var_3.max_items == 3
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_2.python_revision()
    assert var_4 == ''
    var_5 = [var_1]
    with pytest.raises(module_6.ValidationError):
        var_3.validate(var_5)

def test_case_106():
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
    var_1 = 1
    var_2 = 3
    var_3 = module_0.Array(var_0, min_items=var_1, max_items=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.additional_items is False
    assert var_3.min_items == 1
    assert var_3.max_items == 3
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_2.uname()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'platform.uname_result'
    assert len(var_4) == 6
    assert f'{type(module_2.uname_result.processor).__module__}.{type(module_2.uname_result.processor).__qualname__}' == 'functools.cached_property'
    assert module_2.uname_result.processor.attrname == 'processor'
    assert f'{type(module_2.uname_result.processor.lock).__module__}.{type(module_2.uname_result.processor.lock).__qualname__}' == '_thread.RLock'
    var_5 = [var_1]
    var_6 = var_3.validate(var_5)
    var_7 = []
    with pytest.raises(module_6.ValidationError):
        var_3.validate(var_7)

@pytest.mark.xfail(strict=True)
def test_case_107():
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
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Union(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 3
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_108():
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
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_2 = [var_0, var_1, var_1]
    var_3 = module_0.Union(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 3
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = True
    var_5 = var_3.validate(var_4)
    assert var_5 is True
    var_5.validate(var_1)

def test_case_109():
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
    var_1 = 5
    var_2 = module_0.Array(var_0, min_items=var_1, max_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.additional_items is False
    assert var_2.min_items == 5
    assert var_2.max_items == 5
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = [var_1, var_1, var_1]
    with pytest.raises(module_6.ValidationError):
        var_2.validate(var_3)

def test_case_110():
    var_0 = module_2.python_version_tuple()
    with pytest.raises(AssertionError):
        module_0.Array(var_0, min_items=var_0, max_items=var_0)

@pytest.mark.xfail(strict=True)
def test_case_111():
    var_0 = 6
    var_1 = 10
    var_2 = -8
    var_3 = False
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_1, precision=var_3, multiple_of=var_3, coerce_types=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Number'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum == 6
    assert var_4.maximum == 10
    assert var_4.exclusive_minimum == -8
    assert var_4.exclusive_maximum == 10
    assert var_4.multiple_of is False
    assert var_4.precision is False
    assert var_4.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4.validate(var_0)

def test_case_112():
    var_0 = 9
    var_1 = 14
    var_2 = -26
    var_3 = True
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_1, precision=var_3, multiple_of=var_3, coerce_types=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Number'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum == 9
    assert var_4.maximum == 14
    assert var_4.exclusive_minimum == -26
    assert var_4.exclusive_maximum == 14
    assert var_4.multiple_of is True
    assert var_4.precision is True
    assert var_4.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_5 = var_4.validate(var_0)
    assert var_5 == 9

@pytest.mark.xfail(strict=True)
def test_case_113():
    var_0 = 26
    var_1 = 10
    var_2 = 0
    var_3 = 2
    var_4 = True
    var_5 = module_0.Number(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_1, precision=var_3, multiple_of=var_3, coerce_types=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Number'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum == 26
    assert var_5.maximum == 10
    assert var_5.exclusive_minimum == 0
    assert var_5.exclusive_maximum == 10
    assert var_5.multiple_of == 2
    assert var_5.precision == 2
    assert var_5.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_5.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_114():
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
    var_2 = var_1.serialize(var_0)
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
    var_3 = -83.92
    var_2.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_115():
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
    var_1 = [var_0, var_0]
    var_2 = module_0.Array(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'builtins.list'
    assert len(var_2.items) == 2
    assert var_2.additional_items is False
    assert var_2.min_items == 2
    assert var_2.max_items == 2
    assert var_2.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1.__float__()

@pytest.mark.xfail(strict=True)
def test_case_116():
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
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'builtins.list'
    assert len(var_3.items) == 2
    assert var_3.additional_items is False
    assert var_3.min_items == 2
    assert var_3.max_items == 2
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = len(var_2)
    var_5 = module_0.Array(var_2, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.items).__module__}.{type(var_5.items).__qualname__}' == 'builtins.list'
    assert len(var_5.items) == 2
    assert f'{type(var_5.additional_items).__module__}.{type(var_5.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_5.min_items == 2
    assert var_5.max_items is None
    assert var_5.unique_items is False
    var_1.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_117():
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
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Union(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 3
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = -83.92024792475101
    var_4.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_118():
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
    var_2 = [var_0, var_1, var_1, var_0]
    var_3 = module_0.Union(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 4
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = module_2.python_implementation()
    assert var_4 == 'CPython'
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
    var_6 = module_0.String()
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = module_0.Boolean()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.coerce_types is True
    assert module_0.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_0.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_0.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_3.validate(var_4)

@pytest.mark.xfail(strict=True)
def test_case_119():
    var_0 = 9
    var_1 = 14
    var_2 = -26
    var_3 = True
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_1, precision=var_3, multiple_of=var_3, coerce_types=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Number'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum == 9
    assert var_4.maximum == 14
    assert var_4.exclusive_minimum == -26
    assert var_4.exclusive_maximum == 14
    assert var_4.multiple_of is True
    assert var_4.precision is True
    assert var_4.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_120():
    var_0 = 1
    var_1 = 0
    var_2 = '0.01'
    var_3 = True
    var_4 = module_0.Number(minimum=var_0, maximum=var_3, exclusive_minimum=var_1, exclusive_maximum=var_3, precision=var_2, multiple_of=var_1, coerce_types=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Number'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum == 1
    assert var_4.maximum is True
    assert var_4.exclusive_minimum == 0
    assert var_4.exclusive_maximum is True
    assert var_4.multiple_of == 0
    assert var_4.precision == '0.01'
    assert var_4.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_4.validate(var_0)
    assert var_5 == 2

def test_case_121():
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
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.Union(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 4
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = None
    var_4 = module_0.Object(property_names=var_3)
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
    var_5 = var_4.serialize(var_3)
    var_6 = var_0.validate_or_error(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_7 = var_0.__or__(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Union'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.any_of).__module__}.{type(var_7.any_of).__qualname__}' == 'builtins.list'
    assert len(var_7.any_of) == 5

@pytest.mark.xfail(strict=True)
def test_case_122():
    var_0 = 1
    var_1 = 10
    var_2 = 0
    var_3 = 2
    var_4 = True
    var_5 = module_0.Number(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_1, precision=var_3, multiple_of=var_3, coerce_types=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Number'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum == 1
    assert var_5.maximum == 10
    assert var_5.exclusive_minimum == 0
    assert var_5.exclusive_maximum == 10
    assert var_5.multiple_of == 2
    assert var_5.precision == 2
    assert var_5.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Number.numeric_type is None
    assert module_0.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    var_5.validate(var_0)

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
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'builtins.list'
    assert len(var_3.items) == 2
    assert var_3.additional_items is False
    assert var_3.min_items == 2
    assert var_3.max_items == 2
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = len(var_2)
    var_5 = module_0.String()
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
    var_6 = module_0.Decimal(maximum=var_4, precision=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.minimum is None
    assert var_6.maximum == 2
    assert var_6.exclusive_minimum is None
    assert var_6.exclusive_maximum is None
    assert var_6.multiple_of is None
    assert var_6.precision == 2
    assert var_6.coerce_types is True
    var_7 = len(var_2)
    var_8 = var_6.validate_or_error(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_8.value).__module__}.{type(var_8.value).__qualname__}' == 'decimal.Decimal'
    assert var_8.error is None

def test_case_124():
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
    var_1 = 1
    var_2 = -6
    var_3 = module_0.Array(var_0, min_items=var_1, max_items=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.additional_items is False
    assert var_3.min_items == 1
    assert var_3.max_items == -6
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = 1
    var_5 = [var_1, var_4, var_2]
    with pytest.raises(module_6.ValidationError):
        var_3.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_125():
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
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'builtins.list'
    assert len(var_3.items) == 2
    assert var_3.additional_items is False
    assert var_3.min_items == 2
    assert var_3.max_items == 2
    assert var_3.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = len(var_2)
    var_3.serialize(var_4)

@pytest.mark.xfail(strict=True)
def test_case_126():
    var_0 = '|.\t-ff_=~})'
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
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = module_2.python_build()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_1, required=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.properties).__module__}.{type(var_4.properties).__qualname__}' == 'builtins.dict'
    assert len(var_4.properties) == 1
    assert var_4.pattern_properties == {}
    assert f'{type(var_4.additional_properties).__module__}.{type(var_4.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == ['main', 'Jan 13 2026 03:14:08']
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = module_5.ChainMap(*var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.ChainMap'
    assert len(var_5) == 14
    assert f'{type(module_5.ChainMap.fromkeys).__module__}.{type(module_5.ChainMap.fromkeys).__qualname__}' == 'builtins.method'
    assert f'{type(module_5.ChainMap.parents).__module__}.{type(module_5.ChainMap.parents).__qualname__}' == 'builtins.property'
    var_4.validate(var_5)

def test_case_127():
    var_0 = 'agJ'
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
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3, required=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.properties).__module__}.{type(var_4.properties).__qualname__}' == 'builtins.dict'
    assert len(var_4.properties) == 1
    assert var_4.pattern_properties == {}
    assert var_4.additional_properties is False
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert f'{type(var_4.required).__module__}.{type(var_4.required).__qualname__}' == 'builtins.dict'
    assert len(var_4.required) == 1
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    with pytest.raises(module_6.ValidationError):
        var_4.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_128():
    var_0 = 'age'
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
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = [var_0]
    var_4 = True
    var_5 = module_0.Object(properties=var_2, additional_properties=var_4, required=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.properties).__module__}.{type(var_5.properties).__qualname__}' == 'builtins.dict'
    assert len(var_5.properties) == 1
    assert var_5.pattern_properties == {}
    assert var_5.additional_properties is True
    assert var_5.property_names is None
    assert var_5.min_properties is None
    assert var_5.max_properties is None
    assert var_5.required == ['age']
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_0: var_7}
    var_9 = var_5.validate(var_8)
    var_5.__or__(var_9)

def test_case_129():
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
    var_1 = module_0.Array(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.items).__module__}.{type(var_1.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items is None
    assert var_1.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = []
    var_6 = var_1.serialize(var_5)
    var_7 = var_1.serialize(var_6)
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_0.Array(var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Array'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.items == []
    assert var_10.additional_items is False
    assert var_10.min_items == 0
    assert var_10.max_items == 0
    assert var_10.unique_items is False
    var_11 = 'hello'
    var_12 = [var_2, var_11]
    var_13 = var_10.serialize(var_12)
    var_14 = 'world'
    var_15 = [var_3, var_14]
    var_16 = var_10.serialize(var_15)
    var_17 = var_10.serialize(var_7)
    var_18 = []
    var_19 = module_0.Array(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Array'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.items == []
    assert var_19.additional_items is False
    assert var_19.min_items == 0
    assert var_19.max_items == 0
    assert var_19.unique_items is False
    var_20 = [var_2, var_3, var_4]
    var_21 = var_19.serialize(var_20)
    var_22 = []
    var_23 = var_19.serialize(var_22)
    var_24 = var_19.serialize(var_21)

@pytest.mark.xfail(strict=True)
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
    var_1 = module_0.Array(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.items).__module__}.{type(var_1.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items is None
    assert var_1.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = []
    var_6 = var_1.serialize(var_5)
    var_7 = None
    var_8 = var_1.serialize(var_7)
    assert var_8 is None
    var_9 = module_0.Integer()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Integer'
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_11 = [var_9, var_10]
    var_12 = module_0.Array(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Array'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.items).__module__}.{type(var_12.items).__qualname__}' == 'builtins.list'
    assert len(var_12.items) == 2
    assert var_12.additional_items is False
    assert var_12.min_items == 2
    assert var_12.max_items == 2
    assert var_12.unique_items is False
    var_13 = var_12.serialize(var_6)
    var_14 = 'world'
    var_15 = [var_3, var_14]
    var_16 = var_12.serialize(var_15)
    var_17 = var_12.serialize(var_7)
    assert var_17 is None
    var_18 = []
    var_19 = module_0.Array(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Array'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.items == []
    assert var_19.additional_items is False
    assert var_19.min_items == 0
    assert var_19.max_items == 0
    assert var_19.unique_items is False
    var_20 = [var_2, var_3, var_4]
    var_21 = var_19.serialize(var_20)
    var_21.serialize(var_13)

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
    var_1 = module_0.Array(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.items).__module__}.{type(var_1.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items is None
    assert var_1.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)
    var_7 = [var_3, var_5]
    var_8 = var_1.serialize(var_7)
    var_9 = None
    var_10 = var_1.serialize(var_9)
    assert var_10 is None
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
    var_12 = module_0.String()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_13 = [var_11, var_12]
    var_14 = module_0.Array(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Array'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.items).__module__}.{type(var_14.items).__qualname__}' == 'builtins.list'
    assert len(var_14.items) == 2
    assert var_14.additional_items is False
    assert var_14.min_items == 2
    assert var_14.max_items == 2
    assert var_14.unique_items is False
    var_15 = 'hello'
    var_16 = [var_2, var_15]
    var_17 = var_14.serialize(var_16)
    var_18 = 'doQlq'
    var_19 = [var_3, var_18]
    var_20 = var_14.serialize(var_19)
    var_21 = var_14.serialize(var_9)
    assert var_21 is None
    var_22 = []
    var_23 = module_0.Array(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Array'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.items == []
    assert var_23.additional_items is False
    assert var_23.min_items == 0
    assert var_23.max_items == 0
    assert var_23.unique_items is False
    var_24 = [var_2, var_3, var_4]
    var_25 = var_23.serialize(var_24)
    var_26 = []
    var_27 = var_23.serialize(var_26)
    var_28 = var_23.serialize(var_9)
    assert var_28 is None

def test_case_132():
    var_0 = '.\t-fj_=~})'
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
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = [var_0, var_0]
    var_4 = module_0.Object(properties=var_2, additional_properties=var_1, required=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.properties).__module__}.{type(var_4.properties).__qualname__}' == 'builtins.dict'
    assert len(var_4.properties) == 1
    assert var_4.pattern_properties == {}
    assert f'{type(var_4.additional_properties).__module__}.{type(var_4.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == ['.\t-fj_=~})', '.\t-fj_=~})']
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = {var_1: var_3, var_1: var_3}
    with pytest.raises(module_6.ValidationError):
        var_4.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_133():
    var_0 = 'agXe'
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
    var_2 = module_2.python_branch()
    assert var_2 == ''
    var_3 = [var_0]
    var_4 = True
    var_5 = module_0.Object(properties=var_2, additional_properties=var_4, required=var_3)
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
    assert var_5.required == ['agXe']
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = {var_0: var_0, var_0: var_4}
    var_7 = var_5.validate(var_6)
    var_8 = 'name'
    var_5.validate(var_8)

def test_case_134():
    var_0 = 'age'
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
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = [var_0, var_0]
    var_4 = False
    var_5 = module_0.Object(properties=var_2, additional_properties=var_4, required=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.properties).__module__}.{type(var_5.properties).__qualname__}' == 'builtins.dict'
    assert len(var_5.properties) == 1
    assert var_5.pattern_properties == {}
    assert var_5.additional_properties is False
    assert var_5.property_names is None
    assert var_5.min_properties is None
    assert var_5.max_properties is None
    assert var_5.required == ['age', 'age']
    assert module_0.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = 'name'
    var_7 = {var_6: var_3, var_1: var_3}
    with pytest.raises(module_6.ValidationError):
        var_5.validate(var_7)

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
    var_1 = module_0.Array(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.items).__module__}.{type(var_1.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items is None
    assert var_1.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = 1
    var_3 = 3
    var_4 = []
    var_5 = var_1.serialize(var_4)
    var_6 = None
    var_7 = var_1.serialize(var_6)
    assert var_7 is None
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
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_0.Array(var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Array'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.items == []
    assert var_10.additional_items is False
    assert var_10.min_items == 0
    assert var_10.max_items == 0
    assert var_10.unique_items is False
    var_11 = 'hello'
    var_12 = [var_2, var_11]
    var_13 = var_10.serialize(var_12)
    var_14 = True
    with pytest.raises(AssertionError):
        module_0.String(allow_blank=var_14, max_length=var_3, pattern=var_7, format=var_1)

def test_case_136():
    var_0 = 'age'
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
    var_2 = None
    var_3 = var_1.validate_or_error(var_2)
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
    var_5 = {var_1: var_1, var_0: var_1, var_1: var_4, var_0: var_4}
    var_6 = [var_0]
    var_7 = False
    with pytest.raises(AssertionError):
        module_0.Object(properties=var_5, additional_properties=var_7, required=var_6)

@pytest.mark.xfail(strict=True)
def test_case_137():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = ''
    var_5 = (var_3, var_4)
    var_6 = 'C'
    var_7 = (var_6, var_6)
    var_8 = [var_2, var_5, var_7]
    var_9 = module_0.Choice(choices=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Choice'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.choices == [('a', 'A'), ('b', ''), ('C', 'C')]
    assert var_9.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_10 = var_9.validate(var_0)
    assert var_10 == 'a'
    var_11 = var_9.validate(var_3)
    assert var_11 == 'b'
    var_12 = var_9.validate(var_0)
    assert var_12 == 'a'
    var_13 = module_5.ChainMap(*var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.ChainMap'
    assert len(var_13) == 1
    assert f'{type(module_5.ChainMap.fromkeys).__module__}.{type(module_5.ChainMap.fromkeys).__qualname__}' == 'builtins.method'
    assert f'{type(module_5.ChainMap.parents).__module__}.{type(module_5.ChainMap.parents).__qualname__}' == 'builtins.property'
    var_12.validate(var_12)

def test_case_138():
    var_0 = module_2.uname()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'platform.uname_result'
    assert len(var_0) == 6
    assert f'{type(module_2.uname_result.processor).__module__}.{type(module_2.uname_result.processor).__qualname__}' == 'functools.cached_property'
    assert module_2.uname_result.processor.attrname == 'processor'
    assert f'{type(module_2.uname_result.processor.lock).__module__}.{type(module_2.uname_result.processor.lock).__qualname__}' == '_thread.RLock'
    var_1 = 'b'
    var_2 = 'B'
    var_3 = (var_1, var_2)
    var_4 = 'c'
    var_5 = ''
    var_6 = (var_4, var_5)
    var_7 = [var_0, var_3, var_6]
    with pytest.raises(AssertionError):
        module_0.Choice(choices=var_7)

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
    var_2 = [var_0, var_1]
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
    var_4 = 1
    var_5 = 3
    var_6 = module_0.Array(var_2, var_3, var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.items).__module__}.{type(var_6.items).__qualname__}' == 'builtins.list'
    assert len(var_6.items) == 2
    assert f'{type(var_6.additional_items).__module__}.{type(var_6.additional_items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_6.min_items == 1
    assert var_6.max_items == 3
    assert var_6.unique_items is False
    assert module_0.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_7 = 'test'
    var_8 = [var_4, var_7, var_5]
    var_9 = var_6.validate(var_8)
    var_10 = [var_4, var_7]
    var_11 = var_6.validate(var_10)
    var_12 = [var_4]
    var_13 = var_6.validate(var_12)
    var_14 = []
    with pytest.raises(module_6.ValidationError):
        var_6.validate(var_14)