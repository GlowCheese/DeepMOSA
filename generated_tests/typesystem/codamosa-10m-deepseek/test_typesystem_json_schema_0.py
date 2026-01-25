# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import typesystem.composites as module_1
import typesystem.fields as module_2
import re as module_3
import typesystem.schemas as module_4

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = None
    module_0.from_json_schema(var_2)

def test_case_1():
    var_0 = {}
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_2():
    var_0 = {}
    var_1 = None
    var_2 = module_0.type_from_json_schema(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}

def test_case_3():
    var_0 = module_2.Boolean()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = {}
    var_3 = None
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_2, var_3, var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_4():
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
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.ref_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.all_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.any_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

def test_case_9():
    var_0 = None
    var_1 = False
    var_2 = module_0.from_json_schema(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.get_standard_properties(var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    module_0.not_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

def test_case_13():
    var_0 = module_1.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
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
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_2.Float()
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
    var_6 = module_0.to_json_schema(var_5)
    var_7 = module_2.Boolean()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_8 = module_0.to_json_schema(var_7)
    var_9 = module_2.Array()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Array'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.items is None
    assert var_9.additional_items is False
    assert var_9.min_items is None
    assert var_9.max_items is None
    assert var_9.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_10 = module_2.Object()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Object'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.properties == {}
    assert var_10.pattern_properties == {}
    assert var_10.additional_properties is True
    assert var_10.property_names is None
    assert var_10.min_properties is None
    assert var_10.max_properties is None
    assert var_10.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_11 = module_0.to_json_schema(var_10)

def test_case_14():
    var_0 = module_2.Boolean()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = module_0.get_standard_properties(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_15():
    var_0 = 'oneOf'
    var_1 = 'integer'
    var_2 = {var_1: var_1}
    var_3 = [var_2, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.one_of_from_json_schema(var_4, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.one_of).__module__}.{type(var_5.one_of).__qualname__}' == 'builtins.list'
    assert len(var_5.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_6 = len(var_0)
    with pytest.raises(IndexError):
        var_7 = var_5.one_of[var_6]

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = False
    module_0.to_json_schema(var_0, var_0)

def test_case_17():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = '#/components/schemas/MySchema'
    var_1 = 1
    var_2 = module_2.String(min_length=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length == 1
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = module_0.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = None
    module_0.ref_from_json_schema(var_5, var_5)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'oneOf'
    var_2 = 'type'
    var_3 = {var_2: var_1}
    var_4 = 'integer'
    var_5 = {var_2: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_1: var_6}
    module_0.one_of_from_json_schema(var_7, var_0)

def test_case_20():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'oneOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = module_0.one_of_from_json_schema(var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.one_of).__module__}.{type(var_9.one_of).__qualname__}' == 'builtins.list'
    assert len(var_9.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_10 = len(var_2)
    var_11 = 0
    var_12 = var_9.one_of[var_11]
    with pytest.raises(TypeError):
        var_13 = var_9.one_of[var_3]

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'oneOf'
    var_2 = 'type'
    var_3 = 'w'
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = {var_1: var_5}
    module_0.one_of_from_json_schema(var_6, var_0)

def test_case_22():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'format'
    var_4 = 'pattern'
    var_5 = 'default'
    var_6 = 'string'
    var_7 = 5
    var_8 = 10
    var_9 = 'email'
    var_10 = '^[A-Za-z0-9]+$'
    var_11 = 'example@example.com'
    var_12 = {var_4: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = False
    var_14 = module_0.from_json_schema_type(var_12, var_6, var_13, var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.String'
    assert var_14.default == 'example@example.com'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.allow_blank is False
    assert var_14.trim_whitespace is True
    assert var_14.max_length == 10
    assert var_14.min_length == 5
    assert var_14.format == 'email'
    assert var_14.coerce_types is False
    assert var_14.pattern == '^[A-Za-z0-9]+$'
    assert f'{type(var_14.pattern_regex).__module__}.{type(var_14.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_15 = 'minimum'
    var_16 = 'maximum'
    var_17 = 'exclusiveMaximum'
    var_18 = 'multipleOf'
    var_19 = 'integer'
    var_20 = 100
    var_21 = 2
    var_22 = {var_19: var_19, var_15: var_13, var_16: var_20, var_4: var_13, var_17: var_20, var_18: var_21, var_5: var_21}
    var_23 = module_0.from_json_schema_type(var_22, var_19, var_13, var_0)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Integer'
    assert var_23.default == 2
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.minimum is False
    assert var_23.maximum == 100
    assert var_23.exclusive_minimum is None
    assert var_23.exclusive_maximum == 100
    assert var_23.multiple_of == 2
    assert var_23.precision is None
    assert var_23.coerce_types is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_24 = 'number'
    var_25 = module_0.from_json_schema_type(var_22, var_24, var_13, var_0)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Float'
    assert var_25.default == 2
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.minimum is False
    assert var_25.maximum == 100
    assert var_25.exclusive_minimum is None
    assert var_25.exclusive_maximum == 100
    assert var_25.multiple_of == 2
    assert var_25.precision is None
    assert var_25.coerce_types is False
    var_26 = 'boolean'
    var_27 = True
    var_28 = {var_24: var_26, var_5: var_27}
    var_29 = module_0.from_json_schema_type(var_28, var_26, var_13, var_0)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_29.default is True
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert var_29.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_30 = 'items'
    var_31 = 'maxItems'
    var_32 = 'uniqueItems'
    var_33 = 'array'
    var_34 = True
    var_35 = '+example'
    var_36 = [var_35]
    var_37 = {var_2: var_33, var_30: var_22, var_2: var_27, var_31: var_8, var_32: var_34, var_5: var_36}
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_37, var_33, var_13, var_0)

def test_case_23():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'type'
    var_2 = 'maxLength'
    var_3 = 'format'
    var_4 = 'pattern'
    var_5 = 'default'
    var_6 = 'string'
    var_7 = 5
    var_8 = 10
    var_9 = 'email'
    var_10 = 'example@example.com'
    var_11 = {var_1: var_6, var_9: var_7, var_2: var_8, var_3: var_9, var_4: var_6, var_5: var_10}
    var_12 = False
    var_13 = module_0.from_json_schema_type(var_11, var_6, var_12, var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.String'
    assert var_13.default == 'example@example.com'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.allow_blank is True
    assert var_13.trim_whitespace is True
    assert var_13.max_length == 10
    assert var_13.min_length is None
    assert var_13.format == 'email'
    assert var_13.coerce_types is False
    assert var_13.pattern == 'string'
    assert f'{type(var_13.pattern_regex).__module__}.{type(var_13.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_14 = 'minimum'
    var_15 = 'maximum'
    var_16 = 'exclusiveMinimum'
    var_17 = 'exclusiveMaximum'
    var_18 = 'multipleOf'
    var_19 = 'integer'
    var_20 = 1
    var_21 = 100
    var_22 = 101
    var_23 = 2
    var_24 = {var_1: var_19, var_14: var_20, var_15: var_21, var_16: var_12, var_17: var_22, var_18: var_23, var_5: var_23}
    var_25 = module_0.from_json_schema_type(var_24, var_19, var_12, var_0)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Integer'
    assert var_25.default == 2
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.minimum == 1
    assert var_25.maximum == 100
    assert var_25.exclusive_minimum is False
    assert var_25.exclusive_maximum == 101
    assert var_25.multiple_of == 2
    assert var_25.precision is None
    assert var_25.coerce_types is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_26 = 'number'
    var_27 = {var_1: var_26, var_14: var_20, var_15: var_21, var_16: var_12, var_17: var_22, var_18: var_23, var_5: var_23}
    var_28 = module_0.from_json_schema_type(var_27, var_26, var_12, var_0)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Float'
    assert var_28.default == 2
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.minimum == 1
    assert var_28.maximum == 100
    assert var_28.exclusive_minimum is False
    assert var_28.exclusive_maximum == 101
    assert var_28.multiple_of == 2
    assert var_28.precision is None
    assert var_28.coerce_types is False
    var_29 = 'boolean'
    var_30 = True
    var_31 = {var_1: var_29, var_5: var_30}
    var_32 = module_0.from_json_schema_type(var_31, var_29, var_12, var_0)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_32.default is True
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_33 = True
    var_34 = 'example'
    var_35 = 'properties'
    var_36 = 'minProperties'
    var_37 = 'maxProperties'
    var_38 = 'required'
    var_39 = 'object'
    var_40 = 'name'
    var_41 = {var_1: var_6}
    var_42 = {var_40: var_41}
    var_43 = [var_40]
    var_44 = {var_40: var_34}
    var_45 = {var_1: var_39, var_35: var_42, var_36: var_33, var_37: var_23, var_38: var_43, var_5: var_44}
    var_46 = module_0.from_json_schema_type(var_45, var_39, var_12, var_0)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Object'
    assert var_46.default == {'name': 'example'}
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.properties).__module__}.{type(var_46.properties).__qualname__}' == 'builtins.dict'
    assert len(var_46.properties) == 1
    assert var_46.pattern_properties == {}
    assert var_46.additional_properties is None
    assert var_46.property_names is None
    assert var_46.min_properties is True
    assert var_46.max_properties == 2
    assert var_46.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_47 = var_46.properties[var_40]

def test_case_24():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'allOf'
    var_2 = 'type'
    var_3 = 'minLength'
    var_4 = 'string'
    var_5 = 5
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'maxLength'
    var_8 = 10
    var_9 = {var_2: var_4, var_7: var_8}
    var_10 = [var_6, var_9]
    var_11 = {var_1: var_10}
    var_12 = module_0.all_of_from_json_schema(var_11, var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.all_of).__module__}.{type(var_12.all_of).__qualname__}' == 'builtins.list'
    assert len(var_12.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_13 = var_12.all_of
    var_14 = len(var_13)
    assert var_14 == 2
    with pytest.raises(IndexError):
        var_15 = var_12.all_of[var_8]

def test_case_25():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'number'
    var_8 = {var_4: var_7}
    var_9 = 'boolean'
    var_10 = {var_4: var_9}
    var_11 = {var_1: var_6, var_2: var_8, var_3: var_10}
    var_12 = module_0.if_then_else_from_json_schema(var_11, var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.if_clause).__module__}.{type(var_12.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_12.then_clause).__module__}.{type(var_12.then_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_12.else_clause).__module__}.{type(var_12.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_13 = var_12.if_clause
    var_14 = var_12.then_clause
    var_15 = var_12.else_clause
    var_16 = {var_4: var_5}
    var_17 = {var_4: var_7}
    var_18 = {var_1: var_16, var_2: var_17}
    var_19 = module_0.if_then_else_from_json_schema(var_18, var_0)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.if_clause).__module__}.{type(var_19.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_19.then_clause).__module__}.{type(var_19.then_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_19.else_clause).__module__}.{type(var_19.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_20 = var_19.if_clause
    var_21 = var_12.then_clause
    var_22 = {var_4: var_9}
    var_23 = {var_1: var_16, var_3: var_22}
    var_24 = module_0.if_then_else_from_json_schema(var_23, var_0)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.if_clause).__module__}.{type(var_24.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_24.then_clause).__module__}.{type(var_24.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_24.else_clause).__module__}.{type(var_24.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_25 = var_24.if_clause
    var_26 = var_24.else_clause

def test_case_26():
    var_0 = module_2.Boolean()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.from_json_schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.type_from_json_schema(var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = module_0.type_from_json_schema(var_1, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is True
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_5 = module_2.Boolean()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_6 = module_0.to_json_schema(var_3, var_0)
    module_3.search(var_0, var_0)

def test_case_28():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_4 = 'type'
    var_5 = 'minLength'
    var_6 = 'string'
    var_7 = 5
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.from_json_schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length is None
    assert var_9.min_length == 5
    assert var_9.format is None
    assert var_9.coerce_types is False
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = '$ref'
    var_11 = '#/definitions/Example'
    var_12 = {var_10: var_11}
    var_13 = module_0.from_json_schema(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.to == '#/definitions/Example'
    assert f'{type(var_13.definitions).__module__}.{type(var_13.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13.definitions) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_14 = 'enum'
    var_15 = 'value1'
    var_16 = 'value2'
    var_17 = [var_15, var_16]
    var_18 = {var_14: var_17}
    var_19 = module_0.from_json_schema(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Choice'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.choices == [('value1', 'value1'), ('value2', 'value2')]
    assert var_19.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_20 = 'const'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = module_0.from_json_schema(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Const'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.const == 'value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_24 = 'allOf'
    var_25 = {var_4: var_6}
    var_26 = {var_5: var_7}
    var_27 = [var_25, var_26]
    var_28 = {var_24: var_27}
    var_29 = module_0.from_json_schema(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.all_of).__module__}.{type(var_29.all_of).__qualname__}' == 'builtins.list'
    assert len(var_29.all_of) == 2
    var_30 = 'anyOf'
    var_31 = {var_4: var_6}
    var_32 = 'number'
    var_33 = {var_4: var_32}
    var_34 = [var_31, var_33]
    var_35 = {var_30: var_34}
    var_36 = module_0.from_json_schema(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Union'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert f'{type(var_36.any_of).__module__}.{type(var_36.any_of).__qualname__}' == 'builtins.list'
    assert len(var_36.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_37 = 'oneOf'
    var_38 = {var_4: var_6}
    var_39 = {var_4: var_32}
    var_40 = [var_38, var_39]
    var_41 = {var_37: var_40}
    var_42 = module_0.from_json_schema(var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert f'{type(var_42.one_of).__module__}.{type(var_42.one_of).__qualname__}' == 'builtins.list'
    assert len(var_42.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_43 = 'not'
    var_44 = {var_4: var_6}
    var_45 = {var_43: var_44}
    var_46 = module_0.from_json_schema(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.composites.Not'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.negated).__module__}.{type(var_46.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_47 = 'if'
    var_48 = 'then'
    var_49 = 'else'
    var_50 = {var_4: var_6}
    var_51 = {var_5: var_7}
    var_52 = {var_4: var_32}
    var_53 = {var_47: var_50, var_48: var_51, var_49: var_52}
    var_54 = module_0.from_json_schema(var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert f'{type(var_54.if_clause).__module__}.{type(var_54.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_54.then_clause).__module__}.{type(var_54.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_54.else_clause).__module__}.{type(var_54.else_clause).__qualname__}' == 'typesystem.fields.Float'
    var_55 = 'maxLength'
    var_56 = 10
    var_57 = {var_4: var_6, var_5: var_7, var_55: var_56}
    var_58 = module_0.from_json_schema(var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.String'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.allow_blank is False
    assert var_58.trim_whitespace is True
    assert var_58.max_length == 10
    assert var_58.min_length == 5
    assert var_58.format is None
    assert var_58.coerce_types is False
    assert var_58.pattern is None
    assert var_58.pattern_regex is None

def test_case_29():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = module_0.from_json_schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.default == ''
    assert var_7.allow_blank is True
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format is None
    assert var_7.coerce_types is False
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = 'integer'
    var_9 = module_0.from_json_schema(var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.default == ''
    assert var_9.allow_blank is True
    assert var_9.trim_whitespace is True
    assert var_9.max_length is None
    assert var_9.min_length is None
    assert var_9.format is None
    assert var_9.coerce_types is False
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_10 = 'number'
    var_11 = {var_4: var_10}
    var_12 = module_0.from_json_schema(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Float'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum is None
    assert var_12.maximum is None
    assert var_12.exclusive_minimum is None
    assert var_12.exclusive_maximum is None
    assert var_12.multiple_of is None
    assert var_12.precision is None
    assert var_12.coerce_types is False
    var_13 = 'boolean'
    var_14 = {var_4: var_13}
    var_15 = module_0.from_json_schema(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_16 = 'items'
    var_17 = 'array'
    var_18 = {var_4: var_5}
    var_19 = {var_4: var_17, var_16: var_18}
    var_20 = module_0.from_json_schema(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Array'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.items).__module__}.{type(var_20.items).__qualname__}' == 'typesystem.fields.String'
    assert var_20.additional_items is True
    assert var_20.min_items == 0
    assert var_20.max_items is None
    assert var_20.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_21 = 'properties'
    var_22 = 'object'
    var_23 = 'name'
    var_24 = {var_4: var_5}
    var_25 = {var_23: var_24}
    var_26 = {var_4: var_22, var_21: var_25}
    var_27 = module_0.from_json_schema(var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Object'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.properties).__module__}.{type(var_27.properties).__qualname__}' == 'builtins.dict'
    assert len(var_27.properties) == 1
    assert var_27.pattern_properties == {}
    assert var_27.additional_properties is None
    assert var_27.property_names is None
    assert var_27.min_properties is None
    assert var_27.max_properties is None
    assert var_27.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_28 = 'allOT'
    var_29 = {var_4: var_5}
    var_30 = 'minLength'
    var_31 = 5
    var_32 = {var_30: var_31}
    var_33 = [var_29, var_32]
    var_34 = {var_28: var_33}
    var_35 = module_0.from_json_schema(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Any'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    var_36 = 'anyOf'
    var_37 = {var_4: var_5}
    var_38 = {var_4: var_10}
    var_39 = [var_37, var_38]
    var_40 = {var_36: var_39}
    var_41 = module_0.from_json_schema(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Union'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.any_of).__module__}.{type(var_41.any_of).__qualname__}' == 'builtins.list'
    assert len(var_41.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_42 = module_0.from_json_schema(var_34)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.Any'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    var_43 = 'not'
    var_44 = {var_4: var_5}
    var_45 = {var_43: var_44}
    var_46 = module_0.from_json_schema(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.composites.Not'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.negated).__module__}.{type(var_46.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_47 = 'if'
    var_48 = 'then'
    var_49 = {var_30: var_31}
    var_50 = {var_47: var_6, var_48: var_49}
    var_51 = module_0.from_json_schema(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.if_clause).__module__}.{type(var_51.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_51.then_clause).__module__}.{type(var_51.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_51.else_clause).__module__}.{type(var_51.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_52 = module_4.Definitions()
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_52) == 0
    var_53 = '$ref'
    var_54 = '#/components/schemas/Example'
    var_55 = {var_53: var_54}
    var_56 = module_0.from_json_schema(var_55, var_52)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert var_56.to == '#/components/schemas/Example'
    assert f'{type(var_56.definitions).__module__}.{type(var_56.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_56.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_57 = {var_53: var_54}
    var_58 = {var_23: var_57}
    var_59 = {var_4: var_22, var_21: var_58}
    var_60 = module_0.from_json_schema(var_59, var_52)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.Object'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert f'{type(var_60.properties).__module__}.{type(var_60.properties).__qualname__}' == 'builtins.dict'
    assert len(var_60.properties) == 1
    assert var_60.pattern_properties == {}
    assert var_60.additional_properties is None
    assert var_60.property_names is None
    assert var_60.min_properties is None
    assert var_60.max_properties is None
    assert var_60.required == []
    var_61 = 'required'
    var_62 = 'age'
    var_63 = {var_4: var_5}
    var_64 = 'minimum'
    var_65 = {var_4: var_8, var_64: var_2}
    var_66 = {var_23: var_63, var_62: var_65}
    var_67 = [var_23]
    var_68 = {var_4: var_22, var_21: var_66, var_61: var_67}
    var_69 = module_0.from_json_schema(var_68)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.fields.Object'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert f'{type(var_69.properties).__module__}.{type(var_69.properties).__qualname__}' == 'builtins.dict'
    assert len(var_69.properties) == 2
    assert var_69.pattern_properties == {}
    assert var_69.additional_properties is None
    assert var_69.property_names is None
    assert var_69.min_properties is None
    assert var_69.max_properties is None
    assert var_69.required == ['name']
    var_70 = 'additionalProperties'
    var_71 = {var_4: var_5}
    var_72 = {var_4: var_22, var_70: var_71}
    var_73 = module_0.from_json_schema(var_72)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.Object'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert var_73.properties == {}
    assert var_73.pattern_properties == {}
    assert f'{type(var_73.additional_properties).__module__}.{type(var_73.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_73.property_names is None
    assert var_73.min_properties is None
    assert var_73.max_properties is None
    assert var_73.required == []
    var_74 = '^[a-z]+$'
    var_75 = {var_4: var_5}
    var_76 = {var_74: var_75}
    var_77 = {var_4: var_22, var_48: var_76}
    var_78 = module_0.from_json_schema(var_77)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.Object'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert var_78.properties == {}
    assert var_78.pattern_properties == {}
    assert var_78.additional_properties is None
    assert var_78.property_names is None
    assert var_78.min_properties is None
    assert var_78.max_properties is None
    assert var_78.required == []
    var_79 = 'minItems'
    var_80 = {var_4: var_5}
    var_81 = {var_4: var_17, var_16: var_80, var_79: var_0}
    var_82 = module_0.from_json_schema(var_81)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.Array'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert f'{type(var_82.items).__module__}.{type(var_82.items).__qualname__}' == 'typesystem.fields.String'
    assert var_82.additional_items is True
    assert var_82.min_items is True
    assert var_82.max_items is None
    assert var_82.unique_items is False
    var_83 = 'uniqueItems'
    var_84 = {var_4: var_5}
    var_85 = {var_4: var_17, var_16: var_84, var_83: var_0}
    var_86 = module_0.from_json_schema(var_85)
    assert f'{type(var_86).__module__}.{type(var_86).__qualname__}' == 'typesystem.fields.Array'
    assert var_86.title == ''
    assert var_86.description == ''
    assert var_86.allow_null is False
    assert var_86.read_only is False
    assert f'{type(var_86.items).__module__}.{type(var_86.items).__qualname__}' == 'typesystem.fields.String'
    assert var_86.additional_items is True
    assert var_86.min_items == 0
    assert var_86.max_items is None
    assert var_86.unique_items is True
    var_87 = 'maxLength'
    var_88 = 10
    var_89 = {var_30: var_31, var_87: var_88}
    var_90 = module_0.from_json_schema(var_89)
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'typesystem.fields.Union'
    assert var_90.default is None
    assert var_90.title == ''
    assert var_90.description == ''
    assert var_90.allow_null is True
    assert var_90.read_only is False
    assert f'{type(var_90.any_of).__module__}.{type(var_90.any_of).__qualname__}' == 'builtins.list'
    assert len(var_90.any_of) == 5
    var_91 = [var_5, var_10]
    var_92 = {var_4: var_91}
    var_93 = module_0.from_json_schema(var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.Union'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert f'{type(var_93.any_of).__module__}.{type(var_93.any_of).__qualname__}' == 'builtins.list'
    assert len(var_93.any_of) == 2
    var_94 = {var_4: var_5}
    var_95 = module_0.from_json_schema(var_94)
    assert f'{type(var_95).__module__}.{type(var_95).__qualname__}' == 'typesystem.fields.String'
    assert var_95.title == ''
    assert var_95.description == ''
    assert var_95.allow_null is False
    assert var_95.read_only is False
    assert var_95.default == ''
    assert var_95.allow_blank is True
    assert var_95.trim_whitespace is True
    assert var_95.max_length is None
    assert var_95.min_length is None
    assert var_95.format is None
    assert var_95.coerce_types is False
    assert var_95.pattern is None
    assert var_95.pattern_regex is None
    var_96 = module_4.Definitions()
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_96) == 0
    var_97 = {var_53: var_54}
    var_98 = module_0.from_json_schema(var_97, var_96)
    assert f'{type(var_98).__module__}.{type(var_98).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_98.title == ''
    assert var_98.description == ''
    assert var_98.allow_null is False
    assert var_98.read_only is False
    assert var_98.to == '#/components/schemas/Example'
    assert f'{type(var_98.definitions).__module__}.{type(var_98.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_98.definitions) == 0
    var_99 = {}
    var_100 = module_0.from_json_schema(var_99)
    assert f'{type(var_100).__module__}.{type(var_100).__qualname__}' == 'typesystem.fields.Any'
    assert var_100.title == ''
    assert var_100.description == ''
    assert var_100.allow_null is False
    assert var_100.read_only is False

def test_case_30():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.from_json_schema(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.String'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.default == ''
    assert var_6.allow_blank is True
    assert var_6.trim_whitespace is True
    assert var_6.max_length is None
    assert var_6.min_length is None
    assert var_6.format is None
    assert var_6.coerce_types is False
    assert var_6.pattern is None
    assert var_6.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = 'integer'
    var_8 = module_0.from_json_schema(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.default == ''
    assert var_8.allow_blank is True
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format is None
    assert var_8.coerce_types is False
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = module_0.from_json_schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Float'
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
    assert var_11.coerce_types is False
    var_12 = 'boolean'
    var_13 = {var_3: var_12}
    var_14 = module_0.from_json_schema(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_15 = 'items'
    var_16 = 'array'
    var_17 = {var_3: var_4}
    var_18 = {var_3: var_16, var_15: var_17}
    var_19 = module_0.from_json_schema(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Array'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.items).__module__}.{type(var_19.items).__qualname__}' == 'typesystem.fields.String'
    assert var_19.additional_items is True
    assert var_19.min_items == 0
    assert var_19.max_items is None
    assert var_19.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_20 = 'properties'
    var_21 = 'object'
    var_22 = 'name'
    var_23 = {var_3: var_4}
    var_24 = {var_22: var_23}
    var_25 = {var_3: var_21, var_20: var_24}
    var_26 = module_0.from_json_schema(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Object'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.properties).__module__}.{type(var_26.properties).__qualname__}' == 'builtins.dict'
    assert len(var_26.properties) == 1
    assert var_26.pattern_properties == {}
    assert var_26.additional_properties is None
    assert var_26.property_names is None
    assert var_26.min_properties is None
    assert var_26.max_properties is None
    assert var_26.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_27 = 'allOf'
    var_28 = {var_3: var_4}
    var_29 = 'minLength'
    var_30 = 5
    var_31 = {var_29: var_30}
    var_32 = [var_28, var_31]
    var_33 = {var_27: var_32}
    var_34 = module_0.from_json_schema(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.all_of).__module__}.{type(var_34.all_of).__qualname__}' == 'builtins.list'
    assert len(var_34.all_of) == 2
    var_35 = 'anyOf'
    var_36 = {var_3: var_4}
    var_37 = {var_3: var_9}
    var_38 = [var_36, var_37]
    var_39 = {var_35: var_38}
    var_40 = module_0.from_json_schema(var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Union'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert f'{type(var_40.any_of).__module__}.{type(var_40.any_of).__qualname__}' == 'builtins.list'
    assert len(var_40.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_41 = 'oneOf'
    var_42 = {var_3: var_4}
    var_43 = {var_3: var_9}
    var_44 = [var_42, var_43]
    var_45 = {var_41: var_44}
    var_46 = module_0.from_json_schema(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.one_of).__module__}.{type(var_46.one_of).__qualname__}' == 'builtins.list'
    assert len(var_46.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_47 = 'not'
    var_48 = {var_3: var_4}
    var_49 = {var_47: var_48}
    var_50 = module_0.from_json_schema(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.composites.Not'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert f'{type(var_50.negated).__module__}.{type(var_50.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_51 = 'if'
    var_52 = 'then'
    var_53 = {var_29: var_30}
    var_54 = {var_51: var_5, var_52: var_53}
    var_55 = module_0.from_json_schema(var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert f'{type(var_55.if_clause).__module__}.{type(var_55.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_55.then_clause).__module__}.{type(var_55.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_55.else_clause).__module__}.{type(var_55.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_56 = module_4.Definitions()
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_56) == 0
    var_57 = '$ref'
    var_58 = '#/components/schemas/Example'
    var_59 = {var_57: var_58}
    var_60 = module_0.from_json_schema(var_59, var_56)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.to == '#/components/schemas/Example'
    assert f'{type(var_60.definitions).__module__}.{type(var_60.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_60.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_61 = {var_57: var_58}
    var_62 = {var_22: var_61}
    var_63 = {var_3: var_21, var_20: var_62}
    var_64 = module_0.from_json_schema(var_63, var_56)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.Object'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert f'{type(var_64.properties).__module__}.{type(var_64.properties).__qualname__}' == 'builtins.dict'
    assert len(var_64.properties) == 1
    assert var_64.pattern_properties == {}
    assert var_64.additional_properties is None
    assert var_64.property_names is None
    assert var_64.min_properties is None
    assert var_64.max_properties is None
    assert var_64.required == []
    var_65 = 'required'
    var_66 = 'age'
    var_67 = {var_3: var_4}
    var_68 = 'minimum'
    var_69 = {var_3: var_7, var_68: var_0}
    var_70 = {var_22: var_67, var_66: var_69}
    var_71 = [var_22]
    var_72 = {var_3: var_21, var_20: var_70, var_65: var_71}
    var_73 = module_0.from_json_schema(var_72)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.Object'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert f'{type(var_73.properties).__module__}.{type(var_73.properties).__qualname__}' == 'builtins.dict'
    assert len(var_73.properties) == 2
    assert var_73.pattern_properties == {}
    assert var_73.additional_properties is None
    assert var_73.property_names is None
    assert var_73.min_properties is None
    assert var_73.max_properties is None
    assert var_73.required == ['name']
    var_74 = 'additionalProperties'
    var_75 = {var_3: var_4}
    var_76 = {var_3: var_21, var_74: var_75}
    var_77 = module_0.from_json_schema(var_76)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.Object'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert var_77.properties == {}
    assert var_77.pattern_properties == {}
    assert f'{type(var_77.additional_properties).__module__}.{type(var_77.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_77.property_names is None
    assert var_77.min_properties is None
    assert var_77.max_properties is None
    assert var_77.required == []
    var_78 = 'patternProperties'
    var_79 = '^[a-z]+$'
    var_80 = {var_3: var_4}
    var_81 = {var_79: var_80}
    var_82 = {var_3: var_21, var_78: var_81}
    var_83 = module_0.from_json_schema(var_82)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.fields.Object'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert var_83.properties == {}
    assert f'{type(var_83.pattern_properties).__module__}.{type(var_83.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_83.pattern_properties) == 1
    assert var_83.additional_properties is None
    assert var_83.property_names is None
    assert var_83.min_properties is None
    assert var_83.max_properties is None
    assert var_83.required == []
    var_84 = 'minItems'
    var_85 = {var_3: var_4}
    var_86 = {var_3: var_16, var_15: var_85, var_84: var_0}
    var_87 = module_0.from_json_schema(var_86)
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.fields.Array'
    assert var_87.title == ''
    assert var_87.description == ''
    assert var_87.allow_null is False
    assert var_87.read_only is False
    assert f'{type(var_87.items).__module__}.{type(var_87.items).__qualname__}' == 'typesystem.fields.String'
    assert var_87.additional_items is True
    assert var_87.min_items is True
    assert var_87.max_items is None
    assert var_87.unique_items is False
    var_88 = 'uniqueItems'
    var_89 = {var_3: var_16, var_15: var_59, var_88: var_0}
    var_90 = module_0.from_json_schema(var_89)
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'typesystem.fields.Array'
    assert var_90.title == ''
    assert var_90.description == ''
    assert var_90.allow_null is False
    assert var_90.read_only is False
    assert f'{type(var_90.items).__module__}.{type(var_90.items).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_90.additional_items is True
    assert var_90.min_items == 0
    assert var_90.max_items is None
    assert var_90.unique_items is True
    var_91 = 'maxLength'
    var_92 = 10
    var_93 = {var_3: var_4, var_29: var_30, var_91: var_92}
    var_94 = module_0.from_json_schema(var_93)
    assert f'{type(var_94).__module__}.{type(var_94).__qualname__}' == 'typesystem.fields.String'
    assert var_94.title == ''
    assert var_94.description == ''
    assert var_94.allow_null is False
    assert var_94.read_only is False
    assert var_94.allow_blank is False
    assert var_94.trim_whitespace is True
    assert var_94.max_length == 10
    assert var_94.min_length == 5
    assert var_94.format is None
    assert var_94.coerce_types is False
    assert var_94.pattern is None
    assert var_94.pattern_regex is None
    var_95 = [var_4, var_9]
    var_96 = {var_3: var_95}
    var_97 = module_0.from_json_schema(var_96)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.fields.Union'
    assert var_97.title == ''
    assert var_97.description == ''
    assert var_97.allow_null is False
    assert var_97.read_only is False
    assert f'{type(var_97.any_of).__module__}.{type(var_97.any_of).__qualname__}' == 'builtins.list'
    assert len(var_97.any_of) == 2
    var_98 = {var_3: var_4}
    var_99 = module_0.from_json_schema(var_98)
    assert f'{type(var_99).__module__}.{type(var_99).__qualname__}' == 'typesystem.fields.String'
    assert var_99.title == ''
    assert var_99.description == ''
    assert var_99.allow_null is False
    assert var_99.read_only is False
    assert var_99.default == ''
    assert var_99.allow_blank is True
    assert var_99.trim_whitespace is True
    assert var_99.max_length is None
    assert var_99.min_length is None
    assert var_99.format is None
    assert var_99.coerce_types is False
    assert var_99.pattern is None
    assert var_99.pattern_regex is None
    var_100 = module_4.Definitions()
    assert f'{type(var_100).__module__}.{type(var_100).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_100) == 0
    var_101 = {var_57: var_58}
    var_102 = module_0.from_json_schema(var_101, var_100)
    assert f'{type(var_102).__module__}.{type(var_102).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_102.title == ''
    assert var_102.description == ''
    assert var_102.allow_null is False
    assert var_102.read_only is False
    assert var_102.to == '#/components/schemas/Example'
    assert f'{type(var_102.definitions).__module__}.{type(var_102.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_102.definitions) == 0
    var_103 = {}
    var_104 = module_0.from_json_schema(var_103)
    assert f'{type(var_104).__module__}.{type(var_104).__qualname__}' == 'typesystem.fields.Any'
    assert var_104.title == ''
    assert var_104.description == ''
    assert var_104.allow_null is False
    assert var_104.read_only is False

def test_case_31():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_0.ref_from_json_schema(var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == '#/components/schemas/User'
    assert f'{type(var_4.definitions).__module__}.{type(var_4.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_5 = '$ref'
    var_6 = 'http://example.com'
    var_7 = {var_5: var_6}
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_7, var_0)

def test_case_32():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_4.Definitions()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 0
    var_7 = module_0.enum_from_json_schema(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Choice'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_7.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_8 = 'default'
    var_9 = 1
    var_10 = 2
    var_11 = [var_9, var_10, var_9]
    var_12 = {var_0: var_11, var_8: var_10}
    var_13 = module_4.Definitions()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_14 = module_0.enum_from_json_schema(var_12, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Choice'
    assert var_14.default == 2
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.choices == [(1, 1), (2, 2), (1, 1)]
    assert var_14.coerce_types is True
    var_15 = []
    var_16 = {var_0: var_15}
    var_17 = module_4.Definitions()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_17) == 0
    var_18 = module_0.enum_from_json_schema(var_16, var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Choice'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.choices == []
    assert var_18.coerce_types is True

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_1.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
    var_5 = 10
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = module_2.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length == 10
    assert var_8.min_length is True
    assert var_8.format == 'email'
    assert var_8.coerce_types is True
    assert var_8.pattern == '^[a-z]+$'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = 100
    var_11 = 5
    var_12 = module_2.Integer(minimum=var_4, maximum=var_10, multiple_of=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Integer'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum is True
    assert var_12.maximum == 100
    assert var_12.exclusive_minimum is None
    assert var_12.exclusive_maximum is None
    assert var_12.multiple_of == 5
    assert var_12.precision is None
    assert var_12.coerce_types is True
    var_13 = 'integer'
    var_14 = module_0.to_json_schema(var_12)
    var_15 = module_2.Boolean()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_2.Array(var_8, min_items=var_4, max_items=var_5, unique_items=var_4)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Array'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.items).__module__}.{type(var_17.items).__qualname__}' == 'typesystem.fields.String'
    assert var_17.additional_items is False
    assert var_17.min_items is True
    assert var_17.max_items == 10
    assert var_17.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_18 = module_0.to_json_schema(var_17)
    var_19 = 'name'
    var_20 = module_2.String()
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
    var_21 = {var_19: var_20}
    var_22 = [var_19]
    var_23 = module_2.Object(properties=var_21, required=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Object'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.properties).__module__}.{type(var_23.properties).__qualname__}' == 'builtins.dict'
    assert len(var_23.properties) == 1
    assert var_23.pattern_properties == {}
    assert var_23.additional_properties is True
    assert var_23.property_names is None
    assert var_23.min_properties is None
    assert var_23.max_properties is None
    assert var_23.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_2.String()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.String'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.allow_blank is False
    assert var_25.trim_whitespace is True
    assert var_25.max_length is None
    assert var_25.min_length is None
    assert var_25.format is None
    assert var_25.coerce_types is True
    assert var_25.pattern is None
    assert var_25.pattern_regex is None
    var_26 = {var_19: var_25}
    var_27 = module_2.Object(properties=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Object'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.properties).__module__}.{type(var_27.properties).__qualname__}' == 'builtins.dict'
    assert len(var_27.properties) == 1
    assert var_27.pattern_properties == {}
    assert var_27.additional_properties is True
    assert var_27.property_names is None
    assert var_27.min_properties is None
    assert var_27.max_properties is None
    assert var_27.required == []
    var_28 = module_3.escape(var_13)
    assert var_28 == 'integer'
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
    module_0.to_json_schema(var_28)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = var_0.get_default_value()
    var_2 = module_1.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_2.String()
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_0.to_json_schema(var_4)
    var_6 = module_2.Integer()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Integer'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.minimum is None
    assert var_6.maximum is None
    assert var_6.exclusive_minimum is None
    assert var_6.exclusive_maximum is None
    assert var_6.multiple_of is None
    assert var_6.precision is None
    assert var_6.coerce_types is True
    var_7 = module_2.Float()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Float'
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
    var_8 = module_0.to_json_schema(var_7)
    module_0.to_json_schema(var_1)

def test_case_35():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = False
    var_3 = 10
    var_4 = '^[a-z]+$'
    var_5 = 'email'
    var_6 = module_2.String(max_length=var_3, min_length=var_2, pattern=var_4, format=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.String'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.allow_blank is False
    assert var_6.trim_whitespace is True
    assert var_6.max_length == 10
    assert var_6.min_length is False
    assert var_6.format == 'email'
    assert var_6.coerce_types is True
    assert var_6.pattern == '^[a-z]+$'
    assert f'{type(var_6.pattern_regex).__module__}.{type(var_6.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = module_0.to_json_schema(var_6)
    var_8 = 0
    var_9 = 100
    var_10 = 5
    var_11 = module_2.Integer(minimum=var_8, maximum=var_9, multiple_of=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Integer'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.minimum == 0
    assert var_11.maximum == 100
    assert var_11.exclusive_minimum is None
    assert var_11.exclusive_maximum is None
    assert var_11.multiple_of == 5
    assert var_11.precision is None
    assert var_11.coerce_types is True
    var_12 = module_0.to_json_schema(var_11)
    var_13 = module_2.Boolean()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_14 = module_0.to_json_schema(var_13)
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
    var_16 = module_2.Array(var_15, min_items=var_2, max_items=var_3, unique_items=var_2)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Array'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert f'{type(var_16.items).__module__}.{type(var_16.items).__qualname__}' == 'typesystem.fields.String'
    assert var_16.additional_items is False
    assert var_16.min_items is False
    assert var_16.max_items == 10
    assert var_16.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_17 = module_0.to_json_schema(var_16)
    var_18 = 'name'
    var_19 = module_2.String()
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
    var_20 = {var_18: var_19}
    var_21 = [var_18]
    var_22 = module_2.Object(properties=var_20, required=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Object'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.properties).__module__}.{type(var_22.properties).__qualname__}' == 'builtins.dict'
    assert len(var_22.properties) == 1
    assert var_22.pattern_properties == {}
    assert var_22.additional_properties is True
    assert var_22.property_names is None
    assert var_22.min_properties is None
    assert var_22.max_properties is None
    assert var_22.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_23 = module_0.to_json_schema(var_22)
    var_24 = 'Person'
    var_25 = module_2.String()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.String'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.allow_blank is False
    assert var_25.trim_whitespace is True
    assert var_25.max_length is None
    assert var_25.min_length is None
    assert var_25.format is None
    assert var_25.coerce_types is True
    assert var_25.pattern is None
    assert var_25.pattern_regex is None
    var_26 = {var_24: var_22}
    var_27 = module_4.Reference(var_24, var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.to == 'Person'
    assert f'{type(var_27.definitions).__module__}.{type(var_27.definitions).__qualname__}' == 'builtins.dict'
    assert len(var_27.definitions) == 1
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_28 = module_0.to_json_schema(var_27)
    var_29 = module_2.String()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.String'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert var_29.allow_blank is False
    assert var_29.trim_whitespace is True
    assert var_29.max_length is None
    assert var_29.min_length is None
    assert var_29.format is None
    assert var_29.coerce_types is True
    assert var_29.pattern is None
    assert var_29.pattern_regex is None
    var_30 = module_2.Integer()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Integer'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.minimum is None
    assert var_30.maximum is None
    assert var_30.exclusive_minimum is None
    assert var_30.exclusive_maximum is None
    assert var_30.multiple_of is None
    assert var_30.precision is None
    assert var_30.coerce_types is True
    var_31 = [var_29, var_30]
    var_32 = module_2.Union(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Union'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.any_of).__module__}.{type(var_32.any_of).__qualname__}' == 'builtins.list'
    assert len(var_32.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_33 = module_0.to_json_schema(var_32)
    var_34 = module_2.String(min_length=var_2)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.String'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.allow_blank is False
    assert var_34.trim_whitespace is True
    assert var_34.max_length is None
    assert var_34.min_length is False
    assert var_34.format is None
    assert var_34.coerce_types is True
    assert var_34.pattern is None
    assert var_34.pattern_regex is None
    var_35 = module_2.String(max_length=var_3)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.String'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.allow_blank is False
    assert var_35.trim_whitespace is True
    assert var_35.max_length == 10
    assert var_35.min_length is None
    assert var_35.format is None
    assert var_35.coerce_types is True
    assert var_35.pattern is None
    assert var_35.pattern_regex is None
    var_36 = module_1.AllOf(var_31)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert f'{type(var_36.all_of).__module__}.{type(var_36.all_of).__qualname__}' == 'builtins.list'
    assert len(var_36.all_of) == 2
    var_37 = module_2.String(min_length=var_10)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.String'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.allow_blank is False
    assert var_37.trim_whitespace is True
    assert var_37.max_length is None
    assert var_37.min_length == 5
    assert var_37.format is None
    assert var_37.coerce_types is True
    assert var_37.pattern is None
    assert var_37.pattern_regex is None
    var_38 = module_2.String(max_length=var_3)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.String'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.allow_blank is False
    assert var_38.trim_whitespace is True
    assert var_38.max_length == 10
    assert var_38.min_length is None
    assert var_38.format is None
    assert var_38.coerce_types is True
    assert var_38.pattern is None
    assert var_38.pattern_regex is None
    var_39 = module_2.Integer()
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Integer'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.minimum is None
    assert var_39.maximum is None
    assert var_39.exclusive_minimum is None
    assert var_39.exclusive_maximum is None
    assert var_39.multiple_of is None
    assert var_39.precision is None
    assert var_39.coerce_types is True
    var_40 = module_1.IfThenElse(var_37, var_38, var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert f'{type(var_40.if_clause).__module__}.{type(var_40.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_40.then_clause).__module__}.{type(var_40.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_40.else_clause).__module__}.{type(var_40.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_41 = module_0.to_json_schema(var_40)
    var_42 = 'All tests passed!'
    var_43 = print(var_42)

def test_case_36():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
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
    with pytest.raises(AttributeError):
        var_3 = var_2.any_of

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = var_0.get_default_value()
    var_2 = module_1.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_2.String()
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_0.to_json_schema(var_4)
    var_6 = module_2.Integer()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Integer'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.minimum is None
    assert var_6.maximum is None
    assert var_6.exclusive_minimum is None
    assert var_6.exclusive_maximum is None
    assert var_6.multiple_of is None
    assert var_6.precision is None
    assert var_6.coerce_types is True
    var_7 = module_0.to_json_schema(var_6)
    var_8 = module_2.Float()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Float'
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
    var_9 = module_0.to_json_schema(var_8)
    var_10 = module_2.Boolean()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_11 = module_0.to_json_schema(var_10)
    var_12 = module_2.Array()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Array'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.items is None
    assert var_12.additional_items is False
    assert var_12.min_items is None
    assert var_12.max_items is None
    assert var_12.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_13 = module_0.to_json_schema(var_12)
    var_14 = module_2.Object()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Object'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.properties == {}
    assert var_14.pattern_properties == {}
    assert var_14.additional_properties is True
    assert var_14.property_names is None
    assert var_14.min_properties is None
    assert var_14.max_properties is None
    assert var_14.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_15 = module_0.to_json_schema(var_14)
    module_4.Schema()

def test_case_38():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_1.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
    var_5 = 10
    var_6 = '^[a-zA-Z]+$'
    var_7 = 'email'
    var_8 = module_2.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length == 10
    assert var_8.min_length is True
    assert var_8.format == 'email'
    assert var_8.coerce_types is True
    assert var_8.pattern == '^[a-zA-Z]+$'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = 0
    var_11 = 11
    var_12 = 2
    var_13 = module_2.Integer(minimum=var_4, maximum=var_5, exclusive_minimum=var_10, exclusive_maximum=var_11, multiple_of=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum is True
    assert var_13.maximum == 10
    assert var_13.exclusive_minimum == 0
    assert var_13.exclusive_maximum == 11
    assert var_13.multiple_of == 2
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = module_0.to_json_schema(var_13)
    var_15 = module_2.Boolean()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_2.String()
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
    var_18 = module_2.Array(var_17, var_4, var_4, var_5, unique_items=var_4)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Array'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert f'{type(var_18.items).__module__}.{type(var_18.items).__qualname__}' == 'typesystem.fields.String'
    assert var_18.additional_items is True
    assert var_18.min_items is True
    assert var_18.max_items == 10
    assert var_18.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_19 = module_0.to_json_schema(var_18)
    var_20 = 'name'
    var_21 = module_2.String()
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
    var_22 = {var_20: var_21}
    var_23 = '^[a-z]+$'
    var_24 = module_2.Integer()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Integer'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.minimum is None
    assert var_24.maximum is None
    assert var_24.exclusive_minimum is None
    assert var_24.exclusive_maximum is None
    assert var_24.multiple_of is None
    assert var_24.precision is None
    assert var_24.coerce_types is True
    var_25 = {var_23: var_24}
    var_26 = module_2.String()
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
    var_27 = [var_20]
    var_28 = module_2.Object(properties=var_22, pattern_properties=var_25, additional_properties=var_4, property_names=var_26, min_properties=var_4, max_properties=var_5, required=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Object'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.properties).__module__}.{type(var_28.properties).__qualname__}' == 'builtins.dict'
    assert len(var_28.properties) == 1
    assert f'{type(var_28.pattern_properties).__module__}.{type(var_28.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_28.pattern_properties) == 1
    assert var_28.additional_properties is True
    assert f'{type(var_28.property_names).__module__}.{type(var_28.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_28.min_properties is True
    assert var_28.max_properties == 10
    assert var_28.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = module_2.String()
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
    var_31 = {var_20: var_30}
    var_32 = [var_20]
    var_33 = module_4.Schema(var_31)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.fields).__module__}.{type(var_33.fields).__qualname__}' == 'builtins.dict'
    assert len(var_33.fields) == 1
    assert var_33.required == ['name']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_34 = module_0.to_json_schema(var_33)
    var_35 = 'a'
    var_36 = 'A'
    var_37 = (var_35, var_36)
    var_38 = 'b'
    var_39 = 'B'
    var_40 = (var_38, var_39)
    var_41 = [var_37, var_40]
    var_42 = module_2.Choice(choices=var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.Choice'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert var_42.choices == [('a', 'A'), ('b', 'B')]
    assert var_42.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_43 = module_0.to_json_schema(var_42)
    var_44 = 'constant'
    var_45 = module_2.Const(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Const'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.const == 'constant'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_46 = module_0.to_json_schema(var_45)
    var_47 = module_2.String()
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.String'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert var_47.allow_blank is False
    assert var_47.trim_whitespace is True
    assert var_47.max_length is None
    assert var_47.min_length is None
    assert var_47.format is None
    assert var_47.coerce_types is True
    assert var_47.pattern is None
    assert var_47.pattern_regex is None
    var_48 = module_2.Integer()
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.Integer'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert var_48.minimum is None
    assert var_48.maximum is None
    assert var_48.exclusive_minimum is None
    assert var_48.exclusive_maximum is None
    assert var_48.multiple_of is None
    assert var_48.precision is None
    assert var_48.coerce_types is True
    var_49 = [var_47, var_48]
    var_50 = module_2.Union(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.fields.Union'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert f'{type(var_50.any_of).__module__}.{type(var_50.any_of).__qualname__}' == 'builtins.list'
    assert len(var_50.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_51 = module_0.to_json_schema(var_50)
    var_52 = module_2.String()
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.String'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert var_52.allow_blank is False
    assert var_52.trim_whitespace is True
    assert var_52.max_length is None
    assert var_52.min_length is None
    assert var_52.format is None
    assert var_52.coerce_types is True
    assert var_52.pattern is None
    assert var_52.pattern_regex is None
    var_53 = module_2.Integer()
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.Integer'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert var_53.minimum is None
    assert var_53.maximum is None
    assert var_53.exclusive_minimum is None
    assert var_53.exclusive_maximum is None
    assert var_53.multiple_of is None
    assert var_53.precision is None
    assert var_53.coerce_types is True
    var_54 = [var_52, var_53]
    var_55 = module_1.OneOf(var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert f'{type(var_55.one_of).__module__}.{type(var_55.one_of).__qualname__}' == 'builtins.list'
    assert len(var_55.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_56 = module_0.to_json_schema(var_55)
    var_57 = module_2.String()
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.String'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert var_57.allow_blank is False
    assert var_57.trim_whitespace is True
    assert var_57.max_length is None
    assert var_57.min_length is None
    assert var_57.format is None
    assert var_57.coerce_types is True
    assert var_57.pattern is None
    assert var_57.pattern_regex is None
    var_58 = module_2.Integer()
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.Integer'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.minimum is None
    assert var_58.maximum is None
    assert var_58.exclusive_minimum is None
    assert var_58.exclusive_maximum is None
    assert var_58.multiple_of is None
    assert var_58.precision is None
    assert var_58.coerce_types is True
    var_59 = [var_57, var_58]
    var_60 = module_1.AllOf(var_59)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert f'{type(var_60.all_of).__module__}.{type(var_60.all_of).__qualname__}' == 'builtins.list'
    assert len(var_60.all_of) == 2
    var_61 = module_0.to_json_schema(var_60)
    var_62 = module_2.String()
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
    var_63 = module_2.Integer()
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.fields.Integer'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert var_63.minimum is None
    assert var_63.maximum is None
    assert var_63.exclusive_minimum is None
    assert var_63.exclusive_maximum is None
    assert var_63.multiple_of is None
    assert var_63.precision is None
    assert var_63.coerce_types is True
    var_64 = module_2.Boolean()
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert var_64.coerce_types is True
    var_65 = module_1.IfThenElse(var_62, var_63, var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert f'{type(var_65.if_clause).__module__}.{type(var_65.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_65.then_clause).__module__}.{type(var_65.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_65.else_clause).__module__}.{type(var_65.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_66 = module_0.to_json_schema(var_65)
    var_67 = module_2.String()
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
    var_68 = module_1.Not(var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.composites.Not'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert f'{type(var_68.negated).__module__}.{type(var_68.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_69 = module_0.to_json_schema(var_68)
    var_70 = module_2.String()
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
    var_71 = {var_20: var_70}
    var_72 = module_4.Schema(var_71)
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert f'{type(var_72.fields).__module__}.{type(var_72.fields).__qualname__}' == 'builtins.dict'
    assert len(var_72.fields) == 1
    assert var_72.required == ['name']

def test_case_39():
    var_0 = 'test_default'
    var_1 = module_2.Field(default=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default == 'test_default'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_2 = module_0.get_standard_properties(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_2.Field()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    var_4 = module_0.get_standard_properties(var_3)
    var_5 = None
    var_6 = module_2.Field(default=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Field'
    assert var_6.default is None
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    var_7 = module_0.get_standard_properties(var_6)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = module_2.Field(default=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Field'
    assert var_11.default == {'key': 'value'}
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    var_12 = module_0.get_standard_properties(var_11)
    var_13 = module_2.Field(description=var_9, default=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Field'
    assert var_13.default == 'test_default'
    assert var_13.title == ''
    assert var_13.description == 'value'
    assert var_13.allow_null is False
    assert var_13.read_only is False
    var_14 = module_0.get_standard_properties(var_13)

def test_case_40():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = module_0.from_json_schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.default == ''
    assert var_7.allow_blank is True
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format is None
    assert var_7.coerce_types is False
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = 'integer'
    var_9 = {var_4: var_8}
    var_10 = module_0.from_json_schema(var_9)
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
    assert var_10.coerce_types is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_11 = 'number'
    var_12 = {var_4: var_11}
    var_13 = module_0.from_json_schema(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Float'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum is None
    assert var_13.maximum is None
    assert var_13.exclusive_minimum is None
    assert var_13.exclusive_maximum is None
    assert var_13.multiple_of is None
    assert var_13.precision is None
    assert var_13.coerce_types is False
    var_14 = 'boolean'
    var_15 = {var_4: var_14}
    var_16 = module_0.from_json_schema(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_17 = 'items'
    var_18 = 'array'
    var_19 = {var_4: var_5}
    var_20 = {var_4: var_18, var_17: var_19}
    var_21 = module_0.from_json_schema(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Array'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.items).__module__}.{type(var_21.items).__qualname__}' == 'typesystem.fields.String'
    assert var_21.additional_items is True
    assert var_21.min_items == 0
    assert var_21.max_items is None
    assert var_21.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_22 = 'properties'
    var_23 = 'object'
    var_24 = 'name'
    var_25 = {var_4: var_5}
    var_26 = {var_24: var_25}
    var_27 = {var_4: var_23, var_22: var_26}
    var_28 = module_0.from_json_schema(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Object'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.properties).__module__}.{type(var_28.properties).__qualname__}' == 'builtins.dict'
    assert len(var_28.properties) == 1
    assert var_28.pattern_properties == {}
    assert var_28.additional_properties is None
    assert var_28.property_names is None
    assert var_28.min_properties is None
    assert var_28.max_properties is None
    assert var_28.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_29 = 'enum'
    var_30 = 'allOf'
    var_31 = {var_4: var_5}
    var_32 = 'minLength'
    var_33 = -3
    var_34 = {var_32: var_33}
    var_35 = [var_31, var_34]
    var_36 = {var_30: var_35}
    var_37 = module_0.from_json_schema(var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert f'{type(var_37.all_of).__module__}.{type(var_37.all_of).__qualname__}' == 'builtins.list'
    assert len(var_37.all_of) == 2
    var_38 = 'anyOf'
    var_39 = {var_4: var_5}
    var_40 = {var_4: var_11}
    var_41 = [var_39, var_40]
    var_42 = {var_38: var_41}
    var_43 = module_0.from_json_schema(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Union'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert f'{type(var_43.any_of).__module__}.{type(var_43.any_of).__qualname__}' == 'builtins.list'
    assert len(var_43.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_44 = 'oneOf'
    var_45 = {var_4: var_11}
    var_46 = [var_6, var_45]
    var_47 = {var_44: var_46}
    var_48 = module_0.from_json_schema(var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert f'{type(var_48.one_of).__module__}.{type(var_48.one_of).__qualname__}' == 'builtins.list'
    assert len(var_48.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_49 = 'not'
    var_50 = {var_4: var_5}
    var_51 = {var_49: var_50}
    var_52 = module_0.from_json_schema(var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.composites.Not'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert f'{type(var_52.negated).__module__}.{type(var_52.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_53 = 'if'
    var_54 = 'then'
    var_55 = {var_32: var_33}
    var_56 = {var_53: var_9, var_54: var_55}
    var_57 = module_0.from_json_schema(var_56)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert f'{type(var_57.if_clause).__module__}.{type(var_57.if_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_57.then_clause).__module__}.{type(var_57.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_57.else_clause).__module__}.{type(var_57.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_58 = module_4.Definitions()
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_58) == 0
    var_59 = '$ref'
    var_60 = '#/components/schemas/Example'
    var_61 = {var_59: var_60}
    var_62 = module_0.from_json_schema(var_61, var_58)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert var_62.to == '#/components/schemas/Example'
    assert f'{type(var_62.definitions).__module__}.{type(var_62.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_62.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_63 = {var_59: var_60}
    var_64 = {var_24: var_63}
    var_65 = {var_4: var_23, var_22: var_64}
    var_66 = module_0.from_json_schema(var_65, var_58)
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.fields.Object'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert f'{type(var_66.properties).__module__}.{type(var_66.properties).__qualname__}' == 'builtins.dict'
    assert len(var_66.properties) == 1
    assert var_66.pattern_properties == {}
    assert var_66.additional_properties is None
    assert var_66.property_names is None
    assert var_66.min_properties is None
    assert var_66.max_properties is None
    assert var_66.required == []
    var_67 = 'required'
    var_68 = 'age'
    var_69 = {var_29: var_18, var_4: var_5}
    var_70 = ' f5\x0b$/='
    var_71 = {var_4: var_8, var_70: var_2}
    var_72 = {var_24: var_69, var_68: var_71}
    var_73 = [var_24]
    var_74 = {var_4: var_23, var_22: var_72, var_67: var_73}
    var_75 = module_0.from_json_schema(var_74)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.fields.Object'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert f'{type(var_75.properties).__module__}.{type(var_75.properties).__qualname__}' == 'builtins.dict'
    assert len(var_75.properties) == 2
    assert var_75.pattern_properties == {}
    assert var_75.additional_properties is None
    assert var_75.property_names is None
    assert var_75.min_properties is None
    assert var_75.max_properties is None
    assert var_75.required == ['name']
    var_76 = 'additionalProperties'
    var_77 = {var_4: var_5}
    var_78 = {var_4: var_23, var_76: var_77}
    var_79 = module_0.from_json_schema(var_78)
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.fields.Object'
    assert var_79.title == ''
    assert var_79.description == ''
    assert var_79.allow_null is False
    assert var_79.read_only is False
    assert var_79.properties == {}
    assert var_79.pattern_properties == {}
    assert f'{type(var_79.additional_properties).__module__}.{type(var_79.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_79.property_names is None
    assert var_79.min_properties is None
    assert var_79.max_properties is None
    assert var_79.required == []
    var_80 = 'patternProperties'
    var_81 = '^[a-z]+$'
    var_82 = {var_4: var_5}
    var_83 = {var_81: var_82}
    var_84 = {var_4: var_23, var_80: var_83}
    var_85 = module_0.from_json_schema(var_84)
    assert f'{type(var_85).__module__}.{type(var_85).__qualname__}' == 'typesystem.fields.Object'
    assert var_85.title == ''
    assert var_85.description == ''
    assert var_85.allow_null is False
    assert var_85.read_only is False
    assert var_85.properties == {}
    assert f'{type(var_85.pattern_properties).__module__}.{type(var_85.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_85.pattern_properties) == 1
    assert var_85.additional_properties is None
    assert var_85.property_names is None
    assert var_85.min_properties is None
    assert var_85.max_properties is None
    assert var_85.required == []
    var_86 = 'minItems'
    var_87 = {var_4: var_5}
    var_88 = {var_4: var_18, var_17: var_87, var_86: var_0}
    var_89 = module_0.from_json_schema(var_88)
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.fields.Array'
    assert var_89.title == ''
    assert var_89.description == ''
    assert var_89.allow_null is False
    assert var_89.read_only is False
    assert f'{type(var_89.items).__module__}.{type(var_89.items).__qualname__}' == 'typesystem.fields.String'
    assert var_89.additional_items is True
    assert var_89.min_items is True
    assert var_89.max_items is None
    assert var_89.unique_items is False
    var_90 = 'uniqueItems'
    var_91 = {var_4: var_5}
    var_92 = {var_4: var_18, var_17: var_91, var_90: var_0}
    var_93 = module_0.from_json_schema(var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.Array'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert f'{type(var_93.items).__module__}.{type(var_93.items).__qualname__}' == 'typesystem.fields.String'
    assert var_93.additional_items is True
    assert var_93.min_items == 0
    assert var_93.max_items is None
    assert var_93.unique_items is True
    var_94 = 'maxLength'
    var_95 = 10
    var_96 = {var_4: var_5, var_32: var_33, var_94: var_95}
    var_97 = module_0.from_json_schema(var_96)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.fields.String'
    assert var_97.title == ''
    assert var_97.description == ''
    assert var_97.allow_null is False
    assert var_97.read_only is False
    assert var_97.allow_blank is False
    assert var_97.trim_whitespace is True
    assert var_97.max_length == 10
    assert var_97.min_length is None
    assert var_97.format is None
    assert var_97.coerce_types is False
    assert var_97.pattern is None
    assert var_97.pattern_regex is None
    var_98 = [var_5, var_11]
    var_99 = {var_4: var_98}
    var_100 = module_0.from_json_schema(var_99)
    assert f'{type(var_100).__module__}.{type(var_100).__qualname__}' == 'typesystem.fields.Union'
    assert var_100.title == ''
    assert var_100.description == ''
    assert var_100.allow_null is False
    assert var_100.read_only is False
    assert f'{type(var_100.any_of).__module__}.{type(var_100.any_of).__qualname__}' == 'builtins.list'
    assert len(var_100.any_of) == 2
    var_101 = {var_4: var_5}
    var_102 = module_0.from_json_schema(var_101)
    assert f'{type(var_102).__module__}.{type(var_102).__qualname__}' == 'typesystem.fields.String'
    assert var_102.title == ''
    assert var_102.description == ''
    assert var_102.allow_null is False
    assert var_102.read_only is False
    assert var_102.default == ''
    assert var_102.allow_blank is True
    assert var_102.trim_whitespace is True
    assert var_102.max_length is None
    assert var_102.min_length is None
    assert var_102.format is None
    assert var_102.coerce_types is False
    assert var_102.pattern is None
    assert var_102.pattern_regex is None
    var_103 = module_4.Definitions()
    assert f'{type(var_103).__module__}.{type(var_103).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_103) == 0
    var_104 = {var_59: var_60}
    var_105 = module_0.from_json_schema(var_104, var_103)
    assert f'{type(var_105).__module__}.{type(var_105).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_105.title == ''
    assert var_105.description == ''
    assert var_105.allow_null is False
    assert var_105.read_only is False
    assert var_105.to == '#/components/schemas/Example'
    assert f'{type(var_105.definitions).__module__}.{type(var_105.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_105.definitions) == 0
    var_106 = {}
    var_107 = module_0.from_json_schema(var_106)
    assert f'{type(var_107).__module__}.{type(var_107).__qualname__}' == 'typesystem.fields.Any'
    assert var_107.title == ''
    assert var_107.description == ''
    assert var_107.allow_null is False
    assert var_107.read_only is False

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_4.Definitions()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 0
    var_4 = module_0.from_json_schema(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.default == ''
    assert var_4.allow_blank is True
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is None
    assert var_4.format is None
    assert var_4.coerce_types is False
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_0.to_json_schema(var_3)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_6 = module_0.type_from_json_schema(var_5, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Union'
    assert var_6.default is None
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is True
    assert var_6.read_only is False
    assert f'{type(var_6.any_of).__module__}.{type(var_6.any_of).__qualname__}' == 'builtins.list'
    assert len(var_6.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_7 = var_6.any_of
    var_8 = len(var_7)
    var_9 = 0
    var_10 = var_6.any_of[var_9]
    var_11 = 'null'
    var_12 = {var_0: var_11}
    var_13 = module_4.Definitions()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13) == 0
    var_14 = module_0.type_from_json_schema(var_12, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Const'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.const is None
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_15 = module_4.Definitions()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_15) == 0
    module_0.type_from_json_schema(var_10, var_15)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_4.Definitions()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 0
    var_4 = None
    var_5 = var_3.__setitem__(var_4, var_4)
    assert len(var_3) == 1
    var_6 = module_0.from_json_schema(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.String'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.default == ''
    assert var_6.allow_blank is True
    assert var_6.trim_whitespace is True
    assert var_6.max_length is None
    assert var_6.min_length is None
    assert var_6.format is None
    assert var_6.coerce_types is False
    assert var_6.pattern is None
    assert var_6.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxItems', 'minLength', 'maxProperties', 'dependencies', 'multipleOf', 'pattern', 'boolean_schema', 'items', 'minProperties', 'propertyNames', 'required', 'minimum', 'type', 'minItems', 'maxLength', 'exclusiveMaximum', 'uniqueItems', 'contains', 'additionalProperties', 'exclusiveMinimum', 'additionalItems', 'patternProperties', 'properties', 'maximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    module_0.to_json_schema(var_3)