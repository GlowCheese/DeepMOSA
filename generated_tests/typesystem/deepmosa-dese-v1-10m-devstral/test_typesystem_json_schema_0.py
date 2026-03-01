# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import enum as module_3
import typesystem.composites as module_4
import re as module_5

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.from_json_schema(var_0, var_0)

def test_case_1():
    var_0 = {}
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
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
    var_0 = None
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.to_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.ref_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.all_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.any_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = {}
    module_0.not_from_json_schema(var_0, var_0)

def test_case_10():
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
    var_1 = module_1.Array(var_0)
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
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_11():
    var_0 = module_1.Decimal()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
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
def test_case_12():
    var_0 = None
    var_1 = module_2.Reference(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.to is None
    assert var_1.definitions is None
    assert module_2.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_2.Reference.target).__module__}.{type(module_2.Reference.target).__qualname__}' == 'builtins.property'
    var_2 = var_1.get_default_value()
    var_3 = module_3._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    module_0.to_json_schema(var_1, var_3)

def test_case_13():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_4.NeverMatch.errors == {'never': 'This never validates.'}

def test_case_14():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
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
    var_0 = module_1.Integer()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_3._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_0.type_from_json_schema(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

def test_case_17():
    var_0 = module_1.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_18():
    var_0 = module_2.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.type_from_json_schema(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 5
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_4 = module_0.from_json_schema(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False

def test_case_19():
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
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_20():
    var_0 = module_1.Array()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_21():
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
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_22():
    var_0 = module_1.Object()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_23():
    var_0 = module_2.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_24():
    var_0 = module_1.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_0)

def test_case_25():
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
    var_1 = module_1.Integer()
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
    var_3 = module_4.OneOf(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.one_of).__module__}.{type(var_3.one_of).__qualname__}' == 'builtins.list'
    assert len(var_3.one_of) == 2
    assert module_4.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_26():
    var_0 = module_1.Integer()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_1.Boolean()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_2 = module_4.IfThenElse(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
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
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.from_json_schema(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_1.Const(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Const'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.const).__module__}.{type(var_3.const).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Const'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.const).__module__}.{type(var_5.const).__qualname__}' == 'typesystem.fields.Any'
    module_0.to_json_schema(var_0)

def test_case_28():
    var_0 = module_4.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_4.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_29():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_1.Choice(choices=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Choice'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.choices == [('a', 'Option A'), ('b', 'Option B')]
    assert var_7.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_8 = module_0.to_json_schema(var_7)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_30():
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
    var_1 = module_1.Array(var_0)
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
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.from_json_schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.String'
    assert var_3.additional_items is False
    assert var_3.min_items == 0
    assert var_3.max_items is None
    assert var_3.unique_items is False

def test_case_31():
    var_0 = module_1.Integer()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
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
def test_case_32():
    var_0 = module_5.purge()
    assert module_5.ASCII == module_5.RegexFlag.ASCII
    assert module_5.A == module_5.RegexFlag.ASCII
    assert module_5.IGNORECASE == module_5.RegexFlag.IGNORECASE
    assert module_5.I == module_5.RegexFlag.IGNORECASE
    assert module_5.LOCALE == module_5.RegexFlag.LOCALE
    assert module_5.L == module_5.RegexFlag.LOCALE
    assert module_5.UNICODE == module_5.RegexFlag.UNICODE
    assert module_5.U == module_5.RegexFlag.UNICODE
    assert module_5.MULTILINE == module_5.RegexFlag.MULTILINE
    assert module_5.M == module_5.RegexFlag.MULTILINE
    assert module_5.DOTALL == module_5.RegexFlag.DOTALL
    assert module_5.S == module_5.RegexFlag.DOTALL
    assert module_5.VERBOSE == module_5.RegexFlag.VERBOSE
    assert module_5.X == module_5.RegexFlag.VERBOSE
    assert module_5.TEMPLATE == module_5.RegexFlag.TEMPLATE
    assert module_5.T == module_5.RegexFlag.TEMPLATE
    assert module_5.DEBUG == module_5.RegexFlag.DEBUG
    var_1 = module_4.Not(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.Not'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.negated is None
    assert module_4.Not.errors == {'negated': 'Must not match.'}
    module_0.to_json_schema(var_1)

def test_case_33():
    var_0 = module_1.Integer()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.type_from_json_schema(var_1, var_1)
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
    assert var_2.coerce_types is False

def test_case_34():
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
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_1.String()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_0.to_json_schema(var_2)
    var_5 = module_1.Boolean()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_6 = module_4.IfThenElse(var_3, var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.if_clause).__module__}.{type(var_6.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_6.then_clause).__module__}.{type(var_6.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_6.else_clause).__module__}.{type(var_6.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_7 = module_0.to_json_schema(var_6)
    var_8 = module_0.from_json_schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.if_clause).__module__}.{type(var_8.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_8.then_clause).__module__}.{type(var_8.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_8.else_clause).__module__}.{type(var_8.else_clause).__qualname__}' == 'typesystem.fields.Boolean'

def test_case_35():
    var_0 = module_1.Integer()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.from_json_schema(var_1)
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
    assert var_2.coerce_types is False

def test_case_36():
    var_0 = module_1.Array()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.from_json_schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items == 0
    assert var_2.max_items is None
    assert var_2.unique_items is False

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = None
    var_1 = module_3._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_0.from_json_schema(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_4.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.fields.Any'
    assert module_4.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.Not'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.negated).__module__}.{type(var_5.negated).__qualname__}' == 'typesystem.fields.Any'
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = 2478.04
    var_1 = {}
    var_2 = module_1.Decimal(exclusive_maximum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum == pytest.approx(2478.04, abs=0.01, rel=0.01)
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = None
    var_4 = module_0.to_json_schema(var_2, var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
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
    var_6 = {var_5: var_5, var_5: var_5, var_5: var_5}
    var_7 = var_2.serialize(var_3)
    var_8 = module_0.type_from_json_schema(var_6, var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Union'
    assert var_8.default is None
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is True
    assert var_8.read_only is False
    assert f'{type(var_8.any_of).__module__}.{type(var_8.any_of).__qualname__}' == 'builtins.list'
    assert len(var_8.any_of) == 5
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_9 = module_0.from_json_schema(var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Any'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    module_0.to_json_schema(var_5, var_5)

def test_case_39():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_1.String()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_1.Integer()
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
    var_5 = module_1.Object(properties=var_4)
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
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_40():
    var_0 = '^S_'
    var_1 = '^I_'
    var_2 = module_1.String()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_1.Integer()
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
    var_5 = module_1.Object(pattern_properties=var_4)
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
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_41():
    var_0 = True
    var_1 = 'name'
    var_2 = module_1.String()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_0.to_json_schema(var_4, var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_42():
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
    var_1 = module_1.Integer()
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
    var_3 = module_4.AllOf(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.all_of).__module__}.{type(var_3.all_of).__qualname__}' == 'builtins.list'
    assert len(var_3.all_of) == 2
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_43():
    var_0 = 5
    var_1 = module_1.String(min_length=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length == 5
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_44():
    var_0 = False
    var_1 = module_1.String(trim_whitespace=var_0, max_length=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is False
    assert var_1.max_length is False
    assert var_1.min_length is None
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_45():
    var_0 = 'email'
    var_1 = module_1.String(format=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_46():
    var_0 = 10
    var_1 = False
    var_2 = module_1.Integer(minimum=var_0, coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum == 10
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = None
    var_4 = {var_2: var_2, var_0: var_3, var_3: var_2, var_0: var_1}
    var_5 = module_0.to_json_schema(var_2, var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_47():
    var_0 = -1712
    var_1 = None
    var_2 = module_1.Integer(maximum=var_0, exclusive_minimum=var_1, precision=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum == -1712
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_48():
    var_0 = False
    var_1 = None
    var_2 = module_1.Integer(exclusive_minimum=var_0, exclusive_maximum=var_1, multiple_of=var_1)
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_49():
    var_0 = 5
    var_1 = module_1.Array(max_items=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.items is None
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items == 5
    assert var_1.unique_items is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_50():
    var_0 = True
    var_1 = module_1.Array(unique_items=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.items is None
    assert var_1.additional_items is False
    assert var_1.min_items is None
    assert var_1.max_items is None
    assert var_1.unique_items is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_51():
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
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = module_1.Array(additional_items=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Array'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.items is None
    assert f'{type(var_1.additional_items).__module__}.{type(var_1.additional_items).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_1.min_items is None
    assert var_1.max_items is None
    assert var_1.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_52():
    var_0 = 5
    var_1 = module_1.Object(max_properties=var_0)
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
    assert var_1.max_properties == 5
    assert var_1.required == []
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_53():
    var_0 = 2
    var_1 = module_1.Object(min_properties=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties == 2
    assert var_1.max_properties is None
    assert var_1.required == []
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_54():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = module_1.Object(required=var_1)
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
    assert var_2.required == ['name']
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_55():
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
    var_1 = module_1.Object(property_names=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert f'{type(var_1.property_names).__module__}.{type(var_1.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_56():
    var_0 = {}
    var_1 = module_2.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_57():
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
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = module_1.Object(properties=var_0, property_names=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert f'{type(var_1.additional_properties).__module__}.{type(var_1.additional_properties).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_1.property_names).__module__}.{type(var_1.property_names).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_1.min_properties is None
    assert var_1.max_properties is None
    assert var_1.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_58():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'required'
    var_3 = 'default'
    var_4 = 'object'
    var_5 = 'name'
    var_6 = 'string'
    var_7 = {var_0: var_6}
    var_8 = {var_5: var_7}
    var_9 = [var_5]
    var_10 = 'test'
    var_11 = {var_5: var_10}
    var_12 = {var_0: var_4, var_1: var_8, var_2: var_9, var_3: var_11}
    var_13 = module_2.Definitions()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13) == 0
    var_14 = False
    var_15 = module_0.from_json_schema_type(var_12, var_4, var_14, var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Object'
    assert var_15.default == {'name': 'test'}
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.properties).__module__}.{type(var_15.properties).__qualname__}' == 'builtins.dict'
    assert len(var_15.properties) == 1
    assert var_15.pattern_properties == {}
    assert var_15.additional_properties is None
    assert var_15.property_names is None
    assert var_15.min_properties is None
    assert var_15.max_properties is None
    assert var_15.required == ['name']
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_16 = var_15.properties[var_5]

def test_case_59():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'default'
    var_4 = 'string'
    var_5 = 5
    var_6 = 10
    var_7 = 'hello'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_2.Definitions()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_9) == 0
    var_10 = False
    var_11 = module_0.from_json_schema_type(var_8, var_4, var_10, var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.String'
    assert var_11.default == 'hello'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.allow_blank is False
    assert var_11.trim_whitespace is True
    assert var_11.max_length == 10
    assert var_11.min_length == 5
    assert var_11.format is None
    assert var_11.coerce_types is False
    assert var_11.pattern is None
    assert var_11.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}

def test_case_60():
    var_0 = 'if'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_2.Definitions()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 0
    var_6 = module_0.if_then_else_from_json_schema(var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.if_clause).__module__}.{type(var_6.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_6.then_clause).__module__}.{type(var_6.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_6.else_clause).__module__}.{type(var_6.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = var_6.if_clause
    var_8 = var_6.then_clause
    var_9 = var_6.else_clause

def test_case_61():
    var_0 = 5
    var_1 = module_1.Integer(multiple_of=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of == 5
    assert var_1.precision is None
    assert var_1.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_62():
    var_0 = module_2.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'oneOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = 'test'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = module_0.one_of_from_json_schema(var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_9.default == 'test'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.one_of).__module__}.{type(var_9.one_of).__qualname__}' == 'builtins.list'
    assert len(var_9.one_of) == 1
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_4.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}

@pytest.mark.xfail(strict=True)
def test_case_63():
    var_0 = '$ref'
    var_1 = '#/components/schemas/test'
    var_2 = {var_0: var_1}
    var_3 = module_2.Definitions()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 0
    var_4 = module_0.from_json_schema(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == '#/components/schemas/test'
    assert f'{type(var_4.definitions).__module__}.{type(var_4.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_2.Reference.target).__module__}.{type(module_2.Reference.target).__qualname__}' == 'builtins.property'
    var_5 = 'test'
    var_4.validate(var_5)
    assert var_6 == 'test'

def test_case_64():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Choice'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_6.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_7 = None
    var_8 = module_0.to_json_schema(var_6, var_7)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7

@pytest.mark.xfail(strict=True)
def test_case_65():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = {var_1: var_1}
    var_3 = 'minLength'
    var_4 = {var_1: var_1, var_3: var_0}
    var_5 = [var_2, var_4]
    var_6 = {var_0: var_5}
    module_0.from_json_schema(var_6)

@pytest.mark.xfail(strict=True)
def test_case_66():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'minLength'
    var_5 = 5
    var_6 = {var_1: var_2, var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.all_of).__module__}.{type(var_9.all_of).__qualname__}' == 'builtins.list'
    assert len(var_9.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_10 = 'a'
    var_9.validate(var_10)

def test_case_67():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.one_of).__module__}.{type(var_8.one_of).__qualname__}' == 'builtins.list'
    assert len(var_8.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_4.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_9 = 'test'
    var_10 = var_8.validate(var_9)
    assert var_10 == 'test'
    var_11 = 1
    var_12 = var_8.validate(var_11)
    assert var_12 == pytest.approx(1.0, abs=0.01, rel=0.01)

def test_case_68():
    var_0 = 'type'
    var_1 = 'enum'
    var_2 = 'string'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4, var_3]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.from_json_schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.all_of).__module__}.{type(var_7.all_of).__qualname__}' == 'builtins.list'
    assert len(var_7.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = module_0.to_json_schema(var_7)

def test_case_69():
    var_0 = 'type'
    var_1 = 'propertyNames'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = {}
    var_7 = False
    var_8 = module_0.from_json_schema_type(var_5, var_2, var_7, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Object'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.properties == {}
    assert var_8.pattern_properties == {}
    assert var_8.additional_properties is None
    assert f'{type(var_8.property_names).__module__}.{type(var_8.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_8.min_properties is None
    assert var_8.max_properties is None
    assert var_8.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}

def test_case_70():
    var_0 = 'items'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = {var_0: var_4}
    var_6 = module_2.Definitions()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 0
    var_7 = 'array'
    var_8 = False
    var_9 = module_0.from_json_schema_type(var_5, var_7, var_8, var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Array'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.items).__module__}.{type(var_9.items).__qualname__}' == 'builtins.list'
    assert len(var_9.items) == 1
    assert var_9.additional_items is True
    assert var_9.min_items == 0
    assert var_9.max_items is None
    assert var_9.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_10 = var_9.items
    var_11 = var_9.items
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_9.items[var_8]

def test_case_71():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = module_1.Object(required=var_1)
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
    assert var_2.required == ['name']
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_0.from_json_schema(var_3)
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
    assert var_4.required == ['name']

@pytest.mark.xfail(strict=True)
def test_case_72():
    var_0 = module_2.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.type_from_json_schema(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 5
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = var_2.serialize(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 0
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_4 = module_1.Boolean()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_5 = module_4.IfThenElse(var_3, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.if_clause).__module__}.{type(var_5.if_clause).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5.if_clause) == 0
    assert f'{type(var_5.then_clause).__module__}.{type(var_5.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_5.else_clause).__module__}.{type(var_5.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_6 = module_0.to_json_schema(var_5)
    var_7 = var_3.setdefault(var_2)
    assert len(var_0) == 1
    assert len(var_3) == 1
    assert len(var_5.if_clause) == 1
    var_8 = module_0.type_from_json_schema(var_3, var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Union'
    assert var_8.default is None
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is True
    assert var_8.read_only is False
    assert f'{type(var_8.any_of).__module__}.{type(var_8.any_of).__qualname__}' == 'builtins.list'
    assert len(var_8.any_of) == 5
    module_0.to_json_schema(var_0, var_3)

def test_case_73():
    var_0 = 'type'
    var_1 = 'patternProperties'
    var_2 = 'object'
    var_3 = '^S_'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_2.Definitions()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = module_0.type_from_json_schema(var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Object'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.properties == {}
    assert f'{type(var_9.pattern_properties).__module__}.{type(var_9.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_9.pattern_properties) == 1
    assert var_9.additional_properties is None
    assert var_9.property_names is None
    assert var_9.min_properties is None
    assert var_9.max_properties is None
    assert var_9.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_10 = var_9.pattern_properties[var_3]

def test_case_74():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_2.Definitions()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = module_0.type_from_json_schema(var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Const'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.const is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}

def test_case_75():
    var_0 = 'type'
    var_1 = 'additionalProperties'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_5.purge()
    assert module_5.ASCII == module_5.RegexFlag.ASCII
    assert module_5.A == module_5.RegexFlag.ASCII
    assert module_5.IGNORECASE == module_5.RegexFlag.IGNORECASE
    assert module_5.I == module_5.RegexFlag.IGNORECASE
    assert module_5.LOCALE == module_5.RegexFlag.LOCALE
    assert module_5.L == module_5.RegexFlag.LOCALE
    assert module_5.UNICODE == module_5.RegexFlag.UNICODE
    assert module_5.U == module_5.RegexFlag.UNICODE
    assert module_5.MULTILINE == module_5.RegexFlag.MULTILINE
    assert module_5.M == module_5.RegexFlag.MULTILINE
    assert module_5.DOTALL == module_5.RegexFlag.DOTALL
    assert module_5.S == module_5.RegexFlag.DOTALL
    assert module_5.VERBOSE == module_5.RegexFlag.VERBOSE
    assert module_5.X == module_5.RegexFlag.VERBOSE
    assert module_5.TEMPLATE == module_5.RegexFlag.TEMPLATE
    assert module_5.T == module_5.RegexFlag.TEMPLATE
    assert module_5.DEBUG == module_5.RegexFlag.DEBUG
    var_7 = module_0.type_from_json_schema(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert f'{type(var_7.additional_properties).__module__}.{type(var_7.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = var_7.additional_properties

def test_case_76():
    var_0 = 'fixed_value'
    var_1 = module_1.String(pattern=var_0, format=var_0, coerce_types=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'fixed_value'
    assert var_1.coerce_types == 'fixed_value'
    assert var_1.pattern == 'fixed_value'
    assert f'{type(var_1.pattern_regex).__module__}.{type(var_1.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_77():
    var_0 = module_1.Array()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_1 = None
    var_2 = module_1.Array(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert f'{type(var_2.additional_items).__module__}.{type(var_2.additional_items).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.min_items is None
    assert var_2.max_items is None
    assert var_2.unique_items is False
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'boolean_schema', 'maxProperties', 'propertyNames', 'pattern', 'multipleOf', 'items', 'required', 'exclusiveMaximum', 'dependencies', 'exclusiveMinimum', 'contains', 'additionalItems', 'maxItems', 'minItems', 'properties', 'patternProperties', 'maxLength', 'minProperties', 'type', 'additionalProperties', 'maximum', 'uniqueItems', 'minimum', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_0.from_json_schema(var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.items is None
    assert f'{type(var_4.additional_items).__module__}.{type(var_4.additional_items).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.min_items == 0
    assert var_4.max_items is None
    assert var_4.unique_items is False