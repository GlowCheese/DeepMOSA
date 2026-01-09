# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_2

import pytest
import typesystem.composites as module_1
import typesystem.fields as module_3
import typesystem.json_schema as module_0
import typesystem.schemas as module_4


def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_1():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.from_json_schema(var_0)

def test_case_3():
    var_0 = None
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_0.type_from_json_schema(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}

def test_case_4():
    var_0 = None
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.all_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.not_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.NeverMatch()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = module_0.to_json_schema(var_1)
    assert var_2 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = 'U#%!L2_M_lR4TrP&zVS'
    var_1.validation_error(var_3)

def test_case_11():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_0.to_json_schema(var_2, var_1)
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.to_json_schema(var_1)
    assert var_2 is True
    var_3 = None
    module_0.from_json_schema(var_3)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    module_0.any_of_from_json_schema(var_0, var_0)

def test_case_15():
    var_0 = False
    var_1 = module_3.Field(read_only=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Field.errors == {}
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_1)

def test_case_16():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_4.Definitions(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.from_json_schema(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_4.Definitions(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    var_4 = var_1.__setitem__(var_2, var_3)
    assert len(var_1) == 1
    module_0.to_json_schema(var_1)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.NeverMatch()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = module_1.AllOf(var_0, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.all_of).__module__}.{type(var_2.all_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.all_of) == 0
    var_3 = None
    var_4 = module_0.to_json_schema(var_2, var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    module_2.unique(var_3)

def test_case_19():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_0.to_json_schema(var_2, var_1)
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_4 = module_0.from_json_schema(var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_4.Definitions(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = None
    var_4 = module_4.Schema(var_0, **var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_4.fields) == 0
    assert var_4.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_0.from_json_schema(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    var_6 = module_0.to_json_schema(var_4, var_3)
    var_7 = module_0.from_json_schema(var_6, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is None
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = var_1.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8) == 0
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    module_0.enum_from_json_schema(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_4.Definitions(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = None
    var_4 = module_1.NeverMatch()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_5 = module_1.Not(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.Not'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.negated).__module__}.{type(var_5.negated).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_6 = module_0.from_json_schema(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Any'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    var_7 = None
    var_8 = module_0.to_json_schema(var_5, var_7)
    module_0.to_json_schema(var_3)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_4.Definitions(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = None
    var_4 = module_3.Const(var_0, **var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Const'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.const).__module__}.{type(var_4.const).__qualname__}' == 'enum._EnumDict'
    assert len(var_4.const) == 0
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_5 = module_0.from_json_schema(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    var_6 = None
    var_7 = module_0.to_json_schema(var_4, var_6)
    var_8 = module_0.from_json_schema(var_7, var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Const'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.const).__module__}.{type(var_8.const).__qualname__}' == 'enum._EnumDict'
    assert len(var_8.const) == 0
    var_9 = var_1.clear()
    module_0.from_json_schema(var_4, var_3)

def test_case_23():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = 'minLength'
    var_6 = 'number'
    var_7 = module_4.Definitions()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_7) == 0
    var_8 = 'boolean'
    var_9 = {var_3: var_8}
    var_10 = 'null'
    var_11 = {var_3: var_10}
    var_12 = {var_0: var_9, var_2: var_11}
    var_13 = module_0.if_then_else_from_json_schema(var_12, var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.if_clause).__module__}.{type(var_13.if_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_13.then_clause).__module__}.{type(var_13.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_13.else_clause).__module__}.{type(var_13.else_clause).__qualname__}' == 'typesystem.fields.Const'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_14 = var_13.if_clause
    var_15 = var_13.else_clause
    var_16 = 'integer'
    var_17 = {var_3: var_16}
    var_18 = 'minimum'
    var_19 = 0
    var_20 = {var_3: var_16, var_18: var_19}
    var_21 = {var_0: var_17, var_1: var_20}
    var_22 = module_0.if_then_else_from_json_schema(var_21, var_7)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.if_clause).__module__}.{type(var_22.if_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_22.then_clause).__module__}.{type(var_22.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_22.else_clause).__module__}.{type(var_22.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_23 = var_22.if_clause
    var_24 = var_22.then_clause
    var_25 = 'object'
    var_26 = {var_3: var_25}
    var_27 = 'array'
    var_28 = {var_3: var_27}
    var_29 = 'minItems'
    var_30 = 2
    var_31 = {var_3: var_27, var_29: var_30}
    var_32 = {var_3: var_4}
    var_33 = {var_0: var_28, var_1: var_31, var_2: var_32}
    var_34 = {var_3: var_8}
    var_35 = {var_0: var_26, var_1: var_33, var_2: var_34}
    var_36 = module_0.if_then_else_from_json_schema(var_35, var_7)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert f'{type(var_36.if_clause).__module__}.{type(var_36.if_clause).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_36.then_clause).__module__}.{type(var_36.then_clause).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert f'{type(var_36.else_clause).__module__}.{type(var_36.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_37 = var_36.if_clause
    var_38 = var_36.then_clause
    var_39 = var_36.then_clause.if_clause
    var_40 = var_36.then_clause.then_clause
    var_41 = var_36.then_clause.else_clause
    var_42 = var_36.else_clause
    var_43 = {var_3: var_4}
    var_44 = 'default'
    var_45 = 'default_then'
    var_46 = {var_3: var_4, var_44: var_45}
    var_47 = 42
    var_48 = {var_3: var_6, var_44: var_47}
    var_49 = {var_0: var_43, var_1: var_46, var_2: var_48}
    var_50 = module_0.if_then_else_from_json_schema(var_49, var_7)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert f'{type(var_50.if_clause).__module__}.{type(var_50.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_50.then_clause).__module__}.{type(var_50.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_50.else_clause).__module__}.{type(var_50.else_clause).__qualname__}' == 'typesystem.fields.Float'
    var_51 = 'allOf'
    var_52 = {var_3: var_4}
    var_53 = 3
    var_54 = {var_5: var_53}
    var_55 = [var_52, var_54]
    var_56 = {var_51: var_55}
    var_57 = 'pattern'
    var_58 = '^[A-Z]+$'
    var_59 = {var_3: var_4, var_57: var_58}
    var_60 = '^[a-z]+$'
    var_61 = {var_3: var_4, var_57: var_60}
    var_62 = {var_0: var_56, var_1: var_59, var_2: var_61}
    var_63 = module_0.if_then_else_from_json_schema(var_62, var_7)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert f'{type(var_63.if_clause).__module__}.{type(var_63.if_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_63.then_clause).__module__}.{type(var_63.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_63.else_clause).__module__}.{type(var_63.else_clause).__qualname__}' == 'typesystem.fields.String'
    var_64 = var_63.if_clause
    var_65 = var_63.if_clause.all_of
    var_66 = len(var_65)
    assert var_66 == 2
    var_67 = var_63.then_clause
    var_68 = var_63.else_clause
    var_69 = 'All tests passed!'
    var_70 = print(var_69)

def test_case_24():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'integer'
    var_8 = {var_4: var_7}
    var_9 = {var_1: var_6, var_2: var_8, var_3: var_8}
    var_10 = module_0.if_then_else_from_json_schema(var_9, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.if_clause).__module__}.{type(var_10.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_10.then_clause).__module__}.{type(var_10.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_10.else_clause).__module__}.{type(var_10.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_11 = var_10.then_clause
    var_12 = var_10.else_clause

def test_case_25():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'minLength'
    var_7 = 5
    var_8 = {var_4: var_4, var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8, var_3: var_8}
    var_10 = module_0.if_then_else_from_json_schema(var_9, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.if_clause).__module__}.{type(var_10.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_10.then_clause).__module__}.{type(var_10.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_10.else_clause).__module__}.{type(var_10.else_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_11 = var_10.if_clause
    var_12 = var_10.then_clause
    var_13 = var_10.else_clause

def test_case_26():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 27
    var_7 = {var_3: var_4, var_4: var_6}
    var_8 = {var_1: var_5, var_2: var_7, var_4: var_7}
    var_9 = module_0.if_then_else_from_json_schema(var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.if_clause).__module__}.{type(var_9.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_9.then_clause).__module__}.{type(var_9.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_9.else_clause).__module__}.{type(var_9.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_10 = var_9.if_clause
    var_11 = var_9.then_clause

def test_case_27():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'rJ'
    var_3 = "/'U!"
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'minLength'
    var_8 = 5
    var_9 = {var_4: var_5, var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_4: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = module_0.if_then_else_from_json_schema(var_12, var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.if_clause).__module__}.{type(var_13.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_13.then_clause).__module__}.{type(var_13.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_13.else_clause).__module__}.{type(var_13.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_14 = var_13.if_clause
    var_15 = var_13.then_clause
    var_16 = var_13.else_clause

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_4.Definitions(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = None
    var_4 = module_1.OneOf(var_0, **var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.one_of).__module__}.{type(var_4.one_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_4.one_of) == 0
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_5 = module_0.from_json_schema(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    var_6 = module_0.to_json_schema(var_4, var_3)
    var_7 = module_0.from_json_schema(var_6, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.one_of == []
    var_8 = var_1.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8) == 0
    var_8.items()

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_1.OneOf(var_0, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.one_of).__module__}.{type(var_2.one_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.one_of) == 0
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_3 = module_0.from_json_schema(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_0.to_json_schema(var_2, var_1)
    module_0.enum_from_json_schema(var_0, var_1)

def test_case_30():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'maxLength'
    var_4 = 'string'
    var_5 = 5
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = False
    var_9 = module_0.from_json_schema_type(var_7, var_4, var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length == 10
    assert var_9.min_length == 5
    assert var_9.format is None
    assert var_9.coerce_types is False
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'items', 'dependencies', 'patternProperties', 'type', 'maxProperties', 'exclusiveMaximum', 'boolean_schema', 'pattern', 'required', 'additionalItems', 'maxItems', 'multipleOf', 'uniqueItems', 'minimum', 'minItems', 'additionalProperties', 'contains', 'minLength', 'minProperties', 'propertyNames', 'exclusiveMinimum', 'properties', 'maxLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = 'minimum'
    var_11 = 'maximum'
    var_12 = 'integer'
    var_13 = 100
    var_14 = {var_1: var_12, var_10: var_8, var_11: var_13}
    var_15 = module_0.from_json_schema_type(var_14, var_12, var_8, var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Integer'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is False
    assert var_15.maximum == 100
    assert var_15.exclusive_minimum is None
    assert var_15.exclusive_maximum is None
    assert var_15.multiple_of is None
    assert var_15.precision is None
    assert var_15.coerce_types is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_16 = 'exclusiveMinimum'
    var_17 = 'exclusiveMaximum'
    var_18 = 'number'
    var_19 = {var_1: var_18, var_16: var_8, var_17: var_13}
    var_20 = module_0.from_json_schema_type(var_19, var_18, var_8, var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Float'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.minimum is None
    assert var_20.maximum is None
    assert var_20.exclusive_minimum is False
    assert var_20.exclusive_maximum == 100
    assert var_20.multiple_of is None
    assert var_20.precision is None
    assert var_20.coerce_types is False
    var_21 = 'default'
    var_22 = 'boolean'
    var_23 = True
    var_24 = {var_1: var_22, var_21: var_23}
    var_25 = module_0.from_json_schema_type(var_24, var_22, var_8, var_0)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_25.default is True
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.coerce_types is False
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_26 = 'items'
    var_27 = 'minItems'
    var_28 = 'maxItems'
    var_29 = 'array'
    var_30 = {var_1: var_4}
    var_31 = {var_1: var_29, var_26: var_30, var_27: var_23, var_28: var_5}
    var_32 = module_0.from_json_schema_type(var_31, var_29, var_8, var_0)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Array'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.items).__module__}.{type(var_32.items).__qualname__}' == 'typesystem.fields.String'
    assert var_32.additional_items is True
    assert var_32.min_items is True
    assert var_32.max_items == 5
    assert var_32.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_33 = var_32.items
    var_34 = 'properties'
    var_35 = 'required'
    var_36 = 'object'
    var_37 = 'name'
    var_38 = {var_1: var_4}
    var_39 = {var_37: var_38}
    var_40 = [var_37]
    var_41 = {var_1: var_36, var_34: var_39, var_35: var_40}
    var_42 = module_0.from_json_schema_type(var_41, var_36, var_8, var_0)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.Object'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert f'{type(var_42.properties).__module__}.{type(var_42.properties).__qualname__}' == 'builtins.dict'
    assert len(var_42.properties) == 1
    assert var_42.pattern_properties == {}
    assert var_42.additional_properties is None
    assert var_42.property_names is None
    assert var_42.min_properties is None
    assert var_42.max_properties is None
    assert var_42.required == ['name']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_43 = var_42.properties[var_37]
    var_44 = 'allow_null'
    var_45 = {var_1: var_4, var_44: var_23}
    var_46 = module_0.from_json_schema_type(var_45, var_4, var_23, var_0)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.String'
    assert var_46.default is None
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is True
    assert var_46.read_only is False
    assert var_46.allow_blank is True
    assert var_46.trim_whitespace is True
    assert var_46.max_length is None
    assert var_46.min_length is None
    assert var_46.format is None
    assert var_46.coerce_types is False
    assert var_46.pattern is None
    assert var_46.pattern_regex is None
    var_47 = 'pattern'
    var_48 = '^[a-z]+$'
    var_49 = {var_1: var_4, var_47: var_48}
    var_50 = module_0.from_json_schema_type(var_49, var_4, var_8, var_0)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.fields.String'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert var_50.default == ''
    assert var_50.allow_blank is True
    assert var_50.trim_whitespace is True
    assert var_50.max_length is None
    assert var_50.min_length is None
    assert var_50.format is None
    assert var_50.coerce_types is False
    assert var_50.pattern == '^[a-z]+$'
    assert f'{type(var_50.pattern_regex).__module__}.{type(var_50.pattern_regex).__qualname__}' == 're.Pattern'
    var_51 = 'additionalItems'
    var_52 = {var_1: var_4}
    var_53 = {var_1: var_12}
    var_54 = [var_52, var_53]
    var_55 = {var_1: var_29, var_26: var_54, var_51: var_8}
    var_56 = module_0.from_json_schema_type(var_55, var_29, var_8, var_0)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Array'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.items).__module__}.{type(var_56.items).__qualname__}' == 'builtins.list'
    assert len(var_56.items) == 2
    assert var_56.additional_items is False
    assert var_56.min_items == 0
    assert var_56.max_items == 2
    assert var_56.unique_items is False
    var_57 = var_56.items
    var_58 = var_56.items
    var_59 = len(var_58)
    assert var_59 == 2
    var_60 = var_56.items[var_8]
    var_61 = var_56.items[var_23]
    var_62 = 'patternProperties'
    var_63 = {var_1: var_4}
    var_64 = {var_48: var_63}
    var_65 = {var_1: var_36, var_62: var_64}
    var_66 = module_0.from_json_schema_type(var_65, var_36, var_8, var_0)
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.fields.Object'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert var_66.properties == {}
    assert f'{type(var_66.pattern_properties).__module__}.{type(var_66.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_66.pattern_properties) == 1
    assert var_66.additional_properties is None
    assert var_66.property_names is None
    assert var_66.min_properties is None
    assert var_66.max_properties is None
    assert var_66.required == []
    var_67 = var_66.pattern_properties[var_48]
    var_68 = 'propertyNames'
    var_69 = {var_47: var_48}
    var_70 = {var_1: var_36, var_68: var_69}
    var_71 = module_0.from_json_schema_type(var_70, var_36, var_8, var_0)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Object'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.properties == {}
    assert var_71.pattern_properties == {}
    assert var_71.additional_properties is None
    assert f'{type(var_71.property_names).__module__}.{type(var_71.property_names).__qualname__}' == 'typesystem.fields.Union'
    assert var_71.min_properties is None
    assert var_71.max_properties is None
    assert var_71.required == []
    var_72 = var_71.property_names
    var_73 = 'additionalProperties'
    var_74 = {var_1: var_36, var_73: var_8}
    var_75 = module_0.from_json_schema_type(var_74, var_36, var_8, var_0)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.fields.Object'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert var_75.properties == {}
    assert var_75.pattern_properties == {}
    assert var_75.additional_properties is False
    assert var_75.property_names is None
    assert var_75.min_properties is None
    assert var_75.max_properties is None
    assert var_75.required == []
    var_76 = {var_1: var_4}
    var_77 = {var_1: var_36, var_73: var_76}
    var_78 = module_0.from_json_schema_type(var_77, var_36, var_8, var_0)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.Object'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert var_78.properties == {}
    assert var_78.pattern_properties == {}
    assert f'{type(var_78.additional_properties).__module__}.{type(var_78.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_78.property_names is None
    assert var_78.min_properties is None
    assert var_78.max_properties is None
    assert var_78.required == []
    var_79 = var_78.additional_properties
    var_80 = 'uniqueItems'
    var_81 = {var_1: var_29, var_80: var_23}
    var_82 = module_0.from_json_schema_type(var_81, var_29, var_8, var_0)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.Array'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert var_82.items is None
    assert var_82.additional_items is True
    assert var_82.min_items == 0
    assert var_82.max_items is None
    assert var_82.unique_items is True
    var_83 = 'format'
    var_84 = 'email'
    var_85 = {var_1: var_4, var_83: var_84}
    var_86 = module_0.from_json_schema_type(var_85, var_4, var_8, var_0)
    assert f'{type(var_86).__module__}.{type(var_86).__qualname__}' == 'typesystem.fields.String'
    assert var_86.title == ''
    assert var_86.description == ''
    assert var_86.allow_null is False
    assert var_86.read_only is False
    assert var_86.default == ''
    assert var_86.allow_blank is True
    assert var_86.trim_whitespace is True
    assert var_86.max_length is None
    assert var_86.min_length is None
    assert var_86.format == 'email'
    assert var_86.coerce_types is False
    assert var_86.pattern is None
    assert var_86.pattern_regex is None
    var_87 = 'multipleOf'
    var_88 = 2
    var_89 = {var_1: var_12, var_87: var_88}
    var_90 = module_0.from_json_schema_type(var_89, var_12, var_8, var_0)
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'typesystem.fields.Integer'
    assert var_90.title == ''
    assert var_90.description == ''
    assert var_90.allow_null is False
    assert var_90.read_only is False
    assert var_90.minimum is None
    assert var_90.maximum is None
    assert var_90.exclusive_minimum is None
    assert var_90.exclusive_maximum is None
    assert var_90.multiple_of == 2
    assert var_90.precision is None
    assert var_90.coerce_types is False
    var_91 = 0.5
    var_92 = {var_1: var_18, var_87: var_91}
    var_93 = module_0.from_json_schema_type(var_92, var_18, var_8, var_0)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.Float'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert var_93.minimum is None
    assert var_93.maximum is None
    assert var_93.exclusive_minimum is None
    assert var_93.exclusive_maximum is None
    assert var_93.multiple_of == pytest.approx(0.5, abs=0.01, rel=0.01)
    assert var_93.precision is None
    assert var_93.coerce_types is False
    var_94 = 'hello'
    var_95 = {var_1: var_4, var_21: var_94}
    var_96 = module_0.from_json_schema_type(var_95, var_4, var_8, var_0)
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.fields.String'
    assert var_96.default == 'hello'
    assert var_96.title == ''
    assert var_96.description == ''
    assert var_96.allow_null is False
    assert var_96.read_only is False
    assert var_96.allow_blank is True
    assert var_96.trim_whitespace is True
    assert var_96.max_length is None
    assert var_96.min_length is None
    assert var_96.format is None
    assert var_96.coerce_types is False
    assert var_96.pattern is None
    assert var_96.pattern_regex is None
    var_97 = 42
    var_98 = {var_1: var_12, var_21: var_97}
    var_99 = module_0.from_json_schema_type(var_98, var_12, var_8, var_0)
    assert f'{type(var_99).__module__}.{type(var_99).__qualname__}' == 'typesystem.fields.Integer'
    assert var_99.default == 42
    assert var_99.title == ''
    assert var_99.description == ''
    assert var_99.allow_null is False
    assert var_99.read_only is False
    assert var_99.minimum is None
    assert var_99.maximum is None
    assert var_99.exclusive_minimum is None
    assert var_99.exclusive_maximum is None
    assert var_99.multiple_of is None
    assert var_99.precision is None
    assert var_99.coerce_types is False
    var_100 = {var_1: var_22, var_21: var_8}
    var_101 = module_0.from_json_schema_type(var_100, var_22, var_8, var_0)
    assert f'{type(var_101).__module__}.{type(var_101).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_101.default is False
    assert var_101.title == ''
    assert var_101.description == ''
    assert var_101.allow_null is False
    assert var_101.read_only is False
    assert var_101.coerce_types is False
    var_102 = []
    var_103 = {var_1: var_29, var_21: var_102}
    var_104 = module_0.from_json_schema_type(var_103, var_29, var_8, var_0)
    assert f'{type(var_104).__module__}.{type(var_104).__qualname__}' == 'typesystem.fields.Array'
    assert var_104.default == []
    assert var_104.title == ''
    assert var_104.description == ''
    assert var_104.allow_null is False
    assert var_104.read_only is False
    assert var_104.items is None
    assert var_104.additional_items is True
    assert var_104.min_items == 0
    assert var_104.max_items is None
    assert var_104.unique_items is False
    var_105 = {}
    var_106 = {var_1: var_36, var_21: var_105}
    var_107 = module_0.from_json_schema_type(var_106, var_36, var_8, var_0)
    assert f'{type(var_107).__module__}.{type(var_107).__qualname__}' == 'typesystem.fields.Object'
    assert var_107.default == {}
    assert var_107.title == ''
    assert var_107.description == ''
    assert var_107.allow_null is False
    assert var_107.read_only is False
    assert var_107.properties == {}
    assert var_107.pattern_properties == {}
    assert var_107.additional_properties is None
    assert var_107.property_names is None
    assert var_107.min_properties is None
    assert var_107.max_properties is None
    assert var_107.required == []
    var_108 = 'allow_blank'
    var_109 = {var_1: var_4, var_108: var_23}
    var_110 = module_0.from_json_schema_type(var_109, var_4, var_8, var_0)
    assert f'{type(var_110).__module__}.{type(var_110).__qualname__}' == 'typesystem.fields.String'
    assert var_110.title == ''
    assert var_110.description == ''
    assert var_110.allow_null is False
    assert var_110.read_only is False
    assert var_110.default == ''
    assert var_110.allow_blank is True
    assert var_110.trim_whitespace is True
    assert var_110.max_length is None
    assert var_110.min_length is None
    assert var_110.format is None
    assert var_110.coerce_types is False
    assert var_110.pattern is None
    assert var_110.pattern_regex is None
    var_111 = {var_1: var_4, var_108: var_8}
    var_112 = module_0.from_json_schema_type(var_111, var_4, var_8, var_0)
    assert f'{type(var_112).__module__}.{type(var_112).__qualname__}' == 'typesystem.fields.String'
    assert var_112.title == ''
    assert var_112.description == ''
    assert var_112.allow_null is False
    assert var_112.read_only is False
    assert var_112.default == ''
    assert var_112.allow_blank is True
    assert var_112.trim_whitespace is True
    assert var_112.max_length is None
    assert var_112.min_length is None
    assert var_112.format is None
    assert var_112.coerce_types is False
    assert var_112.pattern is None
    assert var_112.pattern_regex is None
    var_113 = 'min_length'
    var_114 = {var_1: var_4, var_113: var_8}
    var_115 = module_0.from_json_schema_type(var_114, var_4, var_8, var_0)
    assert f'{type(var_115).__module__}.{type(var_115).__qualname__}' == 'typesystem.fields.String'
    assert var_115.title == ''
    assert var_115.description == ''
    assert var_115.allow_null is False
    assert var_115.read_only is False
    assert var_115.default == ''
    assert var_115.allow_blank is True
    assert var_115.trim_whitespace is True
    assert var_115.max_length is None
    assert var_115.min_length is None
    assert var_115.format is None
    assert var_115.coerce_types is False
    assert var_115.pattern is None
    assert var_115.pattern_regex is None
    var_116 = {var_1: var_4, var_113: var_23}
    var_117 = module_0.from_json_schema_type(var_116, var_4, var_8, var_0)
    assert f'{type(var_117).__module__}.{type(var_117).__qualname__}' == 'typesystem.fields.String'
    assert var_117.title == ''
    assert var_117.description == ''
    assert var_117.allow_null is False
    assert var_117.read_only is False
    assert var_117.default == ''
    assert var_117.allow_blank is True
    assert var_117.trim_whitespace is True
    assert var_117.max_length is None
    assert var_117.min_length is None
    assert var_117.format is None
    assert var_117.coerce_types is False
    assert var_117.pattern is None
    assert var_117.pattern_regex is None
    var_118 = 'coerce_types'
    var_119 = {var_1: var_4, var_118: var_8}
    var_120 = module_0.from_json_schema_type(var_119, var_4, var_8, var_0)
    assert f'{type(var_120).__module__}.{type(var_120).__qualname__}' == 'typesystem.fields.String'
    assert var_120.title == ''
    assert var_120.description == ''
    assert var_120.allow_null is False
    assert var_120.read_only is False
    assert var_120.default == ''
    assert var_120.allow_blank is True
    assert var_120.trim_whitespace is True
    assert var_120.max_length is None
    assert var_120.min_length is None
    assert var_120.format is None
    assert var_120.coerce_types is False
    assert var_120.pattern is None
    assert var_120.pattern_regex is None
    var_121 = {var_1: var_12, var_118: var_8}
    var_122 = module_0.from_json_schema_type(var_121, var_12, var_8, var_0)
    assert f'{type(var_122).__module__}.{type(var_122).__qualname__}' == 'typesystem.fields.Integer'
    assert var_122.title == ''
    assert var_122.description == ''
    assert var_122.allow_null is False
    assert var_122.read_only is False
    assert var_122.minimum is None
    assert var_122.maximum is None
    assert var_122.exclusive_minimum is None
    assert var_122.exclusive_maximum is None
    assert var_122.multiple_of is None
    assert var_122.precision is None
    assert var_122.coerce_types is False
    var_123 = {var_1: var_18, var_118: var_8}
    var_124 = module_0.from_json_schema_type(var_123, var_18, var_8, var_0)
    assert f'{type(var_124).__module__}.{type(var_124).__qualname__}' == 'typesystem.fields.Float'
    assert var_124.title == ''
    assert var_124.description == ''
    assert var_124.allow_null is False
    assert var_124.read_only is False
    assert var_124.minimum is None
    assert var_124.maximum is None
    assert var_124.exclusive_minimum is None
    assert var_124.exclusive_maximum is None
    assert var_124.multiple_of is None
    assert var_124.precision is None
    assert var_124.coerce_types is False
    var_125 = {var_1: var_22, var_118: var_8}
    var_126 = module_0.from_json_schema_type(var_125, var_22, var_8, var_0)
    assert f'{type(var_126).__module__}.{type(var_126).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_126.title == ''
    assert var_126.description == ''
    assert var_126.allow_null is False
    assert var_126.read_only is False
    assert var_126.coerce_types is False
    var_127 = {var_1: var_29, var_118: var_8}
    var_128 = module_0.from_json_schema_type(var_127, var_29, var_8, var_0)
    assert f'{type(var_128).__module__}.{type(var_128).__qualname__}' == 'typesystem.fields.Array'
    assert var_128.title == ''
    assert var_128.description == ''
    assert var_128.allow_null is False
    assert var_128.read_only is False
    assert var_128.items is None
    assert var_128.additional_items is True
    assert var_128.min_items == 0
    assert var_128.max_items is None
    assert var_128.unique_items is False
    var_129 = {var_1: var_36, var_118: var_8}
    var_130 = module_0.from_json_schema_type(var_129, var_36, var_8, var_0)
    assert f'{type(var_130).__module__}.{type(var_130).__qualname__}' == 'typesystem.fields.Object'
    assert var_130.title == ''
    assert var_130.description == ''
    assert var_130.allow_null is False
    assert var_130.read_only is False
    assert var_130.properties == {}
    assert var_130.pattern_properties == {}
    assert var_130.additional_properties is None
    assert var_130.property_names is None
    assert var_130.min_properties is None
    assert var_130.max_properties is None
    assert var_130.required == []
    var_131 = 'null'
    var_132 = {var_1: var_131}
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_132, var_131, var_23, var_0)