# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import typesystem.composites as module_1
import enum as module_2
import typesystem.fields as module_3
import typesystem.schemas as module_4
import re as module_5
import typesystem.base as module_6

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_0 = {}
    var_1 = None
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_0, var_1, var_1, var_1)

def test_case_5():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
def test_case_6():
    var_0 = None
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_3.Float()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_2 = None
    var_3 = module_0.type_from_json_schema(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_4 = module_0.to_json_schema(var_0)
    var_5 = module_3.Choice()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Choice'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.choices == []
    assert var_5.coerce_types is True
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_6 = module_0.to_json_schema(var_3)
    var_7 = module_0.to_json_schema(var_1, var_4)
    module_0.all_of_from_json_schema(var_6, var_5)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = var_0.values()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_1) == 0
    var_2 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.type_from_json_schema(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = module_0.from_json_schema(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_5 = module_0.from_json_schema(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    var_6 = module_0.to_json_schema(var_3)
    var_7 = var_0.values()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_7) == 0
    module_0.not_from_json_schema(var_7, var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
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
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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

def test_case_12():
    var_0 = module_3.Float()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
def test_case_13():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

def test_case_14():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.type_from_json_schema(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = module_0.from_json_schema(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_5 = module_0.to_json_schema(var_3)
    var_6 = module_0.to_json_schema(var_4)
    assert var_6 is True

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    module_0.any_of_from_json_schema(var_0, var_0)

def test_case_16():
    var_0 = module_3.Float()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    assert var_2.coerce_types is False

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_3.Float()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = None
    var_2 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.type_from_json_schema(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Float'
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
    assert var_3.coerce_types is False
    var_4 = 'B~ODu4z25,AO'
    var_5 = module_0.from_json_schema(var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    module_0.from_json_schema(var_1)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = False
    var_3 = module_3.Integer(multiple_of=var_1, coerce_types=var_2)
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
    assert var_3.coerce_types is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_0.type_from_json_schema(var_4, var_0)
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
    assert var_5.coerce_types is False
    module_5.subn(var_1, var_1, var_1)

def test_case_19():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'default'
    var_5 = 0
    var_6 = 100
    var_7 = 0.5
    var_8 = 99.5
    var_9 = 5
    var_10 = 50
    var_11 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_2: var_9, var_4: var_10}
    var_12 = 'number'
    var_13 = False
    var_14 = module_4.Definitions()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_14) == 0
    var_15 = module_0.from_json_schema_type(var_11, var_12, var_13, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Float'
    assert var_15.default == 50
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum == 0
    assert var_15.maximum == 100
    assert var_15.exclusive_minimum == 5
    assert var_15.exclusive_maximum == pytest.approx(99.5, abs=0.01, rel=0.01)
    assert var_15.multiple_of is None
    assert var_15.precision is None
    assert var_15.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_16 = 10
    var_17 = {var_4: var_16}
    var_18 = True
    var_19 = module_4.Definitions()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_19) == 0
    var_20 = module_0.from_json_schema_type(var_17, var_12, var_18, var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Float'
    assert var_20.default == 10
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is True
    assert var_20.read_only is False
    assert var_20.minimum is None
    assert var_20.maximum is None
    assert var_20.exclusive_minimum is None
    assert var_20.exclusive_maximum is None
    assert var_20.multiple_of is None
    assert var_20.precision is None
    assert var_20.coerce_types is False
    var_21 = 11
    var_22 = 2
    var_23 = {var_0: var_18, var_1: var_16, var_2: var_13, var_3: var_21, var_4: var_22, var_4: var_9}
    var_24 = 'integer'
    var_25 = False
    var_26 = module_4.Definitions()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_26) == 0
    var_27 = module_0.from_json_schema_type(var_23, var_24, var_25, var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Integer'
    assert var_27.default == 5
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.minimum is True
    assert var_27.maximum == 10
    assert var_27.exclusive_minimum is False
    assert var_27.exclusive_maximum == 11
    assert var_27.multiple_of is None
    assert var_27.precision is None
    assert var_27.coerce_types is False
    var_28 = 'minLength'
    var_29 = 'maxLength'
    var_30 = 'C2Gc0zX\x0bA[B@'
    var_31 = 'format'
    var_32 = '^[a-z]+$'
    var_33 = 'email'
    var_34 = 'test'
    var_35 = {var_28: var_22, var_29: var_10, var_30: var_32, var_31: var_33, var_4: var_34}
    var_36 = 'string'
    var_37 = False
    var_38 = module_4.Definitions()
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_38) == 0
    var_39 = module_0.from_json_schema_type(var_35, var_36, var_37, var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.String'
    assert var_39.default == 'test'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.allow_blank is False
    assert var_39.trim_whitespace is True
    assert var_39.max_length == 50
    assert var_39.min_length == 2
    assert var_39.format == 'email'
    assert var_39.coerce_types is False
    assert var_39.pattern is None
    assert var_39.pattern_regex is None
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_40 = {var_28: var_37}
    var_41 = False
    var_42 = module_4.Definitions()
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_42) == 0
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_43 = module_0.from_json_schema_type(var_40, var_36, var_41, var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.String'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.default == ''
    assert var_43.allow_blank is True
    assert var_43.trim_whitespace is True
    assert var_43.max_length is None
    assert var_43.min_length is None
    assert var_43.format is None
    assert var_43.coerce_types is False
    assert var_43.pattern is None
    assert var_43.pattern_regex is None
    var_44 = {var_28: var_18}
    var_45 = False
    var_46 = module_4.Definitions()
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_46) == 0
    var_47 = module_0.from_json_schema_type(var_44, var_36, var_45, var_46)
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
    assert var_47.coerce_types is False
    assert var_47.pattern is None
    assert var_47.pattern_regex is None
    var_48 = {var_4: var_18}
    var_49 = 'boolean'
    var_50 = False
    var_51 = module_4.Definitions()
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_51) == 0
    var_52 = module_0.from_json_schema_type(var_48, var_49, var_50, var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_52.default is True
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert var_52.coerce_types is False
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_53 = {}
    var_54 = module_4.Definitions()
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_54) == 0
    var_55 = module_0.from_json_schema_type(var_53, var_49, var_18, var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_55.default is None
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is True
    assert var_55.read_only is False
    assert var_55.coerce_types is False
    var_56 = 'minItems'
    var_57 = 'maxItems'
    var_58 = 'uniqueItems'
    var_59 = {var_56: var_50, var_57: var_16, var_58: var_18}
    var_60 = 'array'
    var_61 = module_4.Definitions()
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_61) == 0
    var_62 = module_0.from_json_schema_type(var_59, var_60, var_25, var_61)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.fields.Array'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert var_62.items is None
    assert var_62.additional_items is True
    assert var_62.min_items is False
    assert var_62.max_items == 10
    assert var_62.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_63 = 'items'
    var_64 = 'type'
    var_65 = {var_64: var_36}
    var_66 = {var_63: var_65, var_56: var_18}
    var_67 = False
    var_68 = module_4.Definitions()
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_68) == 0
    var_69 = module_0.from_json_schema_type(var_66, var_60, var_67, var_68)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.fields.Array'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert f'{type(var_69.items).__module__}.{type(var_69.items).__qualname__}' == 'typesystem.fields.String'
    assert var_69.additional_items is True
    assert var_69.min_items is True
    assert var_69.max_items is None
    assert var_69.unique_items is False
    var_70 = var_69.items
    var_71 = {var_64: var_36}
    var_72 = {var_64: var_12}
    var_73 = [var_71, var_72]
    var_74 = {var_63: var_73}
    var_75 = False
    var_76 = module_4.Definitions()
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_76) == 0
    var_77 = module_0.from_json_schema_type(var_74, var_60, var_75, var_76)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.Array'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert f'{type(var_77.items).__module__}.{type(var_77.items).__qualname__}' == 'builtins.list'
    assert len(var_77.items) == 2
    assert var_77.additional_items is True
    assert var_77.min_items == 0
    assert var_77.max_items is None
    assert var_77.unique_items is False
    var_78 = var_77.items
    var_79 = var_77.items
    var_80 = len(var_79)
    assert var_80 == 2
    var_81 = 'additionalItems'
    var_82 = False
    var_83 = {var_81: var_82}
    var_84 = False
    var_85 = module_4.Definitions()
    assert f'{type(var_85).__module__}.{type(var_85).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_85) == 0
    var_86 = module_0.from_json_schema_type(var_83, var_60, var_84, var_85)
    assert f'{type(var_86).__module__}.{type(var_86).__qualname__}' == 'typesystem.fields.Array'
    assert var_86.title == ''
    assert var_86.description == ''
    assert var_86.allow_null is False
    assert var_86.read_only is False
    assert var_86.items is None
    assert var_86.additional_items is False
    assert var_86.min_items == 0
    assert var_86.max_items is None
    assert var_86.unique_items is False
    var_87 = {var_64: var_36}
    var_88 = {var_81: var_87}
    var_89 = False
    var_90 = module_4.Definitions()
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_90) == 0
    var_91 = module_0.from_json_schema_type(var_88, var_60, var_89, var_90)
    assert f'{type(var_91).__module__}.{type(var_91).__qualname__}' == 'typesystem.fields.Array'
    assert var_91.title == ''
    assert var_91.description == ''
    assert var_91.allow_null is False
    assert var_91.read_only is False
    assert var_91.items is None
    assert f'{type(var_91.additional_items).__module__}.{type(var_91.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_91.min_items == 0
    assert var_91.max_items is None
    assert var_91.unique_items is False
    var_92 = var_91.additional_items
    var_93 = {}
    var_94 = 'object'
    var_95 = False
    var_96 = module_4.Definitions()
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_96) == 0
    var_97 = module_0.from_json_schema_type(var_93, var_94, var_95, var_96)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.fields.Object'
    assert var_97.title == ''
    assert var_97.description == ''
    assert var_97.allow_null is False
    assert var_97.read_only is False
    assert var_97.properties == {}
    assert var_97.pattern_properties == {}
    assert var_97.additional_properties is None
    assert var_97.property_names is None
    assert var_97.min_properties is None
    assert var_97.max_properties is None
    assert var_97.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    with pytest.raises(module_6.ValidationError):
        var_70.validate(var_70)

def test_case_20():
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
    assert f'{type(var_12.then_clause).__module__}.{type(var_12.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_12.else_clause).__module__}.{type(var_12.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_13 = {var_4: var_5}
    var_14 = {var_4: var_7}
    var_15 = {var_1: var_13, var_2: var_14}
    var_16 = module_0.if_then_else_from_json_schema(var_15, var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert f'{type(var_16.if_clause).__module__}.{type(var_16.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_16.then_clause).__module__}.{type(var_16.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_16.else_clause).__module__}.{type(var_16.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_17 = {var_4: var_5}
    var_18 = {var_4: var_9}
    var_19 = {var_1: var_17, var_3: var_18}
    var_20 = module_0.if_then_else_from_json_schema(var_19, var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.if_clause).__module__}.{type(var_20.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_20.then_clause).__module__}.{type(var_20.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_20.else_clause).__module__}.{type(var_20.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_21 = {var_4: var_5}
    var_22 = {var_1: var_21}
    var_23 = module_0.if_then_else_from_json_schema(var_22, var_0)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.if_clause).__module__}.{type(var_23.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_23.then_clause).__module__}.{type(var_23.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_23.else_clause).__module__}.{type(var_23.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_24 = 'default'
    var_25 = {var_4: var_5}
    var_26 = {var_4: var_7}
    var_27 = 42
    var_28 = {var_1: var_25, var_2: var_26, var_24: var_27}
    var_29 = module_0.if_then_else_from_json_schema(var_28, var_0)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_29.default == 42
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.if_clause).__module__}.{type(var_29.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_29.then_clause).__module__}.{type(var_29.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_29.else_clause).__module__}.{type(var_29.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_30 = 'properties'
    var_31 = 'object'
    var_32 = 'name'
    var_33 = {var_4: var_5}
    var_34 = {var_32: var_33}
    var_35 = {var_4: var_31, var_30: var_34}
    var_36 = 'items'
    var_37 = 'array'
    var_38 = {var_4: var_7}
    var_39 = {var_4: var_37, var_36: var_38}
    var_40 = 'enum'
    var_41 = None
    var_42 = 'unknown'
    var_43 = [var_41, var_42]
    var_44 = {var_40: var_43}
    var_45 = {var_1: var_35, var_2: var_39, var_3: var_44}
    var_46 = module_0.if_then_else_from_json_schema(var_45, var_0)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.if_clause).__module__}.{type(var_46.if_clause).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_46.then_clause).__module__}.{type(var_46.then_clause).__qualname__}' == 'typesystem.fields.Array'
    assert f'{type(var_46.else_clause).__module__}.{type(var_46.else_clause).__qualname__}' == 'typesystem.fields.Choice'

def test_case_21():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 'default'
    var_6 = 0
    var_7 = 100
    var_8 = 0.5
    var_9 = 99.5
    var_10 = 5
    var_11 = 50
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = 'number'
    var_14 = False
    var_15 = module_4.Definitions()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_15) == 0
    var_16 = module_0.from_json_schema_type(var_12, var_13, var_14, var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Float'
    assert var_16.default == 50
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum == 0
    assert var_16.maximum == 100
    assert var_16.exclusive_minimum == pytest.approx(0.5, abs=0.01, rel=0.01)
    assert var_16.exclusive_maximum == pytest.approx(99.5, abs=0.01, rel=0.01)
    assert var_16.multiple_of == 5
    assert var_16.precision is None
    assert var_16.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_17 = 10
    var_18 = {var_5: var_17}
    var_19 = True
    var_20 = module_4.Definitions()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_20) == 0
    var_21 = module_0.from_json_schema_type(var_18, var_13, var_19, var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Float'
    assert var_21.default == 10
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is True
    assert var_21.read_only is False
    assert var_21.minimum is None
    assert var_21.maximum is None
    assert var_21.exclusive_minimum is None
    assert var_21.exclusive_maximum is None
    assert var_21.multiple_of is None
    assert var_21.precision is None
    assert var_21.coerce_types is False
    var_22 = 11
    var_23 = 2
    var_24 = {var_0: var_19, var_1: var_17, var_2: var_14, var_3: var_22, var_4: var_23, var_5: var_10}
    var_25 = 'integer'
    var_26 = False
    var_27 = module_4.Definitions()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_27) == 0
    var_28 = module_0.from_json_schema_type(var_24, var_25, var_26, var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Integer'
    assert var_28.default == 5
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.minimum is True
    assert var_28.maximum == 10
    assert var_28.exclusive_minimum is False
    assert var_28.exclusive_maximum == 11
    assert var_28.multiple_of == 2
    assert var_28.precision is None
    assert var_28.coerce_types is False
    var_29 = 'minLength'
    var_30 = 'maxLength'
    var_31 = 'pattern'
    var_32 = 'format'
    var_33 = '^[a-z]+$'
    var_34 = 'email'
    var_35 = 'test'
    var_36 = {var_29: var_23, var_30: var_11, var_31: var_33, var_32: var_34, var_5: var_35}
    var_37 = 'string'
    var_38 = False
    var_39 = module_4.Definitions()
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_39) == 0
    var_40 = module_0.from_json_schema_type(var_36, var_37, var_38, var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.String'
    assert var_40.default == 'test'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert var_40.allow_blank is False
    assert var_40.trim_whitespace is True
    assert var_40.max_length == 50
    assert var_40.min_length == 2
    assert var_40.format == 'email'
    assert var_40.coerce_types is False
    assert var_40.pattern == '^[a-z]+$'
    assert f'{type(var_40.pattern_regex).__module__}.{type(var_40.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_41 = {var_29: var_38}
    var_42 = False
    var_43 = module_4.Definitions()
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_43) == 0
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_44 = module_0.from_json_schema_type(var_41, var_37, var_42, var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.String'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert var_44.default == ''
    assert var_44.allow_blank is True
    assert var_44.trim_whitespace is True
    assert var_44.max_length is None
    assert var_44.min_length is None
    assert var_44.format is None
    assert var_44.coerce_types is False
    assert var_44.pattern is None
    assert var_44.pattern_regex is None
    var_45 = {var_29: var_19}
    var_46 = False
    var_47 = module_4.Definitions()
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_47) == 0
    var_48 = module_0.from_json_schema_type(var_45, var_37, var_46, var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.String'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert var_48.allow_blank is False
    assert var_48.trim_whitespace is True
    assert var_48.max_length is None
    assert var_48.min_length is None
    assert var_48.format is None
    assert var_48.coerce_types is False
    assert var_48.pattern is None
    assert var_48.pattern_regex is None
    var_49 = {var_5: var_19}
    var_50 = 'boolean'
    var_51 = False
    var_52 = module_4.Definitions()
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_52) == 0
    var_53 = module_0.from_json_schema_type(var_49, var_50, var_51, var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_53.default is True
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert var_53.coerce_types is False
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_54 = {}
    var_55 = module_4.Definitions()
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_55) == 0
    var_56 = module_0.from_json_schema_type(var_54, var_50, var_19, var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_56.default is None
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is True
    assert var_56.read_only is False
    assert var_56.coerce_types is False
    var_57 = 'minItems'
    var_58 = 'maxItems'
    var_59 = 'uniqueItems'
    var_60 = {var_57: var_51, var_58: var_17, var_59: var_19}
    var_61 = 'array'
    var_62 = False
    var_63 = module_4.Definitions()
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_63) == 0
    var_64 = module_0.from_json_schema_type(var_60, var_61, var_62, var_63)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.Array'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert var_64.items is None
    assert var_64.additional_items is True
    assert var_64.min_items is False
    assert var_64.max_items == 10
    assert var_64.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_65 = 'items'
    var_66 = 'type'
    var_67 = {var_66: var_37}
    var_68 = {var_65: var_67, var_57: var_19}
    var_69 = False
    var_70 = module_4.Definitions()
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_70) == 0
    var_71 = module_0.from_json_schema_type(var_68, var_61, var_69, var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Array'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert f'{type(var_71.items).__module__}.{type(var_71.items).__qualname__}' == 'typesystem.fields.String'
    assert var_71.additional_items is True
    assert var_71.min_items is True
    assert var_71.max_items is None
    assert var_71.unique_items is False
    var_72 = var_71.items
    var_73 = {var_66: var_37}
    var_74 = {var_66: var_13}
    var_75 = [var_73, var_74]
    var_76 = {var_65: var_75}
    var_77 = False
    var_78 = module_4.Definitions()
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_78) == 0
    var_79 = module_0.from_json_schema_type(var_76, var_61, var_77, var_78)
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.fields.Array'
    assert var_79.title == ''
    assert var_79.description == ''
    assert var_79.allow_null is False
    assert var_79.read_only is False
    assert f'{type(var_79.items).__module__}.{type(var_79.items).__qualname__}' == 'builtins.list'
    assert len(var_79.items) == 2
    assert var_79.additional_items is True
    assert var_79.min_items == 0
    assert var_79.max_items is None
    assert var_79.unique_items is False
    var_80 = var_79.items
    var_81 = var_79.items
    var_82 = len(var_81)
    assert var_82 == 2
    var_83 = 'additionalItems'
    var_84 = False
    var_85 = {var_83: var_84}
    var_86 = False
    var_87 = module_4.Definitions()
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_87) == 0
    var_88 = module_0.from_json_schema_type(var_85, var_61, var_86, var_87)
    assert f'{type(var_88).__module__}.{type(var_88).__qualname__}' == 'typesystem.fields.Array'
    assert var_88.title == ''
    assert var_88.description == ''
    assert var_88.allow_null is False
    assert var_88.read_only is False
    assert var_88.items is None
    assert var_88.additional_items is False
    assert var_88.min_items == 0
    assert var_88.max_items is None
    assert var_88.unique_items is False
    var_89 = {var_66: var_37}
    var_90 = {var_83: var_89}
    var_91 = False
    var_92 = module_4.Definitions()
    assert f'{type(var_92).__module__}.{type(var_92).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_92) == 0
    var_93 = module_0.from_json_schema_type(var_90, var_61, var_91, var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.Array'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert var_93.items is None
    assert f'{type(var_93.additional_items).__module__}.{type(var_93.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_93.min_items == 0
    assert var_93.max_items is None
    assert var_93.unique_items is False
    var_94 = var_93.additional_items
    var_95 = {}
    var_96 = 'object'
    var_97 = False
    var_98 = module_0.from_json_schema_type(var_95, var_96, var_97, var_70)
    assert f'{type(var_98).__module__}.{type(var_98).__qualname__}' == 'typesystem.fields.Object'
    assert var_98.title == ''
    assert var_98.description == ''
    assert var_98.allow_null is False
    assert var_98.read_only is False
    assert var_98.properties == {}
    assert var_98.pattern_properties == {}
    assert var_98.additional_properties is None
    assert var_98.property_names is None
    assert var_98.min_properties is None
    assert var_98.max_properties is None
    assert var_98.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_99 = 'properties'
    var_100 = 'name'
    var_101 = 'age'
    var_102 = {var_66: var_37}
    var_103 = {var_66: var_25}
    var_104 = {var_100: var_102, var_101: var_103}
    var_105 = {var_99: var_104}
    var_106 = False
    var_107 = module_4.Definitions()
    assert f'{type(var_107).__module__}.{type(var_107).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_107) == 0
    var_108 = module_0.from_json_schema_type(var_105, var_96, var_106, var_107)
    assert f'{type(var_108).__module__}.{type(var_108).__qualname__}' == 'typesystem.fields.Object'
    assert var_108.title == ''
    assert var_108.description == ''
    assert var_108.allow_null is False
    assert var_108.read_only is False
    assert f'{type(var_108.properties).__module__}.{type(var_108.properties).__qualname__}' == 'builtins.dict'
    assert len(var_108.properties) == 2
    assert var_108.pattern_properties == {}
    assert var_108.additional_properties is None
    assert var_108.property_names is None
    assert var_108.min_properties is None
    assert var_108.max_properties is None
    assert var_108.required == []
    var_109 = var_108.properties
    var_110 = 'patternProperties'
    var_111 = '^S_'
    var_112 = '^I_'
    var_113 = {var_66: var_37}
    var_114 = {var_66: var_25}
    var_115 = {var_111: var_113, var_112: var_114}
    var_116 = {var_110: var_115}
    var_117 = False
    var_118 = module_4.Definitions()
    assert f'{type(var_118).__module__}.{type(var_118).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_118) == 0
    var_119 = module_0.from_json_schema_type(var_116, var_96, var_117, var_118)
    assert f'{type(var_119).__module__}.{type(var_119).__qualname__}' == 'typesystem.fields.Object'
    assert var_119.title == ''
    assert var_119.description == ''
    assert var_119.allow_null is False
    assert var_119.read_only is False
    assert var_119.properties == {}
    assert f'{type(var_119.pattern_properties).__module__}.{type(var_119.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_119.pattern_properties) == 2
    assert var_119.additional_properties is None
    assert var_119.property_names is None
    assert var_119.min_properties is None
    assert var_119.max_properties is None
    assert var_119.required == []
    var_120 = var_119.pattern_properties
    var_121 = 'additionalProperties'
    var_122 = False
    var_123 = {var_121: var_122}
    var_124 = False
    var_125 = module_4.Definitions()
    assert f'{type(var_125).__module__}.{type(var_125).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_125) == 0
    var_126 = module_0.from_json_schema_type(var_123, var_96, var_124, var_125)
    assert f'{type(var_126).__module__}.{type(var_126).__qualname__}' == 'typesystem.fields.Object'
    assert var_126.title == ''
    assert var_126.description == ''
    assert var_126.allow_null is False
    assert var_126.read_only is False
    assert var_126.properties == {}
    assert var_126.pattern_properties == {}
    assert var_126.additional_properties is False
    assert var_126.property_names is None
    assert var_126.min_properties is None
    assert var_126.max_properties is None
    assert var_126.required == []
    var_127 = {var_66: var_37}
    var_128 = {var_121: var_127}
    var_129 = False
    var_130 = module_4.Definitions()
    assert f'{type(var_130).__module__}.{type(var_130).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_130) == 0
    var_131 = module_0.from_json_schema_type(var_128, var_96, var_129, var_130)
    assert f'{type(var_131).__module__}.{type(var_131).__qualname__}' == 'typesystem.fields.Object'
    assert var_131.title == ''
    assert var_131.description == ''
    assert var_131.allow_null is False
    assert var_131.read_only is False
    assert var_131.properties == {}
    assert var_131.pattern_properties == {}
    assert f'{type(var_131.additional_properties).__module__}.{type(var_131.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_131.property_names is None
    assert var_131.min_properties is None
    assert var_131.max_properties is None
    assert var_131.required == []
    var_132 = var_131.additional_properties
    var_133 = 'propertyNames'
    var_134 = {var_31: var_33}
    var_135 = {var_133: var_134}
    var_136 = False
    var_137 = module_4.Definitions()
    assert f'{type(var_137).__module__}.{type(var_137).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_137) == 0
    var_138 = module_0.from_json_schema_type(var_135, var_96, var_136, var_137)
    assert f'{type(var_138).__module__}.{type(var_138).__qualname__}' == 'typesystem.fields.Object'
    assert var_138.title == ''
    assert var_138.description == ''
    assert var_138.allow_null is False
    assert var_138.read_only is False
    assert var_138.properties == {}
    assert var_138.pattern_properties == {}
    assert var_138.additional_properties is None
    assert f'{type(var_138.property_names).__module__}.{type(var_138.property_names).__qualname__}' == 'typesystem.fields.Union'
    assert var_138.min_properties is None
    assert var_138.max_properties is None
    assert var_138.required == []
    var_139 = var_138.property_names

def test_case_22():
    var_0 = 'allOf'
    var_1 = 'minLength'
    var_2 = 5
    var_3 = module_5.purge()
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
    var_4 = module_4.Definitions()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = '$ref'
    var_6 = '#/components/schemas/StringType'
    var_7 = {var_5: var_6}
    var_8 = {var_1: var_2}
    var_9 = [var_7, var_8]
    var_10 = {var_0: var_9}
    var_11 = module_0.all_of_from_json_schema(var_10, var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.all_of).__module__}.{type(var_11.all_of).__qualname__}' == 'builtins.list'
    assert len(var_11.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_12 = var_11.all_of

def test_case_23():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = '#/components/schemas/MySchema'
    var_3 = {var_1: var_2}
    var_4 = module_0.ref_from_json_schema(var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == '#/components/schemas/MySchema'
    assert f'{type(var_4.definitions).__module__}.{type(var_4.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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

def test_case_24():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = {var_1: var_1}
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_2, var_0)

def test_case_25():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.type_from_json_schema(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 5
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_0.to_json_schema(var_2, var_0)
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7

def test_case_26():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_0 = '$ref'
    var_1 = {var_0: var_0}
    module_0.from_json_schema(var_1)

def test_case_28():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = var_0.__contains__(var_1)
    assert var_2 is False
    var_3 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_0.get_valid_types(var_3)
    var_5 = module_0.type_from_json_schema(var_3, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert f'{type(var_5.any_of).__module__}.{type(var_5.any_of).__qualname__}' == 'builtins.list'
    assert len(var_5.any_of) == 5
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_6 = module_0.from_json_schema(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Any'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_7 = module_0.from_json_schema(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_8 = module_0.to_json_schema(var_7)
    assert var_8 is False
    var_9 = module_3.Integer(exclusive_maximum=var_8, precision=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Integer'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.minimum is None
    assert var_9.maximum is None
    assert var_9.exclusive_minimum is None
    assert var_9.exclusive_maximum is False
    assert var_9.multiple_of is None
    assert var_9.precision is None
    assert var_9.coerce_types is True
    var_10 = module_0.to_json_schema(var_9)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_3.Float()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_2 = module_0.get_standard_properties(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_4 = module_0.type_from_json_schema(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is True
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_0.to_json_schema(var_0)
    var_6 = var_0.__setitem__(var_3, var_3)
    assert len(var_0) == 1
    var_7 = module_0.from_json_schema(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Any'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    var_8 = module_0.from_json_schema(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Any'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    var_9 = module_3.Choice()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Choice'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.choices == []
    assert var_9.coerce_types is True
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_10 = module_0.to_json_schema(var_4)
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_1.IfThenElse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.if_clause is None
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_0.to_json_schema(var_2)

def test_case_31():
    var_0 = None
    var_1 = module_3.Choice(coerce_types=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Choice'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.choices == []
    assert var_1.coerce_types is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_2 = module_0.to_json_schema(var_1, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_32():
    var_0 = 'enum'
    var_1 = 2
    var_2 = 'a'
    var_3 = True
    var_4 = None
    var_5 = [var_2, var_1, var_3, var_4]
    var_6 = {var_0: var_5}
    var_7 = module_0.from_json_schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Choice'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.choices == [('a', 'a'), (2, 2), (True, True), (None, None)]
    assert var_7.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}

def test_case_33():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 0
    var_3 = module_4.Definitions()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 0
    var_4 = '$ref'
    var_5 = '#/definitions/StringType'
    var_6 = {var_4: var_5}
    var_7 = 'null'
    var_8 = {var_1: var_7}
    var_9 = [var_6, var_8]
    var_10 = {var_0: var_9}
    var_11 = module_0.one_of_from_json_schema(var_10, var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.one_of).__module__}.{type(var_11.one_of).__qualname__}' == 'builtins.list'
    assert len(var_11.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_12 = var_11.one_of
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = var_11.one_of[var_2]

def test_case_34():
    var_0 = 'anyOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = [var_3, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_4.Definitions()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 0
    var_7 = module_0.any_of_from_json_schema(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Union'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.any_of).__module__}.{type(var_7.any_of).__qualname__}' == 'builtins.list'
    assert len(var_7.any_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    with pytest.raises(AttributeError):
        var_8 = var_6.any_of

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_1.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.negated == {}
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_4 = var_0.setdefault(var_1)
    assert len(var_0) == 1
    var_5 = module_0.from_json_schema(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    module_0.to_json_schema(var_3)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = var_0.__len__()
    assert var_1 == 0
    var_2 = None
    var_3 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = var_0.update(**var_3)
    var_5 = module_4.Reference(var_4, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.to is None
    assert var_5.definitions is None
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_6 = module_0.from_json_schema(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Any'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    var_7 = var_5.get_default_value()
    module_0.to_json_schema(var_5)

def test_case_37():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.type_from_json_schema(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_5 = module_3.String(coerce_types=var_2)
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
    assert var_5.coerce_types == {}
    assert var_5.pattern is None
    assert var_5.pattern_regex is None
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = module_0.to_json_schema(var_5, var_1)

def test_case_38():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_3.Const(var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Const'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.const).__module__}.{type(var_2.const).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2.const) == 0
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_3 = module_0.from_json_schema(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    var_4 = module_0.to_json_schema(var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_4.Schema(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.fields).__module__}.{type(var_2.fields).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2.fields) == 0
    assert var_2.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.to_json_schema(var_2, var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = var_0.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_4) == 0
    module_0.type_from_json_schema(var_1, var_1)

def test_case_40():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = False
    var_2 = module_0.from_json_schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_3 = '=@LFc'
    var_4 = module_3.Field(description=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == ''
    assert var_4.description == '=@LFc'
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Field.errors == {}
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_4)

def test_case_41():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_4.Definitions()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = module_0.one_of_from_json_schema(var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.one_of).__module__}.{type(var_9.one_of).__qualname__}' == 'builtins.list'
    assert len(var_9.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.one_of[var_12]
    var_14 = 1
    var_15 = var_9.one_of[var_14]
    var_16 = 'default'
    var_17 = {var_1: var_2}
    var_18 = 'number'
    var_19 = {var_1: var_18}
    var_20 = [var_17, var_19]
    var_21 = 'test'
    var_22 = {var_0: var_20, var_16: var_21}
    var_23 = module_4.Definitions()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_23) == 0
    var_24 = module_0.one_of_from_json_schema(var_22, var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_24.default == 'test'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.one_of).__module__}.{type(var_24.one_of).__qualname__}' == 'builtins.list'
    assert len(var_24.one_of) == 2
    var_25 = 'properties'
    var_26 = 'object'
    var_27 = 'name'
    var_28 = {var_1: var_2}
    var_29 = {var_27: var_28}
    var_30 = {var_1: var_26, var_25: var_29}
    var_31 = 'items'
    var_32 = 'array'
    var_33 = {var_1: var_4}
    var_34 = {var_1: var_32, var_31: var_33}
    var_35 = [var_30, var_34]
    var_36 = {var_0: var_35}
    var_37 = module_4.Definitions()
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_37) == 0
    var_38 = module_0.one_of_from_json_schema(var_36, var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert f'{type(var_38.one_of).__module__}.{type(var_38.one_of).__qualname__}' == 'builtins.list'
    assert len(var_38.one_of) == 2
    var_39 = var_38.one_of
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = var_38.one_of[var_12]
    var_42 = var_38.one_of[var_14]
    var_43 = 'boolean'
    var_44 = {var_1: var_43}
    var_45 = [var_44]
    var_46 = {var_0: var_45}
    var_47 = module_4.Definitions()
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_47) == 0
    var_48 = module_0.one_of_from_json_schema(var_46, var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert f'{type(var_48.one_of).__module__}.{type(var_48.one_of).__qualname__}' == 'builtins.list'
    assert len(var_48.one_of) == 1
    var_49 = var_48.one_of
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = var_48.one_of[var_12]
    var_52 = module_4.Definitions()
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_52) == 0
    var_53 = '$ref'
    var_54 = '#/definitions/StringType'
    var_55 = {var_53: var_54}
    var_56 = {var_1: var_4}
    var_57 = [var_55, var_56]
    var_58 = {var_0: var_57}
    var_59 = module_0.one_of_from_json_schema(var_58, var_52)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.one_of).__module__}.{type(var_59.one_of).__qualname__}' == 'builtins.list'
    assert len(var_59.one_of) == 2
    var_60 = var_59.one_of
    var_61 = len(var_60)
    assert var_61 == 2
    var_62 = var_59.one_of[var_12]
    var_63 = var_59.one_of[var_14]
    var_64 = {var_1: var_2}
    var_65 = {var_1: var_18}
    var_66 = [var_64, var_65]
    var_67 = {var_0: var_66}
    var_68 = module_4.Definitions()
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_68) == 0
    var_69 = module_0.one_of_from_json_schema(var_67, var_68)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert f'{type(var_69.one_of).__module__}.{type(var_69.one_of).__qualname__}' == 'builtins.list'
    assert len(var_69.one_of) == 2
    var_70 = 'enum'
    var_71 = 2
    var_72 = 3
    var_73 = [var_14, var_71, var_72]
    var_74 = {var_70: var_73}
    var_75 = 'const'
    var_76 = 'fixed'
    var_77 = {var_75: var_76}
    var_78 = [var_74, var_77]
    var_79 = {var_0: var_78}
    var_80 = module_4.Definitions()
    assert f'{type(var_80).__module__}.{type(var_80).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_80) == 0
    var_81 = module_0.one_of_from_json_schema(var_79, var_80)
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_81.title == ''
    assert var_81.description == ''
    assert var_81.allow_null is False
    assert var_81.read_only is False
    assert f'{type(var_81.one_of).__module__}.{type(var_81.one_of).__qualname__}' == 'builtins.list'
    assert len(var_81.one_of) == 2
    var_82 = var_81.one_of
    var_83 = len(var_82)
    assert var_83 == 2
    var_84 = var_81.one_of[var_12]
    var_85 = var_81.one_of[var_14]

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = module_4.Definitions()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = {var_1: var_3}
    var_6 = {var_0: var_5}
    module_0.one_of_from_json_schema(var_6, var_4)

def test_case_43():
    var_0 = module_3.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_4 = False
    var_5 = 1
    var_6 = 0
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_3.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length == 0
    assert var_9.min_length == 1
    assert var_9.format == 'email'
    assert var_9.coerce_types is True
    assert var_9.pattern == '^[a-z]+$'
    assert f'{type(var_9.pattern_regex).__module__}.{type(var_9.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = True
    var_12 = module_3.String()
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
    var_13 = module_0.to_json_schema(var_12)
    var_14 = True
    var_15 = module_3.Integer(minimum=var_4, maximum=var_14, exclusive_minimum=var_14, exclusive_maximum=var_5, multiple_of=var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Integer'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is False
    assert var_15.maximum is True
    assert var_15.exclusive_minimum is True
    assert var_15.exclusive_maximum == 1
    assert var_15.multiple_of is False
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_3.Float(minimum=var_4, maximum=var_14)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Float'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum is False
    assert var_17.maximum is True
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of is None
    assert var_17.precision is None
    assert var_17.coerce_types is True
    var_18 = module_0.to_json_schema(var_17)
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
    var_20 = module_0.to_json_schema(var_19)
    var_21 = True
    var_22 = module_3.Boolean()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.coerce_types is True
    var_23 = module_0.to_json_schema(var_22)
    var_24 = module_3.String()
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
    var_25 = True
    var_26 = module_3.Array(var_24, min_items=var_21, max_items=var_6, unique_items=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Array'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.items).__module__}.{type(var_26.items).__qualname__}' == 'typesystem.fields.String'
    assert var_26.additional_items is False
    assert var_26.min_items is True
    assert var_26.max_items == 0
    assert var_26.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_27 = module_0.to_json_schema(var_26)
    var_28 = var_19.get_default_value()
    var_29 = module_3.Integer()
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
    var_30 = module_0.to_json_schema(var_26)
    var_31 = 'items'
    var_32 = var_30[var_31]
    var_33 = var_30[var_31]
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = module_3.Array(additional_items=var_4)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Array'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.items is None
    assert var_35.additional_items is False
    assert var_35.min_items is None
    assert var_35.max_items is None
    assert var_35.unique_items is False
    var_36 = module_0.to_json_schema(var_35)
    var_37 = module_3.String()
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.String'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.allow_blank is False
    assert var_37.trim_whitespace is True
    assert var_37.max_length is None
    assert var_37.min_length is None
    assert var_37.format is None
    assert var_37.coerce_types is True
    assert var_37.pattern is None
    assert var_37.pattern_regex is None
    var_38 = module_3.Array(additional_items=var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Array'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.items is None
    assert f'{type(var_38.additional_items).__module__}.{type(var_38.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_38.min_items is None
    assert var_38.max_items is None
    assert var_38.unique_items is False
    var_39 = module_0.to_json_schema(var_38)
    var_40 = 'additionalItems'
    var_41 = var_39[var_40]
    var_42 = 'name'
    var_43 = 'Qb0n~Y(q(5\n1j'
    var_44 = module_3.String()
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
    var_45 = module_3.Integer()
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
    var_46 = {var_42: var_44, var_43: var_45}
    var_47 = [var_42]
    var_48 = module_3.Object(properties=var_46, min_properties=var_25, max_properties=var_6, required=var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.Object'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert f'{type(var_48.properties).__module__}.{type(var_48.properties).__qualname__}' == 'builtins.dict'
    assert len(var_48.properties) == 2
    assert var_48.pattern_properties == {}
    assert var_48.additional_properties is True
    assert var_48.property_names is None
    assert var_48.min_properties is True
    assert var_48.max_properties == 0
    assert var_48.required == ['name']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_49 = module_0.to_json_schema(var_48)
    var_50 = module_3.String()
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
    var_51 = module_3.Integer()
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Integer'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert var_51.minimum is None
    assert var_51.maximum is None
    assert var_51.exclusive_minimum is None
    assert var_51.exclusive_maximum is None
    assert var_51.multiple_of is None
    assert var_51.precision is None
    assert var_51.coerce_types is True
    var_52 = module_3.Object(additional_properties=var_4)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Object'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert var_52.properties == {}
    assert var_52.pattern_properties == {}
    assert var_52.additional_properties is False
    assert var_52.property_names is None
    assert var_52.min_properties is None
    assert var_52.max_properties is None
    assert var_52.required == []
    var_53 = module_0.to_json_schema(var_52)
    var_54 = module_3.String()
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.String'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert var_54.allow_blank is False
    assert var_54.trim_whitespace is True
    assert var_54.max_length is None
    assert var_54.min_length is None
    assert var_54.format is None
    assert var_54.coerce_types is True
    assert var_54.pattern is None
    assert var_54.pattern_regex is None
    var_55 = module_3.Object(additional_properties=var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Object'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.properties == {}
    assert var_55.pattern_properties == {}
    assert f'{type(var_55.additional_properties).__module__}.{type(var_55.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_55.property_names is None
    assert var_55.min_properties is None
    assert var_55.max_properties is None
    assert var_55.required == []
    var_56 = module_0.to_json_schema(var_55)
    var_57 = 'additionalProperties'
    var_58 = var_56[var_57]
    var_59 = module_3.String(pattern=var_7)
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
    assert var_59.pattern == '^[a-z]+$'
    assert f'{type(var_59.pattern_regex).__module__}.{type(var_59.pattern_regex).__qualname__}' == 're.Pattern'
    var_60 = module_3.Object(property_names=var_59)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.Object'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.properties == {}
    assert var_60.pattern_properties == {}
    assert var_60.additional_properties is True
    assert f'{type(var_60.property_names).__module__}.{type(var_60.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_60.min_properties is None
    assert var_60.max_properties is None
    assert var_60.required == []
    var_61 = module_0.to_json_schema(var_60)
    var_62 = 'Option A'
    var_63 = (var_32, var_62)
    var_64 = 'b'
    var_65 = 'Opt)on B'
    var_66 = (var_64, var_65)
    var_67 = [var_63, var_66]
    var_68 = module_3.Choice(choices=var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.fields.Choice'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert var_68.choices == [({'type': 'string', 'minLength': 1}, 'Option A'), ('b', 'Opt)on B')]
    assert var_68.coerce_types is True
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_69 = module_0.to_json_schema(var_68)
    var_70 = 'fixed_value'
    var_71 = module_3.Const(var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Const'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.const == 'fixed_value'
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_72 = module_0.to_json_schema(var_71)
    var_73 = module_3.String()
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.String'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert var_73.allow_blank is False
    assert var_73.trim_whitespace is True
    assert var_73.max_length is None
    assert var_73.min_length is None
    assert var_73.format is None
    assert var_73.coerce_types is True
    assert var_73.pattern is None
    assert var_73.pattern_regex is None
    var_74 = module_3.Integer()
    assert f'{type(var_74).__module__}.{type(var_74).__qualname__}' == 'typesystem.fields.Integer'
    assert var_74.title == ''
    assert var_74.description == ''
    assert var_74.allow_null is False
    assert var_74.read_only is False
    assert var_74.minimum is None
    assert var_74.maximum is None
    assert var_74.exclusive_minimum is None
    assert var_74.exclusive_maximum is None
    assert var_74.multiple_of is None
    assert var_74.precision is None
    assert var_74.coerce_types is True
    var_75 = len(var_56)
    assert var_75 == 2
    var_76 = module_3.String()
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
    var_77 = module_3.Integer()
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.Integer'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert var_77.minimum is None
    assert var_77.maximum is None
    assert var_77.exclusive_minimum is None
    assert var_77.exclusive_maximum is None
    assert var_77.multiple_of is None
    assert var_77.precision is None
    assert var_77.coerce_types is True
    var_78 = module_1.OneOf(var_75)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert var_78.one_of == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}

def test_case_44():
    var_0 = module_3.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_4 = False
    var_5 = 1
    var_6 = 0
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_3.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length == 0
    assert var_9.min_length == 1
    assert var_9.format == 'email'
    assert var_9.coerce_types is True
    assert var_9.pattern == '^[a-z]+$'
    assert f'{type(var_9.pattern_regex).__module__}.{type(var_9.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = True
    var_12 = module_3.String()
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
    var_13 = module_0.to_json_schema(var_12)
    var_14 = True
    var_15 = module_3.Integer(minimum=var_4, maximum=var_14, exclusive_minimum=var_14, exclusive_maximum=var_5, multiple_of=var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Integer'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is False
    assert var_15.maximum is True
    assert var_15.exclusive_minimum is True
    assert var_15.exclusive_maximum == 1
    assert var_15.multiple_of is False
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_3.Float(minimum=var_4, maximum=var_14)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Float'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum is False
    assert var_17.maximum is True
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of is None
    assert var_17.precision is None
    assert var_17.coerce_types is True
    var_18 = module_0.to_json_schema(var_17)
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
    var_20 = module_0.to_json_schema(var_19)
    var_21 = True
    var_22 = module_3.Boolean()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.coerce_types is True
    var_23 = module_0.to_json_schema(var_22)
    var_24 = module_3.String()
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
    var_25 = True
    var_26 = module_3.Array(var_24, min_items=var_21, max_items=var_6, unique_items=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Array'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.items).__module__}.{type(var_26.items).__qualname__}' == 'typesystem.fields.String'
    assert var_26.additional_items is False
    assert var_26.min_items is True
    assert var_26.max_items == 0
    assert var_26.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_27 = module_0.to_json_schema(var_26)
    var_28 = var_19.get_default_value()
    var_29 = module_3.Integer()
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
    var_30 = module_0.to_json_schema(var_26)
    var_31 = 'items'
    var_32 = var_30[var_31]
    var_33 = var_30[var_31]
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = module_3.Array(additional_items=var_4)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Array'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.items is None
    assert var_35.additional_items is False
    assert var_35.min_items is None
    assert var_35.max_items is None
    assert var_35.unique_items is False
    var_36 = module_0.to_json_schema(var_35)
    var_37 = module_3.String()
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.String'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.allow_blank is False
    assert var_37.trim_whitespace is True
    assert var_37.max_length is None
    assert var_37.min_length is None
    assert var_37.format is None
    assert var_37.coerce_types is True
    assert var_37.pattern is None
    assert var_37.pattern_regex is None
    var_38 = module_3.Array(additional_items=var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Array'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.items is None
    assert f'{type(var_38.additional_items).__module__}.{type(var_38.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_38.min_items is None
    assert var_38.max_items is None
    assert var_38.unique_items is False
    var_39 = module_0.to_json_schema(var_38)
    var_40 = 'additionalItems'
    var_41 = var_39[var_40]
    var_42 = 'name'
    var_43 = 'Qb0n~Y(q(5\n1j'
    var_44 = module_3.String()
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
    var_45 = var_32.setdefault(var_34)
    var_46 = {var_42: var_44, var_43: var_45}
    var_47 = [var_42]
    with pytest.raises(AssertionError):
        module_3.Object(properties=var_46, min_properties=var_25, max_properties=var_6, required=var_47)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = module_3.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_4 = False
    var_5 = 1
    var_6 = 0
    var_7 = '^[a-z]+$'
    var_8 = module_3.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length == 0
    assert var_8.min_length == 1
    assert var_8.format == '^[a-z]+$'
    assert var_8.coerce_types is True
    assert var_8.pattern == '^[a-z]+$'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = True
    var_11 = module_3.String()
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
    var_12 = module_0.to_json_schema(var_11)
    var_13 = True
    var_14 = module_3.Integer(minimum=var_4, maximum=var_13, exclusive_minimum=var_13, exclusive_maximum=var_5, multiple_of=var_3)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum is False
    assert var_14.maximum is True
    assert var_14.exclusive_minimum is True
    assert var_14.exclusive_maximum == 1
    assert var_14.multiple_of is False
    assert var_14.precision is None
    assert var_14.coerce_types is True
    var_15 = module_3.Float(minimum=var_4, maximum=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Float'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is False
    assert var_15.maximum is True
    assert var_15.exclusive_minimum is None
    assert var_15.exclusive_maximum is None
    assert var_15.multiple_of is None
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_3.Boolean()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.coerce_types is True
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_18 = module_0.to_json_schema(var_17)
    var_19 = True
    var_20 = module_3.Boolean()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.coerce_types is True
    var_21 = module_0.to_json_schema(var_20)
    var_22 = module_3.String()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.String'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.allow_blank is False
    assert var_22.trim_whitespace is True
    assert var_22.max_length is None
    assert var_22.min_length is None
    assert var_22.format is None
    assert var_22.coerce_types is True
    assert var_22.pattern is None
    assert var_22.pattern_regex is None
    var_23 = var_17.get_default_value()
    var_24 = module_3.Integer()
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
    module_0.to_json_schema(var_23)

def test_case_46():
    var_0 = module_3.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_4 = False
    var_5 = True
    var_6 = None
    var_7 = 10
    var_8 = module_3.String(allow_blank=var_5, max_length=var_7, min_length=var_6, format=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.default == ''
    assert var_8.allow_blank is True
    assert var_8.trim_whitespace is True
    assert var_8.max_length == 10
    assert var_8.min_length is None
    assert var_8.format is None
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = module_0.to_json_schema(var_8)
    var_11 = 5
    var_12 = module_3.String(allow_blank=var_4, max_length=var_6, min_length=var_11, format=var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.allow_blank is False
    assert var_12.trim_whitespace is True
    assert var_12.max_length is None
    assert var_12.min_length == 5
    assert var_12.format is None
    assert var_12.coerce_types is True
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    var_13 = module_0.to_json_schema(var_12)
    var_14 = 100
    var_15 = module_3.Integer(minimum=var_4, maximum=var_14, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
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
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_3.Float(minimum=var_4, maximum=var_5, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Float'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum is False
    assert var_17.maximum is True
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of is None
    assert var_17.precision is None
    assert var_17.coerce_types is True
    var_18 = module_0.to_json_schema(var_17)
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
    var_20 = module_0.to_json_schema(var_19)
    var_21 = module_3.Boolean()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.coerce_types is True
    var_22 = module_0.to_json_schema(var_21)
    var_23 = module_3.Array(var_6, var_5, var_4, var_6, unique_items=var_4)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Array'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.items is None
    assert var_23.additional_items is True
    assert var_23.min_items is False
    assert var_23.max_items is None
    assert var_23.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.String'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.default == ''
    assert var_25.allow_blank is True
    assert var_25.trim_whitespace is True
    assert var_25.max_length is None
    assert var_25.min_length is None
    assert var_25.format is None
    assert var_25.coerce_types is True
    assert var_25.pattern is None
    assert var_25.pattern_regex is None
    var_26 = module_3.Array(var_25, var_5, var_5, var_7, unique_items=var_4)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Array'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.items).__module__}.{type(var_26.items).__qualname__}' == 'typesystem.fields.String'
    assert var_26.additional_items is True
    assert var_26.min_items is True
    assert var_26.max_items == 10
    assert var_26.unique_items is False
    var_27 = module_0.to_json_schema(var_26)
    var_28 = module_3.Object(properties=var_6, pattern_properties=var_6, additional_properties=var_6, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_6)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Object'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.properties == {}
    assert var_28.pattern_properties == {}
    assert var_28.additional_properties is None
    assert var_28.property_names is None
    assert var_28.min_properties is None
    assert var_28.max_properties is None
    assert var_28.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = 'name'
    var_31 = 'age'
    var_32 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.String'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.default == ''
    assert var_32.allow_blank is True
    assert var_32.trim_whitespace is True
    assert var_32.max_length is None
    assert var_32.min_length is None
    assert var_32.format is None
    assert var_32.coerce_types is True
    assert var_32.pattern is None
    assert var_32.pattern_regex is None
    var_33 = module_3.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
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
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = [var_30]
    var_36 = module_3.Object(properties=var_34, pattern_properties=var_6, additional_properties=var_6, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Object'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert f'{type(var_36.properties).__module__}.{type(var_36.properties).__qualname__}' == 'builtins.dict'
    assert len(var_36.properties) == 2
    assert var_36.pattern_properties == {}
    assert var_36.additional_properties is None
    assert var_36.property_names is None
    assert var_36.min_properties is None
    assert var_36.max_properties is None
    assert var_36.required == ['name']
    var_37 = module_0.to_json_schema(var_36)
    var_38 = 42
    var_39 = module_3.Const(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Const'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.const == 42
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_40 = module_0.to_json_schema(var_39)
    var_41 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.String'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.default == ''
    assert var_41.allow_blank is True
    assert var_41.trim_whitespace is True
    assert var_41.max_length is None
    assert var_41.min_length is None
    assert var_41.format is None
    assert var_41.coerce_types is True
    assert var_41.pattern is None
    assert var_41.pattern_regex is None
    var_42 = module_3.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
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
    var_43 = [var_41, var_42]
    var_44 = module_3.Union(var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Union'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert f'{type(var_44.any_of).__module__}.{type(var_44.any_of).__qualname__}' == 'builtins.list'
    assert len(var_44.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_45 = module_0.to_json_schema(var_44)
    var_46 = 'anyOf'
    var_47 = var_45[var_46]
    var_48 = len(var_47)
    assert var_48 == 2
    var_49 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.fields.String'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert var_49.default == ''
    assert var_49.allow_blank is True
    assert var_49.trim_whitespace is True
    assert var_49.max_length is None
    assert var_49.min_length is None
    assert var_49.format is None
    assert var_49.coerce_types is True
    assert var_49.pattern is None
    assert var_49.pattern_regex is None
    var_50 = module_3.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.fields.Integer'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert var_50.minimum is None
    assert var_50.maximum is None
    assert var_50.exclusive_minimum is None
    assert var_50.exclusive_maximum is None
    assert var_50.multiple_of is None
    assert var_50.precision is None
    assert var_50.coerce_types is True
    var_51 = [var_49, var_50]
    var_52 = module_1.OneOf(var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert f'{type(var_52.one_of).__module__}.{type(var_52.one_of).__qualname__}' == 'builtins.list'
    assert len(var_52.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_53 = module_0.to_json_schema(var_52)
    var_54 = 'oneOf'
    var_55 = var_53[var_54]
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.String'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert var_57.default == ''
    assert var_57.allow_blank is True
    assert var_57.trim_whitespace is True
    assert var_57.max_length is None
    assert var_57.min_length is None
    assert var_57.format is None
    assert var_57.coerce_types is True
    assert var_57.pattern is None
    assert var_57.pattern_regex is None
    var_58 = module_3.Object(properties=var_6, pattern_properties=var_6, additional_properties=var_6, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_6)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.Object'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.properties == {}
    assert var_58.pattern_properties == {}
    assert var_58.additional_properties is None
    assert var_58.property_names is None
    assert var_58.min_properties is None
    assert var_58.max_properties is None
    assert var_58.required == []
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
    var_62 = 'allOf'
    var_63 = var_61[var_62]
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.String'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.default == ''
    assert var_65.allow_blank is True
    assert var_65.trim_whitespace is True
    assert var_65.max_length is None
    assert var_65.min_length is None
    assert var_65.format is None
    assert var_65.coerce_types is True
    assert var_65.pattern is None
    assert var_65.pattern_regex is None
    var_66 = module_1.Not(var_65)
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.composites.Not'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert f'{type(var_66.negated).__module__}.{type(var_66.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_67 = module_0.to_json_schema(var_66)
    var_68 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.fields.String'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert var_68.default == ''
    assert var_68.allow_blank is True
    assert var_68.trim_whitespace is True
    assert var_68.max_length is None
    assert var_68.min_length is None
    assert var_68.format is None
    assert var_68.coerce_types is True
    assert var_68.pattern is None
    assert var_68.pattern_regex is None
    var_69 = module_3.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.fields.Integer'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert var_69.minimum is None
    assert var_69.maximum is None
    assert var_69.exclusive_minimum is None
    assert var_69.exclusive_maximum is None
    assert var_69.multiple_of is None
    assert var_69.precision is None
    assert var_69.coerce_types is True
    var_70 = module_1.IfThenElse(var_68, var_69, var_6)
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert f'{type(var_70.if_clause).__module__}.{type(var_70.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_70.then_clause).__module__}.{type(var_70.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_70.else_clause).__module__}.{type(var_70.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_71 = module_0.to_json_schema(var_70)
    var_72 = module_4.Definitions()
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_72) == 0
    var_73 = module_0.to_json_schema(var_72)

def test_case_47():
    var_0 = module_3.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_3.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length == 10
    assert var_9.min_length == 1
    assert var_9.format == 'email'
    assert var_9.coerce_types is True
    assert var_9.pattern == '^[a-z]+$'
    assert f'{type(var_9.pattern_regex).__module__}.{type(var_9.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = True
    var_12 = module_3.String()
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
    var_13 = module_0.to_json_schema(var_12)
    var_14 = True
    var_15 = None
    var_16 = module_3.String(allow_blank=var_14, min_length=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.String'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.default == ''
    assert var_16.allow_blank is True
    assert var_16.trim_whitespace is True
    assert var_16.max_length is None
    assert var_16.min_length is None
    assert var_16.format is None
    assert var_16.coerce_types is True
    assert var_16.pattern is None
    assert var_16.pattern_regex is None
    var_17 = module_0.to_json_schema(var_16)
    var_18 = 100
    var_19 = 99
    var_20 = 5
    var_21 = module_3.Integer(minimum=var_4, maximum=var_18, exclusive_minimum=var_14, exclusive_maximum=var_19, multiple_of=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Integer'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.minimum is False
    assert var_21.maximum == 100
    assert var_21.exclusive_minimum is True
    assert var_21.exclusive_maximum == 99
    assert var_21.multiple_of == 5
    assert var_21.precision is None
    assert var_21.coerce_types is True
    var_22 = module_0.to_json_schema(var_21)
    var_23 = module_3.Float(minimum=var_4, maximum=var_14)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Float'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.minimum is False
    assert var_23.maximum is True
    assert var_23.exclusive_minimum is None
    assert var_23.exclusive_maximum is None
    assert var_23.multiple_of is None
    assert var_23.precision is None
    assert var_23.coerce_types is True
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_3.Boolean()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.coerce_types is True
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_26 = module_0.to_json_schema(var_25)
    var_27 = True
    var_28 = module_3.Boolean()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.coerce_types is True
    var_29 = module_0.to_json_schema(var_28)
    var_30 = module_3.String()
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
    var_31 = True
    var_32 = module_3.Array(var_30, min_items=var_27, max_items=var_6, unique_items=var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Array'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.items).__module__}.{type(var_32.items).__qualname__}' == 'typesystem.fields.String'
    assert var_32.additional_items is False
    assert var_32.min_items is True
    assert var_32.max_items == 10
    assert var_32.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_33 = module_0.to_json_schema(var_32)
    var_34 = module_3.String()
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
    var_35 = module_3.Integer()
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Integer'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.minimum is None
    assert var_35.maximum is None
    assert var_35.exclusive_minimum is None
    assert var_35.exclusive_maximum is None
    assert var_35.multiple_of is None
    assert var_35.precision is None
    assert var_35.coerce_types is True
    var_36 = [var_34, var_35]
    var_37 = module_3.Array(var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Array'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert f'{type(var_37.items).__module__}.{type(var_37.items).__qualname__}' == 'builtins.list'
    assert len(var_37.items) == 2
    assert var_37.additional_items is False
    assert var_37.min_items == 2
    assert var_37.max_items == 2
    assert var_37.unique_items is False
    var_38 = module_0.to_json_schema(var_37)
    var_39 = 'items'
    var_40 = var_38[var_39]
    var_41 = var_38[var_39]
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = module_3.Array(additional_items=var_4)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Array'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.items is None
    assert var_43.additional_items is False
    assert var_43.min_items is None
    assert var_43.max_items is None
    assert var_43.unique_items is False
    var_44 = module_0.to_json_schema(var_43)
    var_45 = module_3.String()
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.String'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.allow_blank is False
    assert var_45.trim_whitespace is True
    assert var_45.max_length is None
    assert var_45.min_length is None
    assert var_45.format is None
    assert var_45.coerce_types is True
    assert var_45.pattern is None
    assert var_45.pattern_regex is None
    var_46 = module_3.Array(additional_items=var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Array'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert var_46.items is None
    assert f'{type(var_46.additional_items).__module__}.{type(var_46.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_46.min_items is None
    assert var_46.max_items is None
    assert var_46.unique_items is False
    var_47 = module_0.to_json_schema(var_46)
    var_48 = 'additionalItems'
    var_49 = var_47[var_48]
    var_50 = 'name'
    var_51 = 'age'
    var_52 = module_3.String()
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
    var_53 = module_3.Integer()
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
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = [var_50]
    var_56 = module_3.Object(properties=var_54, min_properties=var_31, max_properties=var_6, required=var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Object'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.properties).__module__}.{type(var_56.properties).__qualname__}' == 'builtins.dict'
    assert len(var_56.properties) == 2
    assert var_56.pattern_properties == {}
    assert var_56.additional_properties is True
    assert var_56.property_names is None
    assert var_56.min_properties is True
    assert var_56.max_properties == 10
    assert var_56.required == ['name']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_57 = module_0.to_json_schema(var_56)
    var_58 = '^S_'
    var_59 = '^I_'
    var_60 = module_3.String()
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.String'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.allow_blank is False
    assert var_60.trim_whitespace is True
    assert var_60.max_length is None
    assert var_60.min_length is None
    assert var_60.format is None
    assert var_60.coerce_types is True
    assert var_60.pattern is None
    assert var_60.pattern_regex is None
    var_61 = module_3.Integer()
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.fields.Integer'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert var_61.minimum is None
    assert var_61.maximum is None
    assert var_61.exclusive_minimum is None
    assert var_61.exclusive_maximum is None
    assert var_61.multiple_of is None
    assert var_61.precision is None
    assert var_61.coerce_types is True
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = module_3.Object(pattern_properties=var_62)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.fields.Object'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert var_63.properties == {}
    assert f'{type(var_63.pattern_properties).__module__}.{type(var_63.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_63.pattern_properties) == 2
    assert var_63.additional_properties is True
    assert var_63.property_names is None
    assert var_63.min_properties is None
    assert var_63.max_properties is None
    assert var_63.required == []
    var_64 = module_0.to_json_schema(var_63)
    var_65 = module_3.Object(additional_properties=var_4)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Object'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.properties == {}
    assert var_65.pattern_properties == {}
    assert var_65.additional_properties is False
    assert var_65.property_names is None
    assert var_65.min_properties is None
    assert var_65.max_properties is None
    assert var_65.required == []
    var_66 = module_0.to_json_schema(var_65)
    var_67 = module_3.String()
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
    var_68 = module_3.Object(additional_properties=var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.fields.Object'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert var_68.properties == {}
    assert var_68.pattern_properties == {}
    assert f'{type(var_68.additional_properties).__module__}.{type(var_68.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_68.property_names is None
    assert var_68.min_properties is None
    assert var_68.max_properties is None
    assert var_68.required == []
    var_69 = module_0.to_json_schema(var_68)
    var_70 = 'additionalProperties'
    var_71 = var_69[var_70]
    var_72 = module_3.String(pattern=var_7)
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.String'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert var_72.allow_blank is False
    assert var_72.trim_whitespace is True
    assert var_72.max_length is None
    assert var_72.min_length is None
    assert var_72.format is None
    assert var_72.coerce_types is True
    assert var_72.pattern == '^[a-z]+$'
    assert f'{type(var_72.pattern_regex).__module__}.{type(var_72.pattern_regex).__qualname__}' == 're.Pattern'
    var_73 = module_3.Object(property_names=var_72)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.Object'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert var_73.properties == {}
    assert var_73.pattern_properties == {}
    assert var_73.additional_properties is True
    assert f'{type(var_73.property_names).__module__}.{type(var_73.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_73.min_properties is None
    assert var_73.max_properties is None
    assert var_73.required == []
    var_74 = module_0.to_json_schema(var_73)
    var_75 = 'a'
    var_76 = 'Option A'
    var_77 = (var_75, var_76)
    var_78 = 'b'
    var_79 = 'Option B'
    var_80 = (var_78, var_79)
    var_81 = [var_77, var_80]
    var_82 = module_3.Choice(choices=var_81)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.Choice'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert var_82.choices == [('a', 'Option A'), ('b', 'Option B')]
    assert var_82.coerce_types is True
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_83 = module_0.to_json_schema(var_82)
    var_84 = 'fixed_value'
    var_85 = module_3.Const(var_84)
    assert f'{type(var_85).__module__}.{type(var_85).__qualname__}' == 'typesystem.fields.Const'
    assert var_85.title == ''
    assert var_85.description == ''
    assert var_85.allow_null is False
    assert var_85.read_only is False
    assert var_85.const == 'fixed_value'
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_86 = module_0.to_json_schema(var_85)
    var_87 = module_3.String()
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.fields.String'
    assert var_87.title == ''
    assert var_87.description == ''
    assert var_87.allow_null is False
    assert var_87.read_only is False
    assert var_87.allow_blank is False
    assert var_87.trim_whitespace is True
    assert var_87.max_length is None
    assert var_87.min_length is None
    assert var_87.format is None
    assert var_87.coerce_types is True
    assert var_87.pattern is None
    assert var_87.pattern_regex is None
    var_88 = module_3.Integer()
    assert f'{type(var_88).__module__}.{type(var_88).__qualname__}' == 'typesystem.fields.Integer'
    assert var_88.title == ''
    assert var_88.description == ''
    assert var_88.allow_null is False
    assert var_88.read_only is False
    assert var_88.minimum is None
    assert var_88.maximum is None
    assert var_88.exclusive_minimum is None
    assert var_88.exclusive_maximum is None
    assert var_88.multiple_of is None
    assert var_88.precision is None
    assert var_88.coerce_types is True
    var_89 = [var_87, var_88]
    var_90 = module_3.Union(var_89)
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'typesystem.fields.Union'
    assert var_90.title == ''
    assert var_90.description == ''
    assert var_90.allow_null is False
    assert var_90.read_only is False
    assert f'{type(var_90.any_of).__module__}.{type(var_90.any_of).__qualname__}' == 'builtins.list'
    assert len(var_90.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_91 = 'anyOf'
    with pytest.raises(KeyError):
        var_92 = var_38[var_91]

def test_case_48():
    var_0 = module_3.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_4 = False
    var_5 = 1
    var_6 = 0
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_3.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length == 0
    assert var_9.min_length == 1
    assert var_9.format == 'email'
    assert var_9.coerce_types is True
    assert var_9.pattern == '^[a-z]+$'
    assert f'{type(var_9.pattern_regex).__module__}.{type(var_9.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = True
    var_12 = module_3.String()
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
    var_13 = module_0.to_json_schema(var_12)
    var_14 = True
    var_15 = None
    var_16 = module_3.String(allow_blank=var_14, min_length=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.String'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.default == ''
    assert var_16.allow_blank is True
    assert var_16.trim_whitespace is True
    assert var_16.max_length is None
    assert var_16.min_length is None
    assert var_16.format is None
    assert var_16.coerce_types is True
    assert var_16.pattern is None
    assert var_16.pattern_regex is None
    var_17 = module_0.to_json_schema(var_16)
    var_18 = 100
    var_19 = 5
    var_20 = module_3.Integer(minimum=var_4, maximum=var_18, exclusive_minimum=var_14, exclusive_maximum=var_5, multiple_of=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Integer'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.minimum is False
    assert var_20.maximum == 100
    assert var_20.exclusive_minimum is True
    assert var_20.exclusive_maximum == 1
    assert var_20.multiple_of == 5
    assert var_20.precision is None
    assert var_20.coerce_types is True
    var_21 = module_0.to_json_schema(var_20)
    var_22 = module_3.Float(minimum=var_4, maximum=var_14)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Float'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.minimum is False
    assert var_22.maximum is True
    assert var_22.exclusive_minimum is None
    assert var_22.exclusive_maximum is None
    assert var_22.multiple_of is None
    assert var_22.precision is None
    assert var_22.coerce_types is True
    var_23 = module_0.to_json_schema(var_22)
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
    var_25 = module_0.to_json_schema(var_24)
    var_26 = True
    var_27 = module_3.Boolean()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.coerce_types is True
    var_28 = module_0.to_json_schema(var_27)
    var_29 = module_3.String()
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
    var_30 = True
    var_31 = module_3.Array(var_29, min_items=var_26, max_items=var_6, unique_items=var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Array'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert f'{type(var_31.items).__module__}.{type(var_31.items).__qualname__}' == 'typesystem.fields.String'
    assert var_31.additional_items is False
    assert var_31.min_items is True
    assert var_31.max_items == 0
    assert var_31.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_32 = module_0.to_json_schema(var_31)
    var_33 = module_3.String()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.String'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.allow_blank is False
    assert var_33.trim_whitespace is True
    assert var_33.max_length is None
    assert var_33.min_length is None
    assert var_33.format is None
    assert var_33.coerce_types is True
    assert var_33.pattern is None
    assert var_33.pattern_regex is None
    var_34 = [var_33, var_20]
    var_35 = module_3.Array(var_34)
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
    var_36 = module_0.to_json_schema(var_35)
    var_37 = 'items'
    var_38 = var_36[var_37]
    var_39 = var_36[var_37]
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = module_3.Array(additional_items=var_4)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Array'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.items is None
    assert var_41.additional_items is False
    assert var_41.min_items is None
    assert var_41.max_items is None
    assert var_41.unique_items is False
    var_42 = module_0.to_json_schema(var_41)
    var_43 = module_3.String()
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
    var_44 = module_3.Array(additional_items=var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Array'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert var_44.items is None
    assert f'{type(var_44.additional_items).__module__}.{type(var_44.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_44.min_items is None
    assert var_44.max_items is None
    assert var_44.unique_items is False
    var_45 = module_0.to_json_schema(var_44)
    var_46 = 'additionalItems'
    var_47 = var_45[var_46]
    var_48 = 'name'
    var_49 = 'age'
    var_50 = module_3.String()
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
    var_51 = module_3.Integer()
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Integer'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert var_51.minimum is None
    assert var_51.maximum is None
    assert var_51.exclusive_minimum is None
    assert var_51.exclusive_maximum is None
    assert var_51.multiple_of is None
    assert var_51.precision is None
    assert var_51.coerce_types is True
    var_52 = {var_48: var_50, var_49: var_51}
    var_53 = [var_48]
    var_54 = module_3.Object(properties=var_52, min_properties=var_30, max_properties=var_6, required=var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Object'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert f'{type(var_54.properties).__module__}.{type(var_54.properties).__qualname__}' == 'builtins.dict'
    assert len(var_54.properties) == 2
    assert var_54.pattern_properties == {}
    assert var_54.additional_properties is True
    assert var_54.property_names is None
    assert var_54.min_properties is True
    assert var_54.max_properties == 0
    assert var_54.required == ['name']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_55 = '^S_'
    var_56 = '^I_'
    var_57 = module_3.String()
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
    var_58 = module_3.Integer()
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
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = module_3.Object(pattern_properties=var_59)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.Object'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.properties == {}
    assert f'{type(var_60.pattern_properties).__module__}.{type(var_60.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_60.pattern_properties) == 2
    assert var_60.additional_properties is True
    assert var_60.property_names is None
    assert var_60.min_properties is None
    assert var_60.max_properties is None
    assert var_60.required == []
    var_61 = module_0.to_json_schema(var_60)
    var_62 = module_3.Object(additional_properties=var_4)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.fields.Object'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert var_62.properties == {}
    assert var_62.pattern_properties == {}
    assert var_62.additional_properties is False
    assert var_62.property_names is None
    assert var_62.min_properties is None
    assert var_62.max_properties is None
    assert var_62.required == []
    var_63 = module_0.to_json_schema(var_62)
    var_64 = module_3.String()
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.String'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert var_64.allow_blank is False
    assert var_64.trim_whitespace is True
    assert var_64.max_length is None
    assert var_64.min_length is None
    assert var_64.format is None
    assert var_64.coerce_types is True
    assert var_64.pattern is None
    assert var_64.pattern_regex is None
    var_65 = module_3.Object(additional_properties=var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Object'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.properties == {}
    assert var_65.pattern_properties == {}
    assert f'{type(var_65.additional_properties).__module__}.{type(var_65.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_65.property_names is None
    assert var_65.min_properties is None
    assert var_65.max_properties is None
    assert var_65.required == []
    var_66 = module_0.to_json_schema(var_65)
    var_67 = 'additionalProperties'
    var_68 = var_66[var_67]
    var_69 = module_3.String(pattern=var_7)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.fields.String'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert var_69.allow_blank is False
    assert var_69.trim_whitespace is True
    assert var_69.max_length is None
    assert var_69.min_length is None
    assert var_69.format is None
    assert var_69.coerce_types is True
    assert var_69.pattern == '^[a-z]+$'
    assert f'{type(var_69.pattern_regex).__module__}.{type(var_69.pattern_regex).__qualname__}' == 're.Pattern'
    var_70 = module_3.Object(property_names=var_69)
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.fields.Object'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert var_70.properties == {}
    assert var_70.pattern_properties == {}
    assert var_70.additional_properties is True
    assert f'{type(var_70.property_names).__module__}.{type(var_70.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_70.min_properties is None
    assert var_70.max_properties is None
    assert var_70.required == []
    var_71 = module_0.to_json_schema(var_70)
    var_72 = 'Option A'
    var_73 = (var_38, var_72)
    var_74 = 'b'
    var_75 = 'Option B'
    var_76 = (var_74, var_75)
    var_77 = [var_73, var_76]
    var_78 = module_3.Choice(choices=var_77)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.Choice'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert var_78.choices == [([{'type': 'string', 'minLength': 1}, {'type': 'integer', 'minimum': False, 'maximum': 100, 'exclusiveMinimum': True, 'exclusiveMaximum': 1, 'multipleOf': 5}], 'Option A'), ('b', 'Option B')]
    assert var_78.coerce_types is True
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_79 = module_0.to_json_schema(var_78)
    var_80 = 'fixed_value'
    var_81 = module_3.Const(var_80)
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'typesystem.fields.Const'
    assert var_81.title == ''
    assert var_81.description == ''
    assert var_81.allow_null is False
    assert var_81.read_only is False
    assert var_81.const == 'fixed_value'
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_82 = module_0.to_json_schema(var_81)
    var_83 = module_3.String()
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
    var_84 = module_3.Integer()
    assert f'{type(var_84).__module__}.{type(var_84).__qualname__}' == 'typesystem.fields.Integer'
    assert var_84.title == ''
    assert var_84.description == ''
    assert var_84.allow_null is False
    assert var_84.read_only is False
    assert var_84.minimum is None
    assert var_84.maximum is None
    assert var_84.exclusive_minimum is None
    assert var_84.exclusive_maximum is None
    assert var_84.multiple_of is None
    assert var_84.precision is None
    assert var_84.coerce_types is True
    var_85 = [var_83, var_84]
    var_86 = module_3.Union(var_85)
    assert f'{type(var_86).__module__}.{type(var_86).__qualname__}' == 'typesystem.fields.Union'
    assert var_86.title == ''
    assert var_86.description == ''
    assert var_86.allow_null is False
    assert var_86.read_only is False
    assert f'{type(var_86.any_of).__module__}.{type(var_86.any_of).__qualname__}' == 'builtins.list'
    assert len(var_86.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_87 = module_0.to_json_schema(var_86)
    var_88 = 'anyOf'
    var_89 = var_87[var_88]
    var_90 = len(var_87)
    var_91 = module_3.String()
    assert f'{type(var_91).__module__}.{type(var_91).__qualname__}' == 'typesystem.fields.String'
    assert var_91.title == ''
    assert var_91.description == ''
    assert var_91.allow_null is False
    assert var_91.read_only is False
    assert var_91.allow_blank is False
    assert var_91.trim_whitespace is True
    assert var_91.max_length is None
    assert var_91.min_length is None
    assert var_91.format is None
    assert var_91.coerce_types is True
    assert var_91.pattern is None
    assert var_91.pattern_regex is None
    var_92 = module_3.Integer()
    assert f'{type(var_92).__module__}.{type(var_92).__qualname__}' == 'typesystem.fields.Integer'
    assert var_92.title == ''
    assert var_92.description == ''
    assert var_92.allow_null is False
    assert var_92.read_only is False
    assert var_92.minimum is None
    assert var_92.maximum is None
    assert var_92.exclusive_minimum is None
    assert var_92.exclusive_maximum is None
    assert var_92.multiple_of is None
    assert var_92.precision is None
    assert var_92.coerce_types is True
    var_93 = [var_91, var_92]
    var_94 = module_1.OneOf(var_93)
    assert f'{type(var_94).__module__}.{type(var_94).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_94.title == ''
    assert var_94.description == ''
    assert var_94.allow_null is False
    assert var_94.read_only is False
    assert f'{type(var_94.one_of).__module__}.{type(var_94.one_of).__qualname__}' == 'builtins.list'
    assert len(var_94.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_95 = module_0.to_json_schema(var_94)
    var_96 = 'oneOf'
    var_97 = var_95[var_96]
    var_98 = len(var_97)
    assert var_98 == 2
    var_99 = [var_64, var_29]
    var_100 = module_1.AllOf(var_99)
    assert f'{type(var_100).__module__}.{type(var_100).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_100.title == ''
    assert var_100.description == ''
    assert var_100.allow_null is False
    assert var_100.read_only is False
    assert f'{type(var_100.all_of).__module__}.{type(var_100.all_of).__qualname__}' == 'builtins.list'
    assert len(var_100.all_of) == 2
    var_101 = module_0.to_json_schema(var_100)
    var_102 = 'allOf'
    var_103 = var_101[var_102]
    var_104 = len(var_103)
    assert var_104 == 2
    var_105 = module_1.Not(var_33)
    assert f'{type(var_105).__module__}.{type(var_105).__qualname__}' == 'typesystem.composites.Not'
    assert var_105.title == ''
    assert var_105.description == ''
    assert var_105.allow_null is False
    assert var_105.read_only is False
    assert f'{type(var_105.negated).__module__}.{type(var_105.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_106 = module_0.to_json_schema(var_105)
    var_107 = module_3.String()
    assert f'{type(var_107).__module__}.{type(var_107).__qualname__}' == 'typesystem.fields.String'
    assert var_107.title == ''
    assert var_107.description == ''
    assert var_107.allow_null is False
    assert var_107.read_only is False
    assert var_107.allow_blank is False
    assert var_107.trim_whitespace is True
    assert var_107.max_length is None
    assert var_107.min_length is None
    assert var_107.format is None
    assert var_107.coerce_types is True
    assert var_107.pattern is None
    assert var_107.pattern_regex is None
    var_108 = module_3.Integer()
    assert f'{type(var_108).__module__}.{type(var_108).__qualname__}' == 'typesystem.fields.Integer'
    assert var_108.title == ''
    assert var_108.description == ''
    assert var_108.allow_null is False
    assert var_108.read_only is False
    assert var_108.minimum is None
    assert var_108.maximum is None
    assert var_108.exclusive_minimum is None
    assert var_108.exclusive_maximum is None
    assert var_108.multiple_of is None
    assert var_108.precision is None
    assert var_108.coerce_types is True
    var_109 = module_3.Boolean()
    assert f'{type(var_109).__module__}.{type(var_109).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_109.title == ''
    assert var_109.description == ''
    assert var_109.allow_null is False
    assert var_109.read_only is False
    assert var_109.coerce_types is True
    var_110 = module_1.IfThenElse(var_107, var_108, var_109)
    assert f'{type(var_110).__module__}.{type(var_110).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_110.title == ''
    assert var_110.description == ''
    assert var_110.allow_null is False
    assert var_110.read_only is False
    assert f'{type(var_110.if_clause).__module__}.{type(var_110.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_110.then_clause).__module__}.{type(var_110.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_110.else_clause).__module__}.{type(var_110.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_111 = module_0.to_json_schema(var_110)
    var_112 = module_3.String()
    assert f'{type(var_112).__module__}.{type(var_112).__qualname__}' == 'typesystem.fields.String'
    assert var_112.title == ''
    assert var_112.description == ''
    assert var_112.allow_null is False
    assert var_112.read_only is False
    assert var_112.allow_blank is False
    assert var_112.trim_whitespace is True
    assert var_112.max_length is None
    assert var_112.min_length is None
    assert var_112.format is None
    assert var_112.coerce_types is True
    assert var_112.pattern is None
    assert var_112.pattern_regex is None
    var_113 = module_3.Integer()
    assert f'{type(var_113).__module__}.{type(var_113).__qualname__}' == 'typesystem.fields.Integer'
    assert var_113.title == ''
    assert var_113.description == ''
    assert var_113.allow_null is False
    assert var_113.read_only is False
    assert var_113.minimum is None
    assert var_113.maximum is None
    assert var_113.exclusive_minimum is None
    assert var_113.exclusive_maximum is None
    assert var_113.multiple_of is None
    assert var_113.precision is None
    assert var_113.coerce_types is True
    var_114 = module_1.IfThenElse(var_112, var_113, var_15)
    assert f'{type(var_114).__module__}.{type(var_114).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_114.title == ''
    assert var_114.description == ''
    assert var_114.allow_null is False
    assert var_114.read_only is False
    assert f'{type(var_114.if_clause).__module__}.{type(var_114.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_114.then_clause).__module__}.{type(var_114.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_114.else_clause).__module__}.{type(var_114.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_115 = module_0.to_json_schema(var_114)
    var_116 = module_3.String()
    assert f'{type(var_116).__module__}.{type(var_116).__qualname__}' == 'typesystem.fields.String'
    assert var_116.title == ''
    assert var_116.description == ''
    assert var_116.allow_null is False
    assert var_116.read_only is False
    assert var_116.allow_blank is False
    assert var_116.trim_whitespace is True
    assert var_116.max_length is None
    assert var_116.min_length is None
    assert var_116.format is None
    assert var_116.coerce_types is True
    assert var_116.pattern is None
    assert var_116.pattern_regex is None
    var_117 = module_3.Integer()
    assert f'{type(var_117).__module__}.{type(var_117).__qualname__}' == 'typesystem.fields.Integer'
    assert var_117.title == ''
    assert var_117.description == ''
    assert var_117.allow_null is False
    assert var_117.read_only is False
    assert var_117.minimum is None
    assert var_117.maximum is None
    assert var_117.exclusive_minimum is None
    assert var_117.exclusive_maximum is None
    assert var_117.multiple_of is None
    assert var_117.precision is None
    assert var_117.coerce_types is True
    var_118 = {var_48: var_116, var_49: var_117}
    var_119 = module_4.Schema(var_118)
    assert f'{type(var_119).__module__}.{type(var_119).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_119.title == ''
    assert var_119.description == ''
    assert var_119.allow_null is False
    assert var_119.read_only is False
    assert f'{type(var_119.fields).__module__}.{type(var_119.fields).__qualname__}' == 'builtins.dict'
    assert len(var_119.fields) == 2
    assert var_119.required == ['name', 'age']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_120 = module_0.to_json_schema(var_119)

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = module_3.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_4 = False
    var_5 = 1
    var_6 = 0
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_3.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length == 0
    assert var_9.min_length == 1
    assert var_9.format == 'email'
    assert var_9.coerce_types is True
    assert var_9.pattern == '^[a-z]+$'
    assert f'{type(var_9.pattern_regex).__module__}.{type(var_9.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = True
    var_12 = module_3.String()
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
    var_13 = module_0.to_json_schema(var_12)
    var_14 = True
    var_15 = module_3.Integer(minimum=var_4, maximum=var_14, exclusive_minimum=var_14, exclusive_maximum=var_5, multiple_of=var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Integer'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is False
    assert var_15.maximum is True
    assert var_15.exclusive_minimum is True
    assert var_15.exclusive_maximum == 1
    assert var_15.multiple_of is False
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_3.Float(minimum=var_4, maximum=var_14)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Float'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum is False
    assert var_17.maximum is True
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of is None
    assert var_17.precision is None
    assert var_17.coerce_types is True
    var_18 = module_0.to_json_schema(var_17)
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
    var_20 = module_0.to_json_schema(var_19)
    var_21 = True
    var_22 = module_3.Boolean()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.coerce_types is True
    var_23 = module_0.to_json_schema(var_22)
    var_24 = module_3.String()
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
    var_25 = True
    var_26 = module_3.Array(var_24, min_items=var_21, max_items=var_6, unique_items=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Array'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.items).__module__}.{type(var_26.items).__qualname__}' == 'typesystem.fields.String'
    assert var_26.additional_items is False
    assert var_26.min_items is True
    assert var_26.max_items == 0
    assert var_26.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_27 = module_0.to_json_schema(var_26)
    var_28 = module_3.String()
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
    var_29 = module_3.Integer()
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
    var_30 = module_0.to_json_schema(var_26)
    var_31 = 'items'
    var_32 = var_30[var_31]
    var_33 = var_30[var_31]
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = module_3.Array(additional_items=var_4)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Array'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.items is None
    assert var_35.additional_items is False
    assert var_35.min_items is None
    assert var_35.max_items is None
    assert var_35.unique_items is False
    var_36 = module_0.to_json_schema(var_35)
    var_37 = module_3.String()
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.String'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.allow_blank is False
    assert var_37.trim_whitespace is True
    assert var_37.max_length is None
    assert var_37.min_length is None
    assert var_37.format is None
    assert var_37.coerce_types is True
    assert var_37.pattern is None
    assert var_37.pattern_regex is None
    var_38 = module_3.Array(additional_items=var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Array'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.items is None
    assert f'{type(var_38.additional_items).__module__}.{type(var_38.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_38.min_items is None
    assert var_38.max_items is None
    assert var_38.unique_items is False
    var_39 = module_0.to_json_schema(var_38)
    var_40 = 'additionalItems'
    var_41 = var_39[var_40]
    var_42 = 'name'
    var_43 = 'age'
    var_44 = module_3.String()
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
    var_45 = module_3.Integer()
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
    var_46 = {var_42: var_44, var_43: var_45}
    var_47 = [var_42]
    var_48 = module_3.Object(properties=var_46, min_properties=var_25, max_properties=var_6, required=var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.Object'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert f'{type(var_48.properties).__module__}.{type(var_48.properties).__qualname__}' == 'builtins.dict'
    assert len(var_48.properties) == 2
    assert var_48.pattern_properties == {}
    assert var_48.additional_properties is True
    assert var_48.property_names is None
    assert var_48.min_properties is True
    assert var_48.max_properties == 0
    assert var_48.required == ['name']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_49 = module_0.to_json_schema(var_48)
    var_50 = '^S_'
    var_51 = '^I_'
    var_52 = module_3.String()
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
    var_53 = module_3.Integer()
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
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = module_3.Object(pattern_properties=var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Object'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.properties == {}
    assert f'{type(var_55.pattern_properties).__module__}.{type(var_55.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_55.pattern_properties) == 2
    assert var_55.additional_properties is True
    assert var_55.property_names is None
    assert var_55.min_properties is None
    assert var_55.max_properties is None
    assert var_55.required == []
    var_56 = module_0.to_json_schema(var_55)
    var_57 = module_3.Object(additional_properties=var_4)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.Object'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert var_57.properties == {}
    assert var_57.pattern_properties == {}
    assert var_57.additional_properties is False
    assert var_57.property_names is None
    assert var_57.min_properties is None
    assert var_57.max_properties is None
    assert var_57.required == []
    var_58 = module_0.to_json_schema(var_57)
    var_59 = module_3.String()
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
    var_60 = module_3.Object(additional_properties=var_59)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.Object'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.properties == {}
    assert var_60.pattern_properties == {}
    assert f'{type(var_60.additional_properties).__module__}.{type(var_60.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_60.property_names is None
    assert var_60.min_properties is None
    assert var_60.max_properties is None
    assert var_60.required == []
    var_61 = module_0.to_json_schema(var_60)
    var_62 = 'additionalProperties'
    var_63 = var_61[var_62]
    var_64 = module_3.String(pattern=var_7)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.String'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert var_64.allow_blank is False
    assert var_64.trim_whitespace is True
    assert var_64.max_length is None
    assert var_64.min_length is None
    assert var_64.format is None
    assert var_64.coerce_types is True
    assert var_64.pattern == '^[a-z]+$'
    assert f'{type(var_64.pattern_regex).__module__}.{type(var_64.pattern_regex).__qualname__}' == 're.Pattern'
    var_65 = module_3.Object(property_names=var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Object'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.properties == {}
    assert var_65.pattern_properties == {}
    assert var_65.additional_properties is True
    assert f'{type(var_65.property_names).__module__}.{type(var_65.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_65.min_properties is None
    assert var_65.max_properties is None
    assert var_65.required == []
    var_66 = module_0.to_json_schema(var_65)
    var_67 = 'Option A'
    var_68 = (var_32, var_67)
    var_69 = 'b'
    var_70 = 'Option B'
    var_71 = (var_69, var_70)
    var_72 = [var_68, var_71]
    var_73 = module_3.Choice(choices=var_72)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.Choice'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert var_73.choices == [({'type': 'string', 'minLength': 1}, 'Option A'), ('b', 'Option B')]
    assert var_73.coerce_types is True
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_74 = module_0.to_json_schema(var_73)
    var_75 = 'fixed_value'
    var_76 = module_3.Const(var_75)
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.fields.Const'
    assert var_76.title == ''
    assert var_76.description == ''
    assert var_76.allow_null is False
    assert var_76.read_only is False
    assert var_76.const == 'fixed_value'
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_77 = module_0.to_json_schema(var_76)
    var_78 = module_3.Integer()
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.Integer'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert var_78.minimum is None
    assert var_78.maximum is None
    assert var_78.exclusive_minimum is None
    assert var_78.exclusive_maximum is None
    assert var_78.multiple_of is None
    assert var_78.precision is None
    assert var_78.coerce_types is True
    var_79 = len(var_61)
    assert var_79 == 2
    var_80 = module_3.String()
    assert f'{type(var_80).__module__}.{type(var_80).__qualname__}' == 'typesystem.fields.String'
    assert var_80.title == ''
    assert var_80.description == ''
    assert var_80.allow_null is False
    assert var_80.read_only is False
    assert var_80.allow_blank is False
    assert var_80.trim_whitespace is True
    assert var_80.max_length is None
    assert var_80.min_length is None
    assert var_80.format is None
    assert var_80.coerce_types is True
    assert var_80.pattern is None
    assert var_80.pattern_regex is None
    var_81 = module_3.Integer()
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'typesystem.fields.Integer'
    assert var_81.title == ''
    assert var_81.description == ''
    assert var_81.allow_null is False
    assert var_81.read_only is False
    assert var_81.minimum is None
    assert var_81.maximum is None
    assert var_81.exclusive_minimum is None
    assert var_81.exclusive_maximum is None
    assert var_81.multiple_of is None
    assert var_81.precision is None
    assert var_81.coerce_types is True
    var_82 = module_1.OneOf(var_79)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert var_82.one_of == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    module_0.to_json_schema(var_82)

def test_case_50():
    var_0 = module_3.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    var_4 = False
    var_5 = True
    var_6 = None
    var_7 = 10
    var_8 = module_3.String(allow_blank=var_5, max_length=var_7, min_length=var_6, format=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.default == ''
    assert var_8.allow_blank is True
    assert var_8.trim_whitespace is True
    assert var_8.max_length == 10
    assert var_8.min_length is None
    assert var_8.format is None
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = module_0.to_json_schema(var_8)
    var_11 = 5
    var_12 = module_3.String(allow_blank=var_4, max_length=var_6, min_length=var_11, format=var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.allow_blank is False
    assert var_12.trim_whitespace is True
    assert var_12.max_length is None
    assert var_12.min_length == 5
    assert var_12.format is None
    assert var_12.coerce_types is True
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    var_13 = module_0.to_json_schema(var_12)
    var_14 = 100
    var_15 = module_3.Integer(minimum=var_4, maximum=var_14, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
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
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_3.Float(minimum=var_4, maximum=var_5, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Float'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum is False
    assert var_17.maximum is True
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of is None
    assert var_17.precision is None
    assert var_17.coerce_types is True
    var_18 = module_0.to_json_schema(var_17)
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
    var_20 = module_0.to_json_schema(var_19)
    var_21 = module_3.Boolean()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.coerce_types is True
    var_22 = module_0.to_json_schema(var_21)
    var_23 = module_3.Array(var_6, var_5, var_4, var_6, unique_items=var_4)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Array'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.items is None
    assert var_23.additional_items is True
    assert var_23.min_items is False
    assert var_23.max_items is None
    assert var_23.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.String'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.default == ''
    assert var_25.allow_blank is True
    assert var_25.trim_whitespace is True
    assert var_25.max_length is None
    assert var_25.min_length is None
    assert var_25.format is None
    assert var_25.coerce_types is True
    assert var_25.pattern is None
    assert var_25.pattern_regex is None
    var_26 = module_3.Array(var_25, var_5, var_5, var_7, unique_items=var_4)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Array'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.items).__module__}.{type(var_26.items).__qualname__}' == 'typesystem.fields.String'
    assert var_26.additional_items is True
    assert var_26.min_items is True
    assert var_26.max_items == 10
    assert var_26.unique_items is False
    var_27 = module_0.to_json_schema(var_26)
    var_28 = module_3.Object(properties=var_6, pattern_properties=var_6, additional_properties=var_6, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_6)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Object'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.properties == {}
    assert var_28.pattern_properties == {}
    assert var_28.additional_properties is None
    assert var_28.property_names is None
    assert var_28.min_properties is None
    assert var_28.max_properties is None
    assert var_28.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = 'name'
    var_31 = 'age'
    var_32 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.String'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.default == ''
    assert var_32.allow_blank is True
    assert var_32.trim_whitespace is True
    assert var_32.max_length is None
    assert var_32.min_length is None
    assert var_32.format is None
    assert var_32.coerce_types is True
    assert var_32.pattern is None
    assert var_32.pattern_regex is None
    var_33 = module_3.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
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
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = [var_30]
    var_36 = module_3.Object(properties=var_34, pattern_properties=var_6, additional_properties=var_6, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Object'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert f'{type(var_36.properties).__module__}.{type(var_36.properties).__qualname__}' == 'builtins.dict'
    assert len(var_36.properties) == 2
    assert var_36.pattern_properties == {}
    assert var_36.additional_properties is None
    assert var_36.property_names is None
    assert var_36.min_properties is None
    assert var_36.max_properties is None
    assert var_36.required == ['name']
    var_37 = module_0.to_json_schema(var_36)
    var_38 = 'option1'
    var_39 = (var_38, var_38)
    var_40 = 'option2'
    var_41 = (var_40, var_40)
    var_42 = [var_39, var_41]
    var_43 = module_3.Choice(choices=var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Choice'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.choices == [('option1', 'option1'), ('option2', 'option2')]
    assert var_43.coerce_types is True
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_44 = module_0.to_json_schema(var_43)
    var_45 = 42
    var_46 = module_3.Const(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Const'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert var_46.const == 42
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_47 = module_0.to_json_schema(var_46)
    var_48 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.String'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert var_48.default == ''
    assert var_48.allow_blank is True
    assert var_48.trim_whitespace is True
    assert var_48.max_length is None
    assert var_48.min_length is None
    assert var_48.format is None
    assert var_48.coerce_types is True
    assert var_48.pattern is None
    assert var_48.pattern_regex is None
    var_49 = module_3.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.fields.Integer'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert var_49.minimum is None
    assert var_49.maximum is None
    assert var_49.exclusive_minimum is None
    assert var_49.exclusive_maximum is None
    assert var_49.multiple_of is None
    assert var_49.precision is None
    assert var_49.coerce_types is True
    var_50 = [var_48, var_49]
    var_51 = module_3.Union(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Union'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.any_of).__module__}.{type(var_51.any_of).__qualname__}' == 'builtins.list'
    assert len(var_51.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_52 = module_0.to_json_schema(var_51)
    var_53 = 'anyOf'
    var_54 = var_52[var_53]
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.String'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert var_56.default == ''
    assert var_56.allow_blank is True
    assert var_56.trim_whitespace is True
    assert var_56.max_length is None
    assert var_56.min_length is None
    assert var_56.format is None
    assert var_56.coerce_types is True
    assert var_56.pattern is None
    assert var_56.pattern_regex is None
    var_57 = module_3.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.Integer'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert var_57.minimum is None
    assert var_57.maximum is None
    assert var_57.exclusive_minimum is None
    assert var_57.exclusive_maximum is None
    assert var_57.multiple_of is None
    assert var_57.precision is None
    assert var_57.coerce_types is True
    var_58 = [var_56, var_57]
    var_59 = module_1.OneOf(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.one_of).__module__}.{type(var_59.one_of).__qualname__}' == 'builtins.list'
    assert len(var_59.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_60 = module_0.to_json_schema(var_59)
    var_61 = 'oneOf'
    var_62 = var_60[var_61]
    var_63 = len(var_62)
    assert var_63 == 2
    var_64 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.String'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert var_64.default == ''
    assert var_64.allow_blank is True
    assert var_64.trim_whitespace is True
    assert var_64.max_length is None
    assert var_64.min_length is None
    assert var_64.format is None
    assert var_64.coerce_types is True
    assert var_64.pattern is None
    assert var_64.pattern_regex is None
    var_65 = module_3.Object(properties=var_6, pattern_properties=var_6, additional_properties=var_6, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_6)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Object'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.properties == {}
    assert var_65.pattern_properties == {}
    assert var_65.additional_properties is None
    assert var_65.property_names is None
    assert var_65.min_properties is None
    assert var_65.max_properties is None
    assert var_65.required == []
    var_66 = [var_64, var_65]
    var_67 = module_1.AllOf(var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    assert f'{type(var_67.all_of).__module__}.{type(var_67.all_of).__qualname__}' == 'builtins.list'
    assert len(var_67.all_of) == 2
    var_68 = module_0.to_json_schema(var_67)
    var_69 = 'allOf'
    var_70 = var_68[var_69]
    var_71 = len(var_70)
    assert var_71 == 2
    var_72 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.String'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert var_72.default == ''
    assert var_72.allow_blank is True
    assert var_72.trim_whitespace is True
    assert var_72.max_length is None
    assert var_72.min_length is None
    assert var_72.format is None
    assert var_72.coerce_types is True
    assert var_72.pattern is None
    assert var_72.pattern_regex is None
    var_73 = module_1.Not(var_72)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.composites.Not'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert f'{type(var_73.negated).__module__}.{type(var_73.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_74 = module_0.to_json_schema(var_73)
    var_75 = module_3.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
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
    var_76 = module_3.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
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
    var_77 = module_1.IfThenElse(var_75, var_76, var_6)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert f'{type(var_77.if_clause).__module__}.{type(var_77.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_77.then_clause).__module__}.{type(var_77.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_77.else_clause).__module__}.{type(var_77.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_78 = module_0.to_json_schema(var_77)
    var_79 = module_4.Definitions()
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_79) == 0
    var_80 = module_0.to_json_schema(var_79)

@pytest.mark.xfail(strict=True)
def test_case_51():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = 'enum'
    var_9 = 2
    var_10 = 3
    var_11 = [var_0, var_9, var_10]
    var_12 = {var_8: var_11}
    var_13 = module_0.from_json_schema(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Choice'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.choices == [(True, True), (2, 2), (3, 3)]
    assert var_13.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_14 = 'const'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = module_0.from_json_schema(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Const'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.const == 'value'
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_18 = 'a'
    var_19 = 'b'
    var_20 = [var_18, var_19]
    var_21 = {var_4: var_5, var_8: var_20}
    var_22 = module_0.from_json_schema(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.all_of).__module__}.{type(var_22.all_of).__qualname__}' == 'builtins.list'
    assert len(var_22.all_of) == 2
    var_23 = module_4.Definitions()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_23) == 0
    var_24 = '$ref'
    var_25 = '#/components/schemas/TestSchema'
    var_26 = {var_24: var_25}
    var_27 = module_0.from_json_schema(var_26, var_23)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.to == '#/components/schemas/TestSchema'
    assert f'{type(var_27.definitions).__module__}.{type(var_27.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_27.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_28 = 'allOf'
    var_29 = {var_4: var_5}
    var_30 = 'minLength'
    var_31 = {var_30: var_0}
    var_32 = [var_29, var_31]
    var_33 = {var_28: var_32}
    var_34 = module_0.from_json_schema(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.all_of).__module__}.{type(var_34.all_of).__qualname__}' == 'builtins.list'
    assert len(var_34.all_of) == 2
    var_35 = 'anyOf'
    var_36 = {var_4: var_5}
    var_37 = 'integer'
    var_38 = {var_4: var_37}
    var_39 = [var_36, var_38]
    var_40 = {var_35: var_39}
    var_41 = module_0.from_json_schema(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Union'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.any_of).__module__}.{type(var_41.any_of).__qualname__}' == 'builtins.list'
    assert len(var_41.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_42 = 'oneOf'
    var_43 = {var_4: var_5}
    var_44 = {var_4: var_37}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_0.from_json_schema(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert f'{type(var_47.one_of).__module__}.{type(var_47.one_of).__qualname__}' == 'builtins.list'
    assert len(var_47.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_48 = 'not'
    var_49 = {var_4: var_5}
    var_50 = {var_48: var_49}
    var_51 = module_0.from_json_schema(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.composites.Not'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.negated).__module__}.{type(var_51.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_52 = var_47.get_default_value()
    var_53 = {}
    var_54 = module_0.from_json_schema(var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Any'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    var_55 = 'components'
    var_56 = 'schemas'
    var_57 = 'StringType'
    var_58 = 'IntType'
    var_59 = {var_4: var_5}
    var_60 = {var_4: var_37}
    var_61 = {var_57: var_59, var_58: var_60}
    var_62 = {var_56: var_61}
    var_63 = {var_55: var_62}
    var_64 = module_0.from_json_schema(var_63)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.Any'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    var_65 = 'properties'
    var_66 = 'object'
    var_67 = 'name'
    var_68 = {var_4: var_5}
    var_69 = {var_67: var_68}
    var_70 = {var_4: var_66, var_65: var_69}
    var_71 = module_0.from_json_schema(var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Object'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert f'{type(var_71.properties).__module__}.{type(var_71.properties).__qualname__}' == 'builtins.dict'
    assert len(var_71.properties) == 1
    assert var_71.pattern_properties == {}
    assert var_71.additional_properties is None
    assert var_71.property_names is None
    assert var_71.min_properties is None
    assert var_71.max_properties is None
    assert var_71.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_72 = 'items'
    var_73 = 'array'
    var_74 = {var_4: var_5}
    var_75 = {var_4: var_73, var_72: var_74}
    var_76 = module_0.from_json_schema(var_75)
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.fields.Array'
    assert var_76.title == ''
    assert var_76.description == ''
    assert var_76.allow_null is False
    assert var_76.read_only is False
    assert f'{type(var_76.items).__module__}.{type(var_76.items).__qualname__}' == 'typesystem.fields.String'
    assert var_76.additional_items is True
    assert var_76.min_items == 0
    assert var_76.max_items is None
    assert var_76.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_77 = 'maxLength'
    var_78 = 'pattern'
    var_79 = 100
    var_80 = '^[a-z]+$'
    var_81 = {var_4: var_5, var_30: var_0, var_77: var_79, var_78: var_80}
    var_82 = module_0.from_json_schema(var_81)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.String'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert var_82.allow_blank is False
    assert var_82.trim_whitespace is True
    assert var_82.max_length == 100
    assert var_82.min_length is None
    assert var_82.format is None
    assert var_82.coerce_types is False
    assert var_82.pattern == '^[a-z]+$'
    assert f'{type(var_82.pattern_regex).__module__}.{type(var_82.pattern_regex).__qualname__}' == 're.Pattern'
    var_83 = 'maximum'
    var_84 = 'multipleOf'
    var_85 = 'number'
    var_86 = 5
    var_87 = {var_4: var_85, var_5: var_2, var_83: var_79, var_84: var_86}
    var_88 = module_0.from_json_schema(var_87)
    assert f'{type(var_88).__module__}.{type(var_88).__qualname__}' == 'typesystem.fields.Float'
    assert var_88.title == ''
    assert var_88.description == ''
    assert var_88.allow_null is False
    assert var_88.read_only is False
    assert var_88.minimum is None
    assert var_88.maximum == 100
    assert var_88.exclusive_minimum is None
    assert var_88.exclusive_maximum is None
    assert var_88.multiple_of == 5
    assert var_88.precision is None
    assert var_88.coerce_types is False
    var_89 = module_4.Definitions()
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_89) == 0
    var_90 = 'CustomRef'
    var_91 = {var_24: var_90}
    module_0.from_json_schema(var_91, var_89)

@pytest.mark.xfail(strict=True)
def test_case_52():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
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
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = 'enum'
    var_9 = 2
    var_10 = 3
    var_11 = [var_0, var_9, var_10]
    var_12 = {var_8: var_11}
    var_13 = module_0.from_json_schema(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Choice'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.choices == [(True, True), (2, 2), (3, 3)]
    assert var_13.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_14 = 'const'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = module_0.from_json_schema(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Const'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.const == 'value'
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_18 = 'a'
    var_19 = 'b'
    var_20 = [var_18, var_19]
    var_21 = {var_4: var_5, var_8: var_20}
    var_22 = module_0.from_json_schema(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.all_of).__module__}.{type(var_22.all_of).__qualname__}' == 'builtins.list'
    assert len(var_22.all_of) == 2
    var_23 = module_4.Definitions()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_23) == 0
    var_24 = '$ref'
    var_25 = '#/components/schemas/TestSchema'
    var_26 = {var_24: var_25}
    var_27 = module_0.from_json_schema(var_26, var_23)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.to == '#/components/schemas/TestSchema'
    assert f'{type(var_27.definitions).__module__}.{type(var_27.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_27.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_28 = 'allOf'
    var_29 = {var_4: var_5}
    var_30 = 'minLength'
    var_31 = {var_30: var_0}
    var_32 = [var_29, var_31]
    var_33 = {var_28: var_32}
    var_34 = module_0.from_json_schema(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.all_of).__module__}.{type(var_34.all_of).__qualname__}' == 'builtins.list'
    assert len(var_34.all_of) == 2
    var_35 = 'anyOf'
    var_36 = {var_4: var_5}
    var_37 = 'integer'
    var_38 = {var_4: var_37}
    var_39 = [var_36, var_38]
    var_40 = {var_35: var_39}
    var_41 = module_0.from_json_schema(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Union'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.any_of).__module__}.{type(var_41.any_of).__qualname__}' == 'builtins.list'
    assert len(var_41.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_42 = 'oneOf'
    var_43 = {var_4: var_5}
    var_44 = {var_4: var_37}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_0.from_json_schema(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert f'{type(var_47.one_of).__module__}.{type(var_47.one_of).__qualname__}' == 'builtins.list'
    assert len(var_47.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_48 = 'not'
    var_49 = {var_4: var_5}
    var_50 = {var_48: var_49}
    var_51 = module_0.from_json_schema(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.composites.Not'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.negated).__module__}.{type(var_51.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_52 = 'if'
    var_53 = 'then'
    var_54 = 'else'
    var_55 = {var_4: var_5}
    var_56 = {var_30: var_0}
    var_57 = {var_4: var_37}
    var_58 = {var_52: var_55, var_53: var_56, var_54: var_57}
    var_59 = module_0.from_json_schema(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.if_clause).__module__}.{type(var_59.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_59.then_clause).__module__}.{type(var_59.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_59.else_clause).__module__}.{type(var_59.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_60 = {}
    var_61 = module_0.from_json_schema(var_60)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.fields.Any'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    var_62 = 'components'
    var_63 = 'schemas'
    var_64 = 'StringType'
    var_65 = 'IntType'
    var_66 = {var_4: var_5}
    var_67 = {var_4: var_37}
    var_68 = {var_64: var_66, var_65: var_67}
    var_69 = {var_63: var_68}
    var_70 = {var_62: var_69}
    var_71 = module_0.from_json_schema(var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Any'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    var_72 = 'properties'
    var_73 = 'object'
    var_74 = 'name'
    var_75 = {var_4: var_5}
    var_76 = {var_74: var_75}
    var_77 = {var_4: var_73, var_72: var_76}
    var_78 = module_0.from_json_schema(var_77)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.Object'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert f'{type(var_78.properties).__module__}.{type(var_78.properties).__qualname__}' == 'builtins.dict'
    assert len(var_78.properties) == 1
    assert var_78.pattern_properties == {}
    assert var_78.additional_properties is None
    assert var_78.property_names is None
    assert var_78.min_properties is None
    assert var_78.max_properties is None
    assert var_78.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_79 = 'items'
    var_80 = 'array'
    var_81 = {var_4: var_5}
    var_82 = {var_4: var_80, var_79: var_81}
    var_83 = module_0.from_json_schema(var_82)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.fields.Array'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert f'{type(var_83.items).__module__}.{type(var_83.items).__qualname__}' == 'typesystem.fields.String'
    assert var_83.additional_items is True
    assert var_83.min_items == 0
    assert var_83.max_items is None
    assert var_83.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_84 = 'maxLength'
    var_85 = 'pattern'
    var_86 = 100
    var_87 = '^[a-z]+$'
    var_88 = {var_4: var_5, var_30: var_0, var_84: var_86, var_85: var_87}
    var_89 = module_0.from_json_schema(var_88)
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.fields.String'
    assert var_89.title == ''
    assert var_89.description == ''
    assert var_89.allow_null is False
    assert var_89.read_only is False
    assert var_89.allow_blank is False
    assert var_89.trim_whitespace is True
    assert var_89.max_length == 100
    assert var_89.min_length is None
    assert var_89.format is None
    assert var_89.coerce_types is False
    assert var_89.pattern == '^[a-z]+$'
    assert f'{type(var_89.pattern_regex).__module__}.{type(var_89.pattern_regex).__qualname__}' == 're.Pattern'
    var_90 = 'minimum'
    var_91 = 'maximum'
    var_92 = 'multipleOf'
    var_93 = 'number'
    var_94 = 5
    var_95 = {var_4: var_93, var_90: var_2, var_91: var_86, var_92: var_94}
    var_96 = module_0.from_json_schema(var_95)
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.fields.Float'
    assert var_96.title == ''
    assert var_96.description == ''
    assert var_96.allow_null is False
    assert var_96.read_only is False
    assert var_96.minimum is False
    assert var_96.maximum == 100
    assert var_96.exclusive_minimum is None
    assert var_96.exclusive_maximum is None
    assert var_96.multiple_of == 5
    assert var_96.precision is None
    assert var_96.coerce_types is False
    var_97 = module_4.Definitions()
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_97) == 0
    var_98 = 'CustomRef'
    var_99 = {var_24: var_98}
    module_0.from_json_schema(var_99, var_97)

def test_case_53():
    var_0 = module_3.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
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
    assert module_0.TYPE_CONSTRAINTS == {'exclusiveMinimum', 'properties', 'maxLength', 'items', 'maximum', 'contains', 'dependencies', 'pattern', 'maxProperties', 'multipleOf', 'uniqueItems', 'exclusiveMaximum', 'maxItems', 'patternProperties', 'minimum', 'minItems', 'minProperties', 'additionalProperties', 'propertyNames', 'required', 'boolean_schema', 'type', 'additionalItems', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = False
    var_4 = 1
    var_5 = 10
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = module_3.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length == 10
    assert var_8.min_length == 1
    assert var_8.format == 'email'
    assert var_8.coerce_types is True
    assert var_8.pattern == '^[a-z]+$'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = True
    var_11 = module_3.String()
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
    var_12 = module_0.to_json_schema(var_11)
    var_13 = True
    var_14 = module_3.String(allow_blank=var_13)
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
    var_15 = module_0.to_json_schema(var_14)
    var_16 = 100
    var_17 = 5
    var_18 = module_3.Integer(minimum=var_3, maximum=var_16, exclusive_minimum=var_17, exclusive_maximum=var_4, multiple_of=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Integer'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.minimum is False
    assert var_18.maximum == 100
    assert var_18.exclusive_minimum == 5
    assert var_18.exclusive_maximum == 1
    assert var_18.multiple_of == 5
    assert var_18.precision is None
    assert var_18.coerce_types is True
    var_19 = module_0.to_json_schema(var_18)
    var_20 = module_3.Float(minimum=var_3, maximum=var_13)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Float'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.minimum is False
    assert var_20.maximum is True
    assert var_20.exclusive_minimum is None
    assert var_20.exclusive_maximum is None
    assert var_20.multiple_of is None
    assert var_20.precision is None
    assert var_20.coerce_types is True
    var_21 = module_0.to_json_schema(var_20)
    var_22 = module_3.Boolean()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.coerce_types is True
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_23 = module_0.to_json_schema(var_22)
    var_24 = True
    var_25 = module_3.Boolean()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.coerce_types is True
    var_26 = module_0.to_json_schema(var_25)
    var_27 = module_3.String()
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
    var_28 = True
    var_29 = module_3.Array(var_27, min_items=var_24, max_items=var_5, unique_items=var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Array'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.items).__module__}.{type(var_29.items).__qualname__}' == 'typesystem.fields.String'
    assert var_29.additional_items is False
    assert var_29.min_items is True
    assert var_29.max_items == 10
    assert var_29.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_30 = module_0.to_json_schema(var_29)
    var_31 = module_3.String()
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
    var_32 = module_3.Integer()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Integer'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.minimum is None
    assert var_32.maximum is None
    assert var_32.exclusive_minimum is None
    assert var_32.exclusive_maximum is None
    assert var_32.multiple_of is None
    assert var_32.precision is None
    assert var_32.coerce_types is True
    var_33 = [var_31, var_32]
    var_34 = module_3.Array(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Array'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.items).__module__}.{type(var_34.items).__qualname__}' == 'builtins.list'
    assert len(var_34.items) == 2
    assert var_34.additional_items is False
    assert var_34.min_items == 2
    assert var_34.max_items == 2
    assert var_34.unique_items is False
    var_35 = module_0.to_json_schema(var_34)
    var_36 = 'items'
    var_37 = var_35[var_36]
    var_38 = var_35[var_36]
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = module_3.Array(additional_items=var_3)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Array'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert var_40.items is None
    assert var_40.additional_items is False
    assert var_40.min_items is None
    assert var_40.max_items is None
    assert var_40.unique_items is False
    var_41 = module_0.to_json_schema(var_40)
    var_42 = module_3.String()
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.String'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert var_42.allow_blank is False
    assert var_42.trim_whitespace is True
    assert var_42.max_length is None
    assert var_42.min_length is None
    assert var_42.format is None
    assert var_42.coerce_types is True
    assert var_42.pattern is None
    assert var_42.pattern_regex is None
    var_43 = module_3.Array(additional_items=var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Array'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.items is None
    assert f'{type(var_43.additional_items).__module__}.{type(var_43.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_43.min_items is None
    assert var_43.max_items is None
    assert var_43.unique_items is False
    var_44 = module_0.to_json_schema(var_43)
    var_45 = 'additionalItems'
    var_46 = var_44[var_45]
    var_47 = 'name'
    var_48 = 'age'
    var_49 = module_3.String()
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.fields.String'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert var_49.allow_blank is False
    assert var_49.trim_whitespace is True
    assert var_49.max_length is None
    assert var_49.min_length is None
    assert var_49.format is None
    assert var_49.coerce_types is True
    assert var_49.pattern is None
    assert var_49.pattern_regex is None
    var_50 = module_3.Integer()
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.fields.Integer'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert var_50.minimum is None
    assert var_50.maximum is None
    assert var_50.exclusive_minimum is None
    assert var_50.exclusive_maximum is None
    assert var_50.multiple_of is None
    assert var_50.precision is None
    assert var_50.coerce_types is True
    var_51 = {var_47: var_49, var_48: var_50}
    var_52 = [var_47]
    var_53 = module_3.Object(properties=var_51, min_properties=var_28, max_properties=var_5, required=var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.Object'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert f'{type(var_53.properties).__module__}.{type(var_53.properties).__qualname__}' == 'builtins.dict'
    assert len(var_53.properties) == 2
    assert var_53.pattern_properties == {}
    assert var_53.additional_properties is True
    assert var_53.property_names is None
    assert var_53.min_properties is True
    assert var_53.max_properties == 10
    assert var_53.required == ['name']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_54 = module_0.to_json_schema(var_53)
    var_55 = '^S_'
    var_56 = module_3.String()
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.String'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert var_56.allow_blank is False
    assert var_56.trim_whitespace is True
    assert var_56.max_length is None
    assert var_56.min_length is None
    assert var_56.format is None
    assert var_56.coerce_types is True
    assert var_56.pattern is None
    assert var_56.pattern_regex is None
    var_57 = {var_55: var_56}
    var_58 = module_3.Object(pattern_properties=var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.Object'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.properties == {}
    assert f'{type(var_58.pattern_properties).__module__}.{type(var_58.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_58.pattern_properties) == 1
    assert var_58.additional_properties is True
    assert var_58.property_names is None
    assert var_58.min_properties is None
    assert var_58.max_properties is None
    assert var_58.required == []
    var_59 = module_0.to_json_schema(var_58)
    var_60 = module_3.Object(additional_properties=var_3)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.Object'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.properties == {}
    assert var_60.pattern_properties == {}
    assert var_60.additional_properties is False
    assert var_60.property_names is None
    assert var_60.min_properties is None
    assert var_60.max_properties is None
    assert var_60.required == []
    var_61 = module_0.to_json_schema(var_60)
    var_62 = module_3.String()
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
    var_63 = module_3.Object(additional_properties=var_62)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.fields.Object'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert var_63.properties == {}
    assert var_63.pattern_properties == {}
    assert f'{type(var_63.additional_properties).__module__}.{type(var_63.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_63.property_names is None
    assert var_63.min_properties is None
    assert var_63.max_properties is None
    assert var_63.required == []
    var_64 = module_0.to_json_schema(var_63)
    var_65 = 'additionalProperties'
    var_66 = var_64[var_65]
    var_67 = module_3.String(pattern=var_6)
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
    assert var_67.pattern == '^[a-z]+$'
    assert f'{type(var_67.pattern_regex).__module__}.{type(var_67.pattern_regex).__qualname__}' == 're.Pattern'
    var_68 = module_3.Object(property_names=var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.fields.Object'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert var_68.properties == {}
    assert var_68.pattern_properties == {}
    assert var_68.additional_properties is True
    assert f'{type(var_68.property_names).__module__}.{type(var_68.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_68.min_properties is None
    assert var_68.max_properties is None
    assert var_68.required == []
    var_69 = module_0.to_json_schema(var_68)
    var_70 = 'a'
    var_71 = 'Option A'
    var_72 = (var_70, var_71)
    var_73 = 'b'
    var_74 = 'Option B'
    var_75 = (var_73, var_74)
    var_76 = [var_72, var_75]
    var_77 = module_3.Choice(choices=var_76)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.Choice'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert var_77.choices == [('a', 'Option A'), ('b', 'Option B')]
    assert var_77.coerce_types is True
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_78 = module_0.to_json_schema(var_77)
    var_79 = 'constant_value'
    var_80 = module_3.Const(var_79)
    assert f'{type(var_80).__module__}.{type(var_80).__qualname__}' == 'typesystem.fields.Const'
    assert var_80.title == ''
    assert var_80.description == ''
    assert var_80.allow_null is False
    assert var_80.read_only is False
    assert var_80.const == 'constant_value'
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_81 = module_0.to_json_schema(var_80)
    var_82 = module_3.String()
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.String'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert var_82.allow_blank is False
    assert var_82.trim_whitespace is True
    assert var_82.max_length is None
    assert var_82.min_length is None
    assert var_82.format is None
    assert var_82.coerce_types is True
    assert var_82.pattern is None
    assert var_82.pattern_regex is None
    var_83 = module_3.Integer()
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.fields.Integer'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert var_83.minimum is None
    assert var_83.maximum is None
    assert var_83.exclusive_minimum is None
    assert var_83.exclusive_maximum is None
    assert var_83.multiple_of is None
    assert var_83.precision is None
    assert var_83.coerce_types is True
    var_84 = [var_82, var_83]
    var_85 = module_3.Union(var_84)
    assert f'{type(var_85).__module__}.{type(var_85).__qualname__}' == 'typesystem.fields.Union'
    assert var_85.title == ''
    assert var_85.description == ''
    assert var_85.allow_null is False
    assert var_85.read_only is False
    assert f'{type(var_85.any_of).__module__}.{type(var_85.any_of).__qualname__}' == 'builtins.list'
    assert len(var_85.any_of) == 2
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_86 = module_0.to_json_schema(var_85)
    var_87 = 'anyOf'
    var_88 = var_86[var_87]
    var_89 = len(var_88)
    assert var_89 == 2
    var_90 = module_3.String()
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'typesystem.fields.String'
    assert var_90.title == ''
    assert var_90.description == ''
    assert var_90.allow_null is False
    assert var_90.read_only is False
    assert var_90.allow_blank is False
    assert var_90.trim_whitespace is True
    assert var_90.max_length is None
    assert var_90.min_length is None
    assert var_90.format is None
    assert var_90.coerce_types is True
    assert var_90.pattern is None
    assert var_90.pattern_regex is None
    var_91 = module_3.Integer()
    assert f'{type(var_91).__module__}.{type(var_91).__qualname__}' == 'typesystem.fields.Integer'
    assert var_91.title == ''
    assert var_91.description == ''
    assert var_91.allow_null is False
    assert var_91.read_only is False
    assert var_91.minimum is None
    assert var_91.maximum is None
    assert var_91.exclusive_minimum is None
    assert var_91.exclusive_maximum is None
    assert var_91.multiple_of is None
    assert var_91.precision is None
    assert var_91.coerce_types is True
    var_92 = [var_90, var_91]
    var_93 = module_1.OneOf(var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert f'{type(var_93.one_of).__module__}.{type(var_93.one_of).__qualname__}' == 'builtins.list'
    assert len(var_93.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_94 = module_0.to_json_schema(var_93)
    var_95 = 'oneOf'
    var_96 = var_94[var_95]
    var_97 = len(var_96)
    assert var_97 == 2
    var_98 = module_3.String()
    assert f'{type(var_98).__module__}.{type(var_98).__qualname__}' == 'typesystem.fields.String'
    assert var_98.title == ''
    assert var_98.description == ''
    assert var_98.allow_null is False
    assert var_98.read_only is False
    assert var_98.allow_blank is False
    assert var_98.trim_whitespace is True
    assert var_98.max_length is None
    assert var_98.min_length is None
    assert var_98.format is None
    assert var_98.coerce_types is True
    assert var_98.pattern is None
    assert var_98.pattern_regex is None
    var_99 = 'A'
    var_100 = (var_70, var_99)
    var_101 = 'B'
    var_102 = (var_73, var_101)
    var_103 = [var_100, var_102]
    var_104 = module_3.Choice(choices=var_103)
    assert f'{type(var_104).__module__}.{type(var_104).__qualname__}' == 'typesystem.fields.Choice'
    assert var_104.title == ''
    assert var_104.description == ''
    assert var_104.allow_null is False
    assert var_104.read_only is False
    assert var_104.choices == [('a', 'A'), ('b', 'B')]
    assert var_104.coerce_types is True
    var_105 = [var_98, var_104]
    var_106 = module_1.AllOf(var_105)
    assert f'{type(var_106).__module__}.{type(var_106).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_106.title == ''
    assert var_106.description == ''
    assert var_106.allow_null is False
    assert var_106.read_only is False
    assert f'{type(var_106.all_of).__module__}.{type(var_106.all_of).__qualname__}' == 'builtins.list'
    assert len(var_106.all_of) == 2
    var_107 = module_0.to_json_schema(var_106)
    var_108 = 'allOf'
    var_109 = var_107[var_108]
    var_110 = len(var_109)
    assert var_110 == 2
    var_111 = module_3.String()
    assert f'{type(var_111).__module__}.{type(var_111).__qualname__}' == 'typesystem.fields.String'
    assert var_111.title == ''
    assert var_111.description == ''
    assert var_111.allow_null is False
    assert var_111.read_only is False
    assert var_111.allow_blank is False
    assert var_111.trim_whitespace is True
    assert var_111.max_length is None
    assert var_111.min_length is None
    assert var_111.format is None
    assert var_111.coerce_types is True
    assert var_111.pattern is None
    assert var_111.pattern_regex is None
    var_112 = module_1.Not(var_111)
    assert f'{type(var_112).__module__}.{type(var_112).__qualname__}' == 'typesystem.composites.Not'
    assert var_112.title == ''
    assert var_112.description == ''
    assert var_112.allow_null is False
    assert var_112.read_only is False
    assert f'{type(var_112.negated).__module__}.{type(var_112.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_113 = module_0.to_json_schema(var_112)
    var_114 = (var_70, var_99)
    var_115 = [var_114]
    var_116 = module_3.Choice(choices=var_115)
    assert f'{type(var_116).__module__}.{type(var_116).__qualname__}' == 'typesystem.fields.Choice'
    assert var_116.title == ''
    assert var_116.description == ''
    assert var_116.allow_null is False
    assert var_116.read_only is False
    assert var_116.choices == [('a', 'A')]
    assert var_116.coerce_types is True
    var_117 = module_3.String()
    assert f'{type(var_117).__module__}.{type(var_117).__qualname__}' == 'typesystem.fields.String'
    assert var_117.title == ''
    assert var_117.description == ''
    assert var_117.allow_null is False
    assert var_117.read_only is False
    assert var_117.allow_blank is False
    assert var_117.trim_whitespace is True
    assert var_117.max_length is None
    assert var_117.min_length is None
    assert var_117.format is None
    assert var_117.coerce_types is True
    assert var_117.pattern is None
    assert var_117.pattern_regex is None
    var_118 = module_3.Integer()
    assert f'{type(var_118).__module__}.{type(var_118).__qualname__}' == 'typesystem.fields.Integer'
    assert var_118.title == ''
    assert var_118.description == ''
    assert var_118.allow_null is False
    assert var_118.read_only is False
    assert var_118.minimum is None
    assert var_118.maximum is None
    assert var_118.exclusive_minimum is None
    assert var_118.exclusive_maximum is None
    assert var_118.multiple_of is None
    assert var_118.precision is None
    assert var_118.coerce_types is True
    var_119 = module_1.IfThenElse(var_116, var_117, var_118)
    assert f'{type(var_119).__module__}.{type(var_119).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_119.title == ''
    assert var_119.description == ''
    assert var_119.allow_null is False
    assert var_119.read_only is False
    assert f'{type(var_119.if_clause).__module__}.{type(var_119.if_clause).__qualname__}' == 'typesystem.fields.Choice'
    assert f'{type(var_119.then_clause).__module__}.{type(var_119.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_119.else_clause).__module__}.{type(var_119.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_120 = module_0.to_json_schema(var_119)
    var_121 = (var_70, var_99)
    var_122 = [var_121]
    var_123 = module_3.Choice(choices=var_122)
    assert f'{type(var_123).__module__}.{type(var_123).__qualname__}' == 'typesystem.fields.Choice'
    assert var_123.title == ''
    assert var_123.description == ''
    assert var_123.allow_null is False
    assert var_123.read_only is False
    assert var_123.choices == [('a', 'A')]
    assert var_123.coerce_types is True
    var_124 = module_3.String()
    assert f'{type(var_124).__module__}.{type(var_124).__qualname__}' == 'typesystem.fields.String'
    assert var_124.title == ''
    assert var_124.description == ''
    assert var_124.allow_null is False
    assert var_124.read_only is False
    assert var_124.allow_blank is False
    assert var_124.trim_whitespace is True
    assert var_124.max_length is None
    assert var_124.min_length is None
    assert var_124.format is None
    assert var_124.coerce_types is True
    assert var_124.pattern is None
    assert var_124.pattern_regex is None
    var_125 = module_1.IfThenElse(var_123, var_124)
    assert f'{type(var_125).__module__}.{type(var_125).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_125.title == ''
    assert var_125.description == ''
    assert var_125.allow_null is False
    assert var_125.read_only is False
    assert f'{type(var_125.if_clause).__module__}.{type(var_125.if_clause).__qualname__}' == 'typesystem.fields.Choice'
    assert f'{type(var_125.then_clause).__module__}.{type(var_125.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_125.else_clause).__module__}.{type(var_125.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_126 = module_0.to_json_schema(var_125)
    var_127 = 'User'
    var_128 = module_3.String()
    assert f'{type(var_128).__module__}.{type(var_128).__qualname__}' == 'typesystem.fields.String'
    assert var_128.title == ''
    assert var_128.description == ''
    assert var_128.allow_null is False
    assert var_128.read_only is False
    assert var_128.allow_blank is False
    assert var_128.trim_whitespace is True
    assert var_128.max_length is None
    assert var_128.min_length is None
    assert var_128.format is None
    assert var_128.coerce_types is True
    assert var_128.pattern is None
    assert var_128.pattern_regex is None
    var_129 = {var_47: var_128}
    var_130 = module_3.Object(properties=var_129)
    assert f'{type(var_130).__module__}.{type(var_130).__qualname__}' == 'typesystem.fields.Object'
    assert var_130.title == ''
    assert var_130.description == ''
    assert var_130.allow_null is False
    assert var_130.read_only is False
    assert f'{type(var_130.properties).__module__}.{type(var_130.properties).__qualname__}' == 'builtins.dict'
    assert len(var_130.properties) == 1
    assert var_130.pattern_properties == {}
    assert var_130.additional_properties is True
    assert var_130.property_names is None
    assert var_130.min_properties is None
    assert var_130.max_properties is None
    assert var_130.required == []
    var_131 = {var_127: var_130}
    var_132 = module_4.Reference(var_127, var_131)
    assert f'{type(var_132).__module__}.{type(var_132).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_132.title == ''
    assert var_132.description == ''
    assert var_132.allow_null is False
    assert var_132.read_only is False
    assert var_132.to == 'User'
    assert f'{type(var_132.definitions).__module__}.{type(var_132.definitions).__qualname__}' == 'builtins.dict'
    assert len(var_132.definitions) == 1
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_133 = module_0.to_json_schema(var_132)