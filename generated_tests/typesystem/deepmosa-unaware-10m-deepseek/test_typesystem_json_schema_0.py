# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import enum as module_0
import typesystem.json_schema as module_1
import typesystem.fields as module_2
import typesystem.composites as module_3
import typesystem.schemas as module_4
import re as module_5

def test_case_0():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2

def test_case_1():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.type_from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is True
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 5
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = True
    module_1.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_1.ref_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_1.all_of_from_json_schema(var_0, var_0)

def test_case_5():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.type_from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is True
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 5
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = module_3.Not(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_1.to_json_schema(var_2)
    var_4 = module_1.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.Not'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.negated).__module__}.{type(var_4.negated).__qualname__}' == 'typesystem.fields.Union'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_1.one_of_from_json_schema(var_0, var_0)

def test_case_7():
    var_0 = None
    var_1 = {}
    var_2 = 'b[r?>/26Gvg+q8'
    var_3 = True
    with pytest.raises(AssertionError):
        module_1.from_json_schema_type(var_1, var_2, var_3, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_1.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = {}
    module_1.not_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_1.const_from_json_schema(var_0, var_0)

def test_case_11():
    var_0 = True
    var_1 = module_1.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = ''
    var_3 = {var_2: var_2}
    var_4 = module_1.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = True
    module_1.to_json_schema(var_0, var_0)

def test_case_13():
    var_0 = module_2.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Choice'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.choices == []
    assert var_0.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_1 = module_1.to_json_schema(var_0)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2

def test_case_14():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_4.Schema(var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.fields).__module__}.{type(var_1.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.fields) == 0
    assert var_1.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = module_1.from_json_schema(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_3 = module_3.IfThenElse(var_1, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.if_clause).__module__}.{type(var_3.if_clause).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_4 = module_1.to_json_schema(var_3)

def test_case_15():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.type_from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is True
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 5
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = module_1.to_json_schema(var_1)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7

def test_case_16():
    var_0 = None
    var_1 = module_0._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_3.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_3.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_1.get_standard_properties(var_2)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_5 = module_1.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    var_6 = module_1.to_json_schema(var_3, var_0)

def test_case_17():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_1.to_json_schema(var_0)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_3.Not(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.Not'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.negated).__module__}.{type(var_1.negated).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.negated) == 0
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    module_1.to_json_schema(var_1)

def test_case_19():
    var_0 = module_2.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Choice'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.choices == []
    assert var_0.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_1 = module_1.to_json_schema(var_0)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = module_1.from_json_schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Choice'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.choices == []
    assert var_2.coerce_types is True
    var_3 = module_1.to_json_schema(var_2)

def test_case_20():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_4.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.fields).__module__}.{type(var_1.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.fields) == 0
    assert var_1.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = module_1.to_json_schema(var_1)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2

def test_case_21():
    var_0 = None
    var_1 = module_0._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_4.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.fields).__module__}.{type(var_2.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.fields) == 0
    assert var_2.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_3.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.schemas.Schema'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_1.get_standard_properties(var_2)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_5 = module_1.from_json_schema(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    var_6 = module_1.to_json_schema(var_3)
    var_7 = module_2.Const(var_5, **var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Const'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.const).__module__}.{type(var_7.const).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_8 = module_1.to_json_schema(var_7, var_0)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = module_0._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_4.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to is None
    assert f'{type(var_2.definitions).__module__}.{type(var_2.definitions).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
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
    var_4 = module_3.Not(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.Not'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.negated).__module__}.{type(var_4.negated).__qualname__}' == 'typesystem.schemas.Reference'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_5 = module_0._EnumDict()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'enum._EnumDict'
    assert len(var_5) == 0
    var_6 = module_4.Schema(var_1, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_6.fields) == 0
    assert var_6.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_1.from_json_schema(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Any'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    module_1.to_json_schema(var_4)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = module_0._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_3.OneOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.one_of).__module__}.{type(var_2.one_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.one_of) == 0
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_3 = module_3.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.composites.OneOf'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_4 = var_3.serialize(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'enum._EnumDict'
    assert len(var_4) == 0
    var_5 = module_1.from_json_schema(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_6 = module_1.to_json_schema(var_3)
    var_7 = module_5.purge()
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
    var_8 = module_1.from_json_schema(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.Not'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.negated).__module__}.{type(var_8.negated).__qualname__}' == 'typesystem.composites.OneOf'
    module_1.to_json_schema(var_0)

def test_case_24():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_3.AllOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.all_of).__module__}.{type(var_1.all_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.all_of) == 0
    var_2 = module_3.Not(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.composites.AllOf'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_4.Schema(var_0, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_3.fields) == 0
    assert var_3.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_1.to_json_schema(var_2)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_5 = module_1.to_json_schema(var_3)

def test_case_25():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_3.AllOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.all_of).__module__}.{type(var_1.all_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.all_of) == 0
    var_2 = module_3.Not(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.composites.AllOf'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_4.Schema(var_0, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_3.fields) == 0
    assert var_3.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_1.from_json_schema(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_5 = module_1.to_json_schema(var_1)
    var_6 = module_1.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.all_of == []
    var_7 = module_1.to_json_schema(var_2)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = module_0._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_2.Const(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Const'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.const is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_3 = module_3.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.fields.Const'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_4.Schema(var_1, **var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_4.fields) == 0
    assert var_4.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_1.from_json_schema(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_6 = module_1.to_json_schema(var_3)
    var_7 = module_5.purge()
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
    var_8 = module_1.from_json_schema(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.Not'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.negated).__module__}.{type(var_8.negated).__qualname__}' == 'typesystem.fields.Const'
    var_9 = module_1.to_json_schema(var_4)
    module_1.to_json_schema(var_7)

def test_case_27():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_4.Schema(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.fields).__module__}.{type(var_2.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.fields) == 0
    assert var_2.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_1.from_json_schema(var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_4 = module_1.to_json_schema(var_2, var_1)
    var_5 = module_5.purge()
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
    var_6 = module_1.from_json_schema(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Any'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    var_7 = module_2.Array(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Array'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.items is None
    assert var_7.additional_items is False
    assert var_7.min_items is None
    assert var_7.max_items is None
    assert var_7.unique_items is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_8 = module_1.to_json_schema(var_7)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = module_0._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_1.type_from_json_schema(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 5
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_3.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_4 = 'py\\O.]HP,Gn\t0u;Lj'
    var_5 = '*F=dQ\nGm9'
    var_6 = {var_4: var_3, var_5: var_2}
    var_7 = module_4.Schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 2
    assert var_7.required == ['py\\O.]HP,Gn\t0u;Lj']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = var_3.get_default_value()
    var_9 = module_1.from_json_schema(var_6, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Any'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    var_10 = module_1.to_json_schema(var_7)
    var_11 = module_1.from_json_schema(var_10, var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Object'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.properties).__module__}.{type(var_11.properties).__qualname__}' == 'builtins.dict'
    assert len(var_11.properties) == 2
    assert var_11.pattern_properties == {}
    assert var_11.additional_properties is None
    assert var_11.property_names is None
    assert var_11.min_properties is None
    assert var_11.max_properties is None
    assert var_11.required == ['py\\O.]HP,Gn\t0u;Lj']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    module_1.to_json_schema(var_8)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = None
    var_1 = module_0._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_3.Not(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.negated is None
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_3 = {var_2: var_2, var_2: var_2}
    var_4 = module_4.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert f'{type(var_4.required).__module__}.{type(var_4.required).__qualname__}' == 'builtins.list'
    assert len(var_4.required) == 1
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_1.from_json_schema(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_6 = module_3.IfThenElse(var_2, else_clause=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.if_clause).__module__}.{type(var_6.if_clause).__qualname__}' == 'typesystem.composites.Not'
    assert f'{type(var_6.then_clause).__module__}.{type(var_6.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_6.else_clause).__module__}.{type(var_6.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_1.to_json_schema(var_4)

def test_case_30():
    var_0 = None
    var_1 = module_0._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_1.type_from_json_schema(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 5
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_3.Not(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.negated is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_4 = ''
    var_5 = {var_4: var_3, var_4: var_2}
    var_6 = module_4.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_1.from_json_schema(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Any'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    var_8 = module_1.to_json_schema(var_6)

def test_case_31():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = module_1.to_json_schema(var_0)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_3 = '#/definitions/Address'
    var_4 = {var_1: var_3}
    var_5 = module_1.ref_from_json_schema(var_4, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.to == '#/definitions/Address'
    assert f'{type(var_5.definitions).__module__}.{type(var_5.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_6 = 'http://example.com/schema.json'
    var_7 = {var_1: var_6}
    with pytest.raises(AssertionError):
        module_1.ref_from_json_schema(var_7, var_0)

def test_case_32():
    var_0 = '$ref'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_0}
    with pytest.raises(AssertionError):
        module_1.ref_from_json_schema(var_2, var_1)

def test_case_33():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'enum'
    var_2 = 'red'
    var_3 = 'green'
    var_4 = 'blue'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = module_1.enum_from_json_schema(var_6, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Choice'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.choices == [('red', 'red'), ('green', 'green'), ('blue', 'blue')]
    assert var_7.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_8 = 1
    var_9 = 2
    var_10 = [var_8, var_9, var_8]
    var_11 = {var_1: var_10}
    var_12 = module_1.enum_from_json_schema(var_11, var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Choice'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.choices == [(1, 1), (2, 2), (1, 1)]
    assert var_12.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_13 = 'default'
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 'c'
    var_17 = [var_14, var_15, var_16]
    var_18 = {var_1: var_17, var_13: var_15}
    var_19 = module_1.enum_from_json_schema(var_18, var_0)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Choice'
    assert var_19.default == 'b'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_19.coerce_types is True
    var_20 = 'key'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = [var_8, var_9, var_8]
    var_24 = True
    var_25 = [var_22, var_23, var_24]
    var_26 = {var_1: var_25}
    var_27 = module_1.enum_from_json_schema(var_26, var_0)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Choice'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.choices == [({'key': 'value'}, {'key': 'value'}), ([1, 2, 1], [1, 2, 1]), (True, True)]
    assert var_27.coerce_types is True
    var_28 = 'only'
    var_29 = [var_28]
    var_30 = {var_1: var_29}
    var_31 = module_1.enum_from_json_schema(var_30, var_0)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Choice'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.choices == [('only', 'only')]
    assert var_31.coerce_types is True

@pytest.mark.xfail(strict=True)
def test_case_34():
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
    var_9 = module_1.one_of_from_json_schema(var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.one_of).__module__}.{type(var_9.one_of).__qualname__}' == 'builtins.list'
    assert len(var_9.one_of) == 2
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.one_of[var_12]
    var_14 = 1
    var_15 = var_9.one_of[var_14]
    var_16 = 'default'
    var_17 = {var_2: var_3}
    var_18 = {var_2: var_5}
    var_19 = [var_17, var_18]
    var_20 = 'test'
    var_21 = {var_1: var_19, var_16: var_20}
    var_22 = module_1.one_of_from_json_schema(var_21, var_0)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_22.default == 'test'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.one_of).__module__}.{type(var_22.one_of).__qualname__}' == 'builtins.list'
    assert len(var_22.one_of) == 2
    var_23 = 'properties'
    var_24 = 'object'
    var_25 = 'name'
    var_26 = {var_2: var_3}
    var_27 = {var_25: var_26}
    var_28 = {var_2: var_24, var_23: var_27}
    var_29 = 'items'
    var_30 = 'array'
    var_31 = {var_2: var_23}
    var_32 = {var_2: var_30, var_29: var_31}
    var_33 = [var_28, var_32]
    var_34 = {var_1: var_33}
    module_1.one_of_from_json_schema(var_34, var_0)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_1.to_json_schema(var_0)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = 'f'
    var_3 = {var_2: var_2}
    var_4 = module_1.type_from_json_schema(var_1, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is True
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_1.from_json_schema(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_6 = module_1.to_json_schema(var_4, var_0)
    var_7 = var_5.serialize(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Any'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    var_8 = module_1.from_json_schema(var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Union'
    assert var_8.default is None
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.any_of).__module__}.{type(var_8.any_of).__qualname__}' == 'builtins.list'
    assert len(var_8.any_of) == 5
    var_9 = module_3.IfThenElse(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.if_clause).__module__}.{type(var_9.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_9.then_clause).__module__}.{type(var_9.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_9.else_clause).__module__}.{type(var_9.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_10 = module_1.to_json_schema(var_9)
    var_11 = module_1.from_json_schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.if_clause).__module__}.{type(var_11.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_11.then_clause).__module__}.{type(var_11.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_11.else_clause).__module__}.{type(var_11.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_12 = var_3.get(var_8)
    module_4.Reference(var_10, var_12, **var_6)

def test_case_36():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_1: var_5, var_2: var_5}
    var_7 = module_1.if_then_else_from_json_schema(var_6, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.if_clause).__module__}.{type(var_7.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_7.then_clause).__module__}.{type(var_7.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_7.else_clause).__module__}.{type(var_7.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_8 = var_7.if_clause
    var_9 = var_7.then_clause

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == '#/components/schemas/User'
    assert f'{type(var_4.definitions).__module__}.{type(var_4.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4.definitions) == 0
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_5 = module_1.ref_from_json_schema(var_3, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.to == '#/components/schemas/User'
    assert f'{type(var_5.definitions).__module__}.{type(var_5.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5.definitions) == 0
    var_6 = '#/definitions/Address'
    var_7 = {var_1: var_6}
    var_8 = module_1.ref_from_json_schema(var_7, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.to == '#/definitions/Address'
    assert f'{type(var_8.definitions).__module__}.{type(var_8.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8.definitions) == 0
    var_9 = module_4.Definitions()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_9) == 0
    var_10 = module_1.ref_from_json_schema(var_7, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.to == '#/definitions/Address'
    assert f'{type(var_10.definitions).__module__}.{type(var_10.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_10.definitions) == 0
    var_11 = None
    module_1.if_then_else_from_json_schema(var_3, var_11)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'minLength'
    var_7 = 5
    var_8 = {var_3: var_4, var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = module_1.if_then_else_from_json_schema(var_9, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.if_clause).__module__}.{type(var_10.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_10.then_clause).__module__}.{type(var_10.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_10.else_clause).__module__}.{type(var_10.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_11 = var_10.if_clause
    var_12 = var_10.then_clause
    var_11.validate(var_8)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = 'minLength'
    var_6 = 5
    var_7 = {var_3: var_4, var_5: var_6}
    var_8 = {var_1: var_7, var_2: var_7}
    var_9 = module_1.if_then_else_from_json_schema(var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.if_clause).__module__}.{type(var_9.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_9.then_clause).__module__}.{type(var_9.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_9.else_clause).__module__}.{type(var_9.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_10 = var_9.if_clause
    var_11 = var_9.then_clause
    var_12 = 'else'
    var_13 = {var_3: var_4}
    var_14 = {var_3: var_4, var_5: var_6}
    var_15 = 'minimum'
    var_16 = 'integer'
    var_17 = 0
    var_18 = {var_3: var_16, var_15: var_17}
    var_19 = {var_1: var_13, var_2: var_14, var_12: var_18}
    var_20 = module_1.if_then_else_from_json_schema(var_19, var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.if_clause).__module__}.{type(var_20.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_20.then_clause).__module__}.{type(var_20.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_20.else_clause).__module__}.{type(var_20.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_21 = var_20.if_clause
    var_22 = var_20.then_clause
    var_23 = var_20.else_clause
    var_24 = 'default'
    var_25 = 'boolean'
    var_26 = {var_3: var_25}
    var_27 = {var_3: var_25}
    var_28 = 'null'
    var_29 = {var_3: var_28}
    var_30 = True
    var_31 = {var_1: var_26, var_2: var_27, var_12: var_29, var_24: var_30}
    var_32 = module_1.if_then_else_from_json_schema(var_31, var_0)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_32.default is True
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.if_clause).__module__}.{type(var_32.if_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_32.then_clause).__module__}.{type(var_32.then_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_32.else_clause).__module__}.{type(var_32.else_clause).__qualname__}' == 'typesystem.fields.Const'
    var_33 = 'properties'
    var_34 = 'object'
    var_35 = 'x'
    var_36 = {var_3: var_16}
    var_37 = {var_35: var_36}
    var_38 = {var_3: var_34, var_33: var_37}
    var_39 = 'required'
    var_40 = [var_35]
    var_41 = {var_3: var_34, var_39: var_40}
    var_42 = 'y'
    var_43 = {var_3: var_4}
    var_44 = {var_42: var_43}
    var_45 = {var_3: var_34, var_33: var_44}
    var_46 = {var_1: var_38, var_2: var_41, var_12: var_45}
    var_47 = module_1.if_then_else_from_json_schema(var_46, var_0)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert f'{type(var_47.if_clause).__module__}.{type(var_47.if_clause).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_47.then_clause).__module__}.{type(var_47.then_clause).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_47.else_clause).__module__}.{type(var_47.else_clause).__qualname__}' == 'typesystem.fields.Object'
    var_48 = var_47.if_clause
    var_49 = var_47.then_clause
    var_50 = var_47.else_clause
    var_51 = 'minItems'
    var_52 = 'array'
    var_53 = {var_3: var_52, var_51: var_30}
    var_54 = 'uniqueItems'
    var_55 = {var_3: var_52, var_54: var_30}
    var_56 = {var_1: var_53, var_2: var_55}
    var_57 = module_1.if_then_else_from_json_schema(var_56, var_0)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert f'{type(var_57.if_clause).__module__}.{type(var_57.if_clause).__qualname__}' == 'typesystem.fields.Array'
    assert f'{type(var_57.then_clause).__module__}.{type(var_57.then_clause).__qualname__}' == 'typesystem.fields.Array'
    assert f'{type(var_57.else_clause).__module__}.{type(var_57.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_58 = var_57.if_clause
    var_59 = var_57.then_clause
    var_60 = {var_3: var_23, var_15: var_17}
    var_61 = 'maximum'
    var_62 = 100
    var_63 = {var_3: var_58, var_61: var_62}
    var_64 = -100
    var_65 = {var_3: var_48, var_15: var_64}
    var_66 = {var_1: var_60, var_2: var_63, var_12: var_65}
    module_1.if_then_else_from_json_schema(var_66, var_0)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.type_from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is True
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 5
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_2 = [var_1]
    var_3 = module_3.OneOf(var_2, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.one_of).__module__}.{type(var_3.one_of).__qualname__}' == 'builtins.list'
    assert len(var_3.one_of) == 1
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_4 = None
    var_5 = module_3.Not(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.Not'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.negated is None
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_6 = module_1.to_json_schema(var_3)
    module_1.from_json_schema(var_4, var_3)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = None
    var_1 = True
    var_2 = module_2.String(min_length=var_0, format=var_0, coerce_types=var_1)
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
    var_3 = module_1.from_json_schema(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_4 = module_1.to_json_schema(var_2, var_0)
    module_1.to_json_schema(var_4, var_0)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = var_0.__setitem__(var_1, var_1)
    assert len(var_0) == 1
    module_1.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'minLength'
    var_6 = 5
    var_7 = {var_2: var_3, var_5: var_6}
    var_8 = {var_1: var_4, var_2: var_7}
    var_9 = module_1.if_then_else_from_json_schema(var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.if_clause).__module__}.{type(var_9.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_9.then_clause).__module__}.{type(var_9.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_9.else_clause).__module__}.{type(var_9.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_10 = var_9.if_clause
    var_11 = var_9.then_clause
    var_10.validate(var_7)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = module_4.Definitions()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Float'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.minimum == 0
    assert var_9.maximum == 10
    assert var_9.exclusive_minimum is None
    assert var_9.exclusive_maximum is None
    assert var_9.multiple_of is None
    assert var_9.precision is None
    assert var_9.coerce_types is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_10 = 'exclusiveMinimum'
    var_11 = 'multipleOf'
    var_12 = 'integer'
    var_13 = 5
    var_14 = 2
    var_15 = {var_0: var_12, var_10: var_13, var_11: var_14}
    var_16 = True
    var_17 = module_4.Definitions()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_17) == 0
    var_18 = module_1.from_json_schema_type(var_15, var_12, var_16, var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Integer'
    assert var_18.default is None
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is True
    assert var_18.read_only is False
    assert var_18.minimum is None
    assert var_18.maximum is None
    assert var_18.exclusive_minimum == 5
    assert var_18.exclusive_maximum is None
    assert var_18.multiple_of == 2
    assert var_18.precision is None
    assert var_18.coerce_types is False
    var_19 = 'minLength'
    var_20 = 'maxLength'
    var_21 = 'pattern'
    var_22 = 'string'
    var_23 = 3
    var_24 = '^a.*z$'
    var_25 = {var_0: var_22, var_19: var_23, var_20: var_5, var_21: var_24}
    var_26 = False
    var_27 = module_4.Definitions()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_27) == 0
    var_28 = module_1.from_json_schema_type(var_25, var_22, var_26, var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.String'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.allow_blank is False
    assert var_28.trim_whitespace is True
    assert var_28.max_length == 10
    assert var_28.min_length == 3
    assert var_28.format is None
    assert var_28.coerce_types is False
    assert var_28.pattern == '^a.*z$'
    assert f'{type(var_28.pattern_regex).__module__}.{type(var_28.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_29 = module_4.Definitions()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_29) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_30 = 'items'
    var_31 = 'minItems'
    var_32 = 'array'
    var_33 = {var_0: var_22}
    var_34 = {var_0: var_32, var_30: var_33, var_31: var_16}
    var_35 = False
    var_36 = module_4.Definitions()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_36) == 0
    var_37 = module_1.from_json_schema_type(var_34, var_32, var_35, var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Array'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert f'{type(var_37.items).__module__}.{type(var_37.items).__qualname__}' == 'typesystem.fields.String'
    assert var_37.additional_items is True
    assert var_37.min_items is True
    assert var_37.max_items is None
    assert var_37.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_38 = {var_0: var_22}
    var_39 = {var_0: var_3}
    var_40 = [var_38, var_39]
    var_41 = {var_0: var_32, var_30: var_40}
    var_42 = False
    var_43 = module_4.Definitions()
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_43) == 0
    var_44 = module_1.from_json_schema_type(var_41, var_32, var_42, var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Array'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert f'{type(var_44.items).__module__}.{type(var_44.items).__qualname__}' == 'builtins.list'
    assert len(var_44.items) == 2
    assert var_44.additional_items is True
    assert var_44.min_items == 0
    assert var_44.max_items is None
    assert var_44.unique_items is False
    var_45 = var_44.items
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = var_44.items[var_42]
    var_48 = var_44.items[var_16]
    var_49 = 'properties'
    var_50 = 'object'
    var_51 = 'name'
    var_52 = {var_0: var_22}
    var_53 = {var_51: var_52}
    var_54 = {var_0: var_50, var_49: var_53}
    var_55 = False
    var_56 = module_4.Definitions()
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_56) == 0
    var_57 = module_1.from_json_schema_type(var_54, var_50, var_55, var_56)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.Object'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert f'{type(var_57.properties).__module__}.{type(var_57.properties).__qualname__}' == 'builtins.dict'
    assert len(var_57.properties) == 1
    assert var_57.pattern_properties == {}
    assert var_57.additional_properties is None
    assert var_57.property_names is None
    assert var_57.min_properties is None
    assert var_57.max_properties is None
    assert var_57.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_58 = var_57.properties[var_51]
    var_59 = 'patternProperties'
    var_60 = '^S_'
    var_61 = {var_0: var_22}
    var_62 = {var_60: var_61}
    var_63 = {var_0: var_50, var_59: var_62}
    var_64 = module_4.Definitions()
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_64) == 0
    var_65 = module_1.from_json_schema_type(var_63, var_50, var_16, var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Object'
    assert var_65.default is None
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is True
    assert var_65.read_only is False
    assert var_65.properties == {}
    assert f'{type(var_65.pattern_properties).__module__}.{type(var_65.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_65.pattern_properties) == 1
    assert var_65.additional_properties is None
    assert var_65.property_names is None
    assert var_65.min_properties is None
    assert var_65.max_properties is None
    assert var_65.required == []
    var_66 = var_65.pattern_properties[var_60]
    var_67 = 'additionalProperties'
    var_68 = False
    var_69 = {var_0: var_50, var_67: var_68}
    var_70 = False
    var_71 = module_4.Definitions()
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_71) == 0
    var_72 = module_1.from_json_schema_type(var_69, var_50, var_70, var_71)
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.Object'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert var_72.properties == {}
    assert var_72.pattern_properties == {}
    assert var_72.additional_properties is False
    assert var_72.property_names is None
    assert var_72.min_properties is None
    assert var_72.max_properties is None
    assert var_72.required == []
    var_73 = module_4.Definitions()
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_73) == 0
    module_2.Object(properties=var_15, pattern_properties=var_46, min_properties=var_47)

def test_case_45():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'minLength'
    var_7 = 5
    var_8 = {var_3: var_4, var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = module_1.if_then_else_from_json_schema(var_9, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.if_clause).__module__}.{type(var_10.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_10.then_clause).__module__}.{type(var_10.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_10.else_clause).__module__}.{type(var_10.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_11 = var_10.if_clause
    var_12 = var_10.then_clause
    var_13 = 'else'
    var_14 = 'minimum'
    var_15 = 'number'
    var_16 = 0
    var_17 = {var_3: var_15, var_14: var_16}
    var_18 = 'maximum'
    var_19 = 100
    var_20 = {var_3: var_15, var_14: var_16, var_18: var_19}
    var_21 = -1
    var_22 = {var_3: var_15, var_18: var_21}
    var_23 = {var_1: var_17, var_2: var_20, var_13: var_22}
    var_24 = module_1.if_then_else_from_json_schema(var_23, var_0)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.if_clause).__module__}.{type(var_24.if_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_24.then_clause).__module__}.{type(var_24.then_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_24.else_clause).__module__}.{type(var_24.else_clause).__qualname__}' == 'typesystem.fields.Float'
    var_25 = var_24.if_clause
    var_26 = var_24.then_clause
    var_27 = var_24.else_clause
    var_28 = '$ref'
    var_29 = '#/definitions/Positive'
    var_30 = {var_28: var_29}
    var_31 = {var_3: var_4}
    var_32 = 'boolean'
    var_33 = {var_3: var_32}
    var_34 = {var_1: var_30, var_2: var_31, var_13: var_33}
    var_35 = module_1.if_then_else_from_json_schema(var_34, var_0)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.if_clause).__module__}.{type(var_35.if_clause).__qualname__}' == 'typesystem.schemas.Reference'
    assert f'{type(var_35.then_clause).__module__}.{type(var_35.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_35.else_clause).__module__}.{type(var_35.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_36 = var_35.if_clause
    var_37 = var_35.then_clause
    var_38 = var_35.else_clause
    var_39 = 'default'
    var_40 = 'array'
    var_41 = {var_3: var_40}
    var_42 = 'minItems'
    var_43 = 2
    var_44 = {var_3: var_40, var_42: var_43}
    var_45 = 'maxItems'
    var_46 = 1
    var_47 = {var_3: var_40, var_45: var_46}
    var_48 = []
    var_49 = {var_1: var_41, var_2: var_44, var_13: var_47, var_39: var_48}
    var_50 = module_1.if_then_else_from_json_schema(var_49, var_0)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_50.default == []
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert f'{type(var_50.if_clause).__module__}.{type(var_50.if_clause).__qualname__}' == 'typesystem.fields.Array'
    assert f'{type(var_50.then_clause).__module__}.{type(var_50.then_clause).__qualname__}' == 'typesystem.fields.Array'
    assert f'{type(var_50.else_clause).__module__}.{type(var_50.else_clause).__qualname__}' == 'typesystem.fields.Array'
    var_51 = 'allOf'
    var_52 = 'object'
    var_53 = {var_3: var_52}
    var_54 = 'required'
    var_55 = 'status'
    var_56 = [var_55]
    var_57 = {var_54: var_56}
    var_58 = [var_53, var_57]
    var_59 = {var_51: var_58}
    var_60 = 'properties'
    var_61 = {var_3: var_4}
    var_62 = {var_55: var_61}
    var_63 = {var_3: var_52, var_60: var_62}
    var_64 = {var_1: var_59, var_2: var_63}
    var_65 = module_1.if_then_else_from_json_schema(var_64, var_0)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert f'{type(var_65.if_clause).__module__}.{type(var_65.if_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_65.then_clause).__module__}.{type(var_65.then_clause).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_65.else_clause).__module__}.{type(var_65.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_66 = var_65.if_clause
    var_67 = var_65.then_clause

def test_case_46():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = False
    var_3 = 1
    var_4 = 10
    var_5 = '^[a-z]+$'
    var_6 = module_2.String(max_length=var_4, min_length=var_3, pattern=var_5, format=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.String'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.allow_blank is False
    assert var_6.trim_whitespace is True
    assert var_6.max_length == 10
    assert var_6.min_length == 1
    assert var_6.format == '^[a-z]+$'
    assert var_6.coerce_types is True
    assert var_6.pattern == '^[a-z]+$'
    assert f'{type(var_6.pattern_regex).__module__}.{type(var_6.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = module_1.get_standard_properties(var_6)
    var_8 = module_1.to_json_schema(var_6)
    var_9 = False
    var_10 = module_2.Boolean(coerce_types=var_9, **var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_11 = module_2.Integer()
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
    var_12 = [var_6, var_11]
    var_13 = module_2.Array(var_12)
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
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_14 = module_1.to_json_schema(var_13)
    with pytest.raises(KeyError):
        var_15 = var_14[var_10]

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = module_3.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_2.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_2.String()
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
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 2
    var_15 = module_2.Integer(minimum=var_4, maximum=var_6, exclusive_minimum=var_4, exclusive_maximum=var_6, multiple_of=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Integer'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is False
    assert var_15.maximum == 10
    assert var_15.exclusive_minimum is False
    assert var_15.exclusive_maximum == 10
    assert var_15.multiple_of == 2
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_1.to_json_schema(var_15)
    var_17 = True
    var_18 = module_2.Float(minimum=var_4, maximum=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Float'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.minimum is False
    assert var_18.maximum is True
    assert var_18.exclusive_minimum is None
    assert var_18.exclusive_maximum is None
    assert var_18.multiple_of is None
    assert var_18.precision is None
    assert var_18.coerce_types is True
    var_19 = module_1.to_json_schema(var_18)
    var_20 = True
    var_21 = module_2.Boolean()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_22 = module_1.to_json_schema(var_21)
    var_23 = var_0.validate(var_13)
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
    var_25 = [var_9, var_24]
    var_26 = module_2.Array(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Array'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.items).__module__}.{type(var_26.items).__qualname__}' == 'builtins.list'
    assert len(var_26.items) == 2
    assert var_26.additional_items is False
    assert var_26.min_items == 2
    assert var_26.max_items == 2
    assert var_26.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_27 = module_1.to_json_schema(var_26)
    var_28 = 'items'
    var_29 = var_27[var_28]
    var_30 = var_27[var_28]
    var_31 = module_2.Array(additional_items=var_4)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Array'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.items is None
    assert var_31.additional_items is False
    assert var_31.min_items is None
    assert var_31.max_items is None
    assert var_31.unique_items is False
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_2.String()
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
    var_34 = module_2.Array(additional_items=var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Array'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.items is None
    assert f'{type(var_34.additional_items).__module__}.{type(var_34.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_34.min_items is None
    assert var_34.max_items is None
    assert var_34.unique_items is False
    var_35 = module_1.to_json_schema(var_34)
    var_36 = True
    var_37 = 'name'
    var_38 = 'age'
    var_39 = module_2.String()
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
    var_40 = module_2.Integer()
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Integer'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert var_40.minimum is None
    assert var_40.maximum is None
    assert var_40.exclusive_minimum is None
    assert var_40.exclusive_maximum is None
    assert var_40.multiple_of is None
    assert var_40.precision is None
    assert var_40.coerce_types is True
    var_41 = {var_37: var_39, var_38: var_40}
    var_42 = '^test_'
    var_43 = module_2.Boolean()
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.coerce_types is True
    var_44 = {var_42: var_43}
    var_45 = [var_37]
    var_46 = module_2.Object(properties=var_41, pattern_properties=var_44, additional_properties=var_4, min_properties=var_36, max_properties=var_6, required=var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Object'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.properties).__module__}.{type(var_46.properties).__qualname__}' == 'builtins.dict'
    assert len(var_46.properties) == 2
    assert f'{type(var_46.pattern_properties).__module__}.{type(var_46.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_46.pattern_properties) == 1
    assert var_46.additional_properties is False
    assert var_46.property_names is None
    assert var_46.min_properties is True
    assert var_46.max_properties == 10
    assert var_46.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_2.String()
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
    assert var_48.coerce_types is True
    assert var_48.pattern is None
    assert var_48.pattern_regex is None
    var_49 = module_2.Object(additional_properties=var_48)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.fields.Object'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert var_49.properties == {}
    assert var_49.pattern_properties == {}
    assert f'{type(var_49.additional_properties).__module__}.{type(var_49.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_49.property_names is None
    assert var_49.min_properties is None
    assert var_49.max_properties is None
    assert var_49.required == []
    var_50 = module_1.to_json_schema(var_49)
    var_51 = module_2.String(pattern=var_7)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.String'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert var_51.allow_blank is False
    assert var_51.trim_whitespace is True
    assert var_51.max_length is None
    assert var_51.min_length is None
    assert var_51.format is None
    assert var_51.coerce_types is True
    assert var_51.pattern == '^[a-z]+$'
    assert f'{type(var_51.pattern_regex).__module__}.{type(var_51.pattern_regex).__qualname__}' == 're.Pattern'
    module_1.to_json_schema(var_29)

def test_case_48():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = module_3.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_2.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = False
    var_13 = module_2.String(allow_blank=var_12, min_length=var_4)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.String'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.allow_blank is False
    assert var_13.trim_whitespace is True
    assert var_13.max_length is None
    assert var_13.min_length is False
    assert var_13.format is None
    assert var_13.coerce_types is True
    assert var_13.pattern is None
    assert var_13.pattern_regex is None
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 100
    var_16 = 2
    var_17 = module_2.Integer(minimum=var_4, maximum=var_15, exclusive_minimum=var_4, exclusive_maximum=var_15, multiple_of=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Integer'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum is False
    assert var_17.maximum == 100
    assert var_17.exclusive_minimum is False
    assert var_17.exclusive_maximum == 100
    assert var_17.multiple_of == 2
    assert var_17.precision is None
    assert var_17.coerce_types is True
    var_18 = module_1.to_json_schema(var_17)
    var_19 = True
    var_20 = module_2.Float(minimum=var_4, maximum=var_19)
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
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_2.Boolean()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_24 = module_1.to_json_schema(var_23)
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
    var_26 = True
    var_27 = module_2.Array(var_25, min_items=var_22, max_items=var_6, unique_items=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Array'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.items).__module__}.{type(var_27.items).__qualname__}' == 'typesystem.fields.String'
    assert var_27.additional_items is False
    assert var_27.min_items is True
    assert var_27.max_items == 10
    assert var_27.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_28 = module_1.to_json_schema(var_27)
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
    var_31 = module_1.to_json_schema(var_27)
    var_32 = 'items'
    var_33 = var_31[var_32]
    var_34 = var_31[var_32]
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = True
    var_37 = 'name'
    var_38 = 'age'
    var_39 = module_2.String()
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
    var_40 = module_2.Integer()
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Integer'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert var_40.minimum is None
    assert var_40.maximum is None
    assert var_40.exclusive_minimum is None
    assert var_40.exclusive_maximum is None
    assert var_40.multiple_of is None
    assert var_40.precision is None
    assert var_40.coerce_types is True
    var_41 = {var_37: var_39, var_38: var_40}
    var_42 = [var_37]
    var_43 = module_2.Object(properties=var_41, additional_properties=var_4, min_properties=var_36, max_properties=var_16, required=var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Object'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert f'{type(var_43.properties).__module__}.{type(var_43.properties).__qualname__}' == 'builtins.dict'
    assert len(var_43.properties) == 2
    assert var_43.pattern_properties == {}
    assert var_43.additional_properties is False
    assert var_43.property_names is None
    assert var_43.min_properties is True
    assert var_43.max_properties == 2
    assert var_43.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_44 = module_1.to_json_schema(var_43)
    var_45 = module_2.String()
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
    var_46 = {var_7: var_45}
    var_47 = module_2.Object(pattern_properties=var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.Object'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert var_47.properties == {}
    assert f'{type(var_47.pattern_properties).__module__}.{type(var_47.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_47.pattern_properties) == 1
    assert var_47.additional_properties is True
    assert var_47.property_names is None
    assert var_47.min_properties is None
    assert var_47.max_properties is None
    assert var_47.required == []
    var_48 = module_1.to_json_schema(var_47)
    var_49 = 'A'
    var_50 = 'Option A'
    var_51 = (var_49, var_50)
    var_52 = 'B'
    var_53 = 'Option B'
    var_54 = (var_52, var_53)
    var_55 = [var_51, var_54]
    var_56 = module_2.Choice(choices=var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Choice'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert var_56.choices == [('A', 'Option A'), ('B', 'Option B')]
    assert var_56.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_57 = module_1.to_json_schema(var_56)
    var_58 = 'fixed_value'
    var_59 = module_2.Const(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.fields.Const'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert var_59.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_60 = module_1.to_json_schema(var_59)
    var_61 = module_2.String()
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.fields.String'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert var_61.allow_blank is False
    assert var_61.trim_whitespace is True
    assert var_61.max_length is None
    assert var_61.min_length is None
    assert var_61.format is None
    assert var_61.coerce_types is True
    assert var_61.pattern is None
    assert var_61.pattern_regex is None
    var_62 = module_2.Integer()
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.fields.Integer'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert var_62.minimum is None
    assert var_62.maximum is None
    assert var_62.exclusive_minimum is None
    assert var_62.exclusive_maximum is None
    assert var_62.multiple_of is None
    assert var_62.precision is None
    assert var_62.coerce_types is True
    var_63 = [var_61, var_62]
    var_64 = module_2.Union(var_63)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.Union'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert f'{type(var_64.any_of).__module__}.{type(var_64.any_of).__qualname__}' == 'builtins.list'
    assert len(var_64.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_65 = module_1.to_json_schema(var_64)
    var_66 = 'anyOf'
    var_67 = var_65[var_66]
    var_68 = module_2.String()
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.fields.String'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert var_68.allow_blank is False
    assert var_68.trim_whitespace is True
    assert var_68.max_length is None
    assert var_68.min_length is None
    assert var_68.format is None
    assert var_68.coerce_types is True
    assert var_68.pattern is None
    assert var_68.pattern_regex is None
    var_69 = module_2.Integer()
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
    var_70 = [var_68, var_69]
    var_71 = module_3.OneOf(var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert f'{type(var_71.one_of).__module__}.{type(var_71.one_of).__qualname__}' == 'builtins.list'
    assert len(var_71.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_72 = module_1.to_json_schema(var_71)
    var_73 = 'oneOf'
    var_74 = var_72[var_73]
    var_75 = len(var_74)
    assert var_75 == 2
    var_76 = module_2.String(min_length=var_36)
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.fields.String'
    assert var_76.title == ''
    assert var_76.description == ''
    assert var_76.allow_null is False
    assert var_76.read_only is False
    assert var_76.allow_blank is False
    assert var_76.trim_whitespace is True
    assert var_76.max_length is None
    assert var_76.min_length is True
    assert var_76.format is None
    assert var_76.coerce_types is True
    assert var_76.pattern is None
    assert var_76.pattern_regex is None
    var_77 = module_2.String(max_length=var_6)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.String'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert var_77.allow_blank is False
    assert var_77.trim_whitespace is True
    assert var_77.max_length == 10
    assert var_77.min_length is None
    assert var_77.format is None
    assert var_77.coerce_types is True
    assert var_77.pattern is None
    assert var_77.pattern_regex is None
    var_78 = [var_76, var_77]
    var_79 = module_3.AllOf(var_78)
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_79.title == ''
    assert var_79.description == ''
    assert var_79.allow_null is False
    assert var_79.read_only is False
    assert f'{type(var_79.all_of).__module__}.{type(var_79.all_of).__qualname__}' == 'builtins.list'
    assert len(var_79.all_of) == 2
    var_80 = module_1.to_json_schema(var_79)
    var_81 = 'allOf'
    var_82 = var_80[var_81]
    var_83 = len(var_82)
    assert var_83 == 2
    var_84 = 5
    var_85 = module_2.String(min_length=var_84)
    assert f'{type(var_85).__module__}.{type(var_85).__qualname__}' == 'typesystem.fields.String'
    assert var_85.title == ''
    assert var_85.description == ''
    assert var_85.allow_null is False
    assert var_85.read_only is False
    assert var_85.allow_blank is False
    assert var_85.trim_whitespace is True
    assert var_85.max_length is None
    assert var_85.min_length == 5
    assert var_85.format is None
    assert var_85.coerce_types is True
    assert var_85.pattern is None
    assert var_85.pattern_regex is None
    var_86 = 20
    var_87 = module_2.String(max_length=var_86)
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.fields.String'
    assert var_87.title == ''
    assert var_87.description == ''
    assert var_87.allow_null is False
    assert var_87.read_only is False
    assert var_87.allow_blank is False
    assert var_87.trim_whitespace is True
    assert var_87.max_length == 20
    assert var_87.min_length is None
    assert var_87.format is None
    assert var_87.coerce_types is True
    assert var_87.pattern is None
    assert var_87.pattern_regex is None
    var_88 = module_2.Integer()
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
    var_89 = module_3.IfThenElse(var_85, var_87, var_88)
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_89.title == ''
    assert var_89.description == ''
    assert var_89.allow_null is False
    assert var_89.read_only is False
    assert f'{type(var_89.if_clause).__module__}.{type(var_89.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_89.then_clause).__module__}.{type(var_89.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_89.else_clause).__module__}.{type(var_89.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_90 = module_1.to_json_schema(var_89)
    var_91 = module_2.String()
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
    var_92 = module_3.Not(var_91)
    assert f'{type(var_92).__module__}.{type(var_92).__qualname__}' == 'typesystem.composites.Not'
    assert var_92.title == ''
    assert var_92.description == ''
    assert var_92.allow_null is False
    assert var_92.read_only is False
    assert f'{type(var_92.negated).__module__}.{type(var_92.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_93 = module_1.to_json_schema(var_92)
    var_94 = module_2.String()
    assert f'{type(var_94).__module__}.{type(var_94).__qualname__}' == 'typesystem.fields.String'
    assert var_94.title == ''
    assert var_94.description == ''
    assert var_94.allow_null is False
    assert var_94.read_only is False
    assert var_94.allow_blank is False
    assert var_94.trim_whitespace is True
    assert var_94.max_length is None
    assert var_94.min_length is None
    assert var_94.format is None
    assert var_94.coerce_types is True
    assert var_94.pattern is None
    assert var_94.pattern_regex is None
    var_95 = {var_37: var_94}
    var_96 = module_2.Object(properties=var_95)
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.fields.Object'
    assert var_96.title == ''
    assert var_96.description == ''
    assert var_96.allow_null is False
    assert var_96.read_only is False
    assert f'{type(var_96.properties).__module__}.{type(var_96.properties).__qualname__}' == 'builtins.dict'
    assert len(var_96.properties) == 1
    assert var_96.pattern_properties == {}
    assert var_96.additional_properties is True
    assert var_96.property_names is None
    assert var_96.min_properties is None
    assert var_96.max_properties is None
    assert var_96.required == []

def test_case_49():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = module_4.Definitions()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Float'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.minimum == 0
    assert var_9.maximum == 10
    assert var_9.exclusive_minimum is None
    assert var_9.exclusive_maximum is None
    assert var_9.multiple_of is None
    assert var_9.precision is None
    assert var_9.coerce_types is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_10 = 'multipleOf'
    var_11 = 'integer'
    var_12 = 5
    var_13 = 2
    var_14 = {var_0: var_11, var_0: var_12, var_10: var_13}
    var_15 = True
    var_16 = module_4.Definitions()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_16) == 0
    var_17 = module_1.from_json_schema_type(var_14, var_11, var_15, var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Integer'
    assert var_17.default is None
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is True
    assert var_17.read_only is False
    assert var_17.minimum is None
    assert var_17.maximum is None
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of == 2
    assert var_17.precision is None
    assert var_17.coerce_types is False
    var_18 = 'pattern'
    var_19 = 'string'
    var_20 = False
    var_21 = module_4.Definitions()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_21) == 0
    var_22 = module_1.from_json_schema_type(var_14, var_19, var_20, var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.String'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.default == ''
    assert var_22.allow_blank is True
    assert var_22.trim_whitespace is True
    assert var_22.max_length is None
    assert var_22.min_length is None
    assert var_22.format is None
    assert var_22.coerce_types is False
    assert var_22.pattern is None
    assert var_22.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_23 = 'default'
    var_24 = 'boolean'
    var_25 = {var_0: var_24, var_23: var_15}
    var_26 = module_4.Definitions()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_26) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_27 = module_1.from_json_schema_type(var_25, var_24, var_15, var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_27.default is True
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is True
    assert var_27.read_only is False
    assert var_27.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_28 = 'items'
    var_29 = 'minItems'
    var_30 = 'array'
    var_31 = {var_0: var_19}
    var_32 = {var_0: var_30, var_28: var_31, var_29: var_15}
    var_33 = False
    var_34 = module_4.Definitions()
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_34) == 0
    var_35 = module_1.from_json_schema_type(var_32, var_30, var_33, var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Array'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.items).__module__}.{type(var_35.items).__qualname__}' == 'typesystem.fields.String'
    assert var_35.additional_items is True
    assert var_35.min_items is True
    assert var_35.max_items is None
    assert var_35.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_36 = var_35.items
    var_37 = {var_0: var_19}
    var_38 = {var_0: var_3}
    var_39 = [var_37, var_38]
    var_40 = {var_0: var_30, var_28: var_39}
    var_41 = False
    var_42 = module_4.Definitions()
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_42) == 0
    var_43 = module_1.from_json_schema_type(var_40, var_30, var_41, var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Array'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert f'{type(var_43.items).__module__}.{type(var_43.items).__qualname__}' == 'builtins.list'
    assert len(var_43.items) == 2
    assert var_43.additional_items is True
    assert var_43.min_items == 0
    assert var_43.max_items is None
    assert var_43.unique_items is False
    var_44 = var_43.items
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = var_43.items[var_41]
    var_47 = var_43.items[var_15]
    var_48 = 'properties'
    var_49 = 'object'
    var_50 = 'name'
    var_51 = {var_0: var_19}
    var_52 = {var_50: var_51}
    var_53 = {var_0: var_49, var_48: var_52}
    var_54 = False
    var_55 = module_4.Definitions()
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_55) == 0
    var_56 = module_1.from_json_schema_type(var_53, var_49, var_54, var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Object'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.properties).__module__}.{type(var_56.properties).__qualname__}' == 'builtins.dict'
    assert len(var_56.properties) == 1
    assert var_56.pattern_properties == {}
    assert var_56.additional_properties is None
    assert var_56.property_names is None
    assert var_56.min_properties is None
    assert var_56.max_properties is None
    assert var_56.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_57 = var_56.properties[var_50]
    var_58 = 'patternProperties'
    var_59 = '^S_'
    var_60 = {var_0: var_19}
    var_61 = {var_59: var_60}
    var_62 = {var_0: var_49, var_58: var_61}
    var_63 = module_4.Definitions()
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_63) == 0
    var_64 = module_1.from_json_schema_type(var_62, var_49, var_15, var_63)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.Object'
    assert var_64.default is None
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is True
    assert var_64.read_only is False
    assert var_64.properties == {}
    assert f'{type(var_64.pattern_properties).__module__}.{type(var_64.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_64.pattern_properties) == 1
    assert var_64.additional_properties is None
    assert var_64.property_names is None
    assert var_64.min_properties is None
    assert var_64.max_properties is None
    assert var_64.required == []
    var_65 = var_64.pattern_properties[var_59]
    var_66 = 'additionalProperties'
    var_67 = False
    var_68 = {var_0: var_49, var_66: var_67}
    var_69 = False
    var_70 = module_4.Definitions()
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_70) == 0
    var_71 = module_1.from_json_schema_type(var_68, var_49, var_69, var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Object'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.properties == {}
    assert var_71.pattern_properties == {}
    assert var_71.additional_properties is False
    assert var_71.property_names is None
    assert var_71.min_properties is None
    assert var_71.max_properties is None
    assert var_71.required == []
    var_72 = 'propertyNames'
    var_73 = '^[a-z]+$'
    var_74 = {var_18: var_73}
    var_75 = {var_0: var_49, var_72: var_74}
    var_76 = False
    var_77 = var_45.__dir__()
    var_78 = module_1.from_json_schema_type(var_75, var_49, var_76, var_77)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.Object'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert var_78.properties == {}
    assert var_78.pattern_properties == {}
    assert var_78.additional_properties is None
    assert f'{type(var_78.property_names).__module__}.{type(var_78.property_names).__qualname__}' == 'typesystem.fields.Union'
    assert var_78.min_properties is None
    assert var_78.max_properties is None
    assert var_78.required == []
    var_79 = var_78.property_names
    var_80 = 'test'
    var_81 = {var_0: var_19, var_23: var_80}
    var_82 = False
    var_83 = module_4.Definitions()
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_83) == 0
    var_84 = module_1.from_json_schema_type(var_81, var_19, var_82, var_83)
    assert f'{type(var_84).__module__}.{type(var_84).__qualname__}' == 'typesystem.fields.String'
    assert var_84.default == 'test'
    assert var_84.title == ''
    assert var_84.description == ''
    assert var_84.allow_null is False
    assert var_84.read_only is False
    assert var_84.allow_blank is True
    assert var_84.trim_whitespace is True
    assert var_84.max_length is None
    assert var_84.min_length is None
    assert var_84.format is None
    assert var_84.coerce_types is False
    assert var_84.pattern is None
    assert var_84.pattern_regex is None
    var_85 = {var_0: var_3}
    var_86 = False
    var_87 = module_1.from_json_schema_type(var_85, var_3, var_86, var_69)
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.fields.Float'
    assert var_87.title == ''
    assert var_87.description == ''
    assert var_87.allow_null is False
    assert var_87.read_only is False
    assert var_87.minimum is None
    assert var_87.maximum is None
    assert var_87.exclusive_minimum is None
    assert var_87.exclusive_maximum is None
    assert var_87.multiple_of is None
    assert var_87.precision is None
    assert var_87.coerce_types is False
    var_88 = {}
    var_89 = 'invalid'
    var_90 = False
    var_91 = module_4.Definitions()
    assert f'{type(var_91).__module__}.{type(var_91).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_91) == 0
    with pytest.raises(AssertionError):
        module_1.from_json_schema_type(var_88, var_89, var_90, var_91)

def test_case_50():
    var_0 = True
    var_1 = 1
    var_2 = '^[a-z]+$'
    var_3 = module_2.String(max_length=var_0, min_length=var_1, pattern=var_2, format=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is True
    assert var_3.min_length == 1
    assert var_3.format == '^[a-z]+$'
    assert var_3.coerce_types is True
    assert var_3.pattern == '^[a-z]+$'
    assert f'{type(var_3.pattern_regex).__module__}.{type(var_3.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_1.to_json_schema(var_3)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_5 = module_2.Integer()
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

def test_case_51():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = False
    var_3 = 1
    var_4 = 10
    var_5 = '^[a-z]+$'
    var_6 = 'email'
    var_7 = module_2.String(max_length=var_4, min_length=var_3, pattern=var_5, format=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length == 10
    assert var_7.min_length == 1
    assert var_7.format == 'email'
    assert var_7.coerce_types is True
    assert var_7.pattern == '^[a-z]+$'
    assert f'{type(var_7.pattern_regex).__module__}.{type(var_7.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = False
    var_9 = module_1.get_standard_properties(var_7)
    var_10 = module_2.String()
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
    var_11 = module_1.to_json_schema(var_10)
    var_12 = True
    var_13 = module_2.Float(minimum=var_2, maximum=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Float'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum is False
    assert var_13.maximum is True
    assert var_13.exclusive_minimum is None
    assert var_13.exclusive_maximum is None
    assert var_13.multiple_of is None
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = module_1.to_json_schema(var_13)
    var_15 = True
    var_16 = module_2.Boolean(coerce_types=var_15, **var_9)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_17 = module_2.Integer()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Integer'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum is None
    assert var_17.maximum is None
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of is None
    assert var_17.precision is None
    assert var_17.coerce_types is True
    var_18 = [var_7, var_17]
    var_19 = module_2.Array(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Array'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.items).__module__}.{type(var_19.items).__qualname__}' == 'builtins.list'
    assert len(var_19.items) == 2
    assert var_19.additional_items is False
    assert var_19.min_items == 2
    assert var_19.max_items == 2
    assert var_19.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_20 = module_1.to_json_schema(var_19)
    with pytest.raises(TypeError):
        var_21 = var_1[var_14]

def test_case_52():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = module_3.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_2.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_2.String()
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
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 2
    var_15 = module_2.Integer(minimum=var_4, maximum=var_6, exclusive_minimum=var_4, exclusive_maximum=var_6, multiple_of=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Integer'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is False
    assert var_15.maximum == 10
    assert var_15.exclusive_minimum is False
    assert var_15.exclusive_maximum == 10
    assert var_15.multiple_of == 2
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_1.to_json_schema(var_15)
    var_17 = True
    var_18 = module_2.Float(minimum=var_4, maximum=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Float'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.minimum is False
    assert var_18.maximum is True
    assert var_18.exclusive_minimum is None
    assert var_18.exclusive_maximum is None
    assert var_18.multiple_of is None
    assert var_18.precision is None
    assert var_18.coerce_types is True
    var_19 = module_1.to_json_schema(var_18)
    var_20 = True
    var_21 = module_2.Boolean()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_22 = module_1.to_json_schema(var_21)
    var_23 = var_0.validate(var_13)
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
    var_25 = [var_9, var_24]
    var_26 = module_2.Array(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Array'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.items).__module__}.{type(var_26.items).__qualname__}' == 'builtins.list'
    assert len(var_26.items) == 2
    assert var_26.additional_items is False
    assert var_26.min_items == 2
    assert var_26.max_items == 2
    assert var_26.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_27 = module_1.to_json_schema(var_26)
    var_28 = 'items'
    var_29 = var_27[var_28]
    var_30 = var_27[var_28]
    var_31 = module_2.Array(additional_items=var_4)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Array'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.items is None
    assert var_31.additional_items is False
    assert var_31.min_items is None
    assert var_31.max_items is None
    assert var_31.unique_items is False
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_2.String()
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
    var_34 = module_2.Array(additional_items=var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Array'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.items is None
    assert f'{type(var_34.additional_items).__module__}.{type(var_34.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_34.min_items is None
    assert var_34.max_items is None
    assert var_34.unique_items is False
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_2.String()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.String'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.allow_blank is False
    assert var_36.trim_whitespace is True
    assert var_36.max_length is None
    assert var_36.min_length is None
    assert var_36.format is None
    assert var_36.coerce_types is True
    assert var_36.pattern is None
    assert var_36.pattern_regex is None

def test_case_53():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = False
    var_3 = 1
    var_4 = 10
    var_5 = '^[a-z]+$'
    var_6 = 'email'
    var_7 = module_2.String(max_length=var_4, min_length=var_3, pattern=var_5, format=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length == 10
    assert var_7.min_length == 1
    assert var_7.format == 'email'
    assert var_7.coerce_types is True
    assert var_7.pattern == '^[a-z]+$'
    assert f'{type(var_7.pattern_regex).__module__}.{type(var_7.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_1.to_json_schema(var_7)
    var_9 = True
    var_10 = module_2.String()
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
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 2
    var_13 = module_2.Integer(minimum=var_2, maximum=var_4, exclusive_minimum=var_2, exclusive_maximum=var_4, multiple_of=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum is False
    assert var_13.maximum == 10
    assert var_13.exclusive_minimum is False
    assert var_13.exclusive_maximum == 10
    assert var_13.multiple_of == 2
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = True
    var_15 = module_2.Float(minimum=var_2, maximum=var_14)
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
    var_16 = module_1.to_json_schema(var_15)
    var_17 = True
    var_18 = module_2.Boolean()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_19 = module_1.to_json_schema(var_18)
    var_20 = var_0.validate(var_11)
    var_21 = module_2.Integer()
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
    var_22 = [var_7, var_21]
    var_23 = module_2.Array(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Array'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.items).__module__}.{type(var_23.items).__qualname__}' == 'builtins.list'
    assert len(var_23.items) == 2
    assert var_23.additional_items is False
    assert var_23.min_items == 2
    assert var_23.max_items == 2
    assert var_23.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_24 = module_1.to_json_schema(var_23)
    var_25 = 'items'
    var_26 = var_24[var_25]
    var_27 = var_24[var_25]
    var_28 = module_2.Array(additional_items=var_2)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Array'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.items is None
    assert var_28.additional_items is False
    assert var_28.min_items is None
    assert var_28.max_items is None
    assert var_28.unique_items is False
    var_29 = module_1.to_json_schema(var_28)
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
    var_31 = module_2.Array(additional_items=var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Array'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.items is None
    assert f'{type(var_31.additional_items).__module__}.{type(var_31.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_31.min_items is None
    assert var_31.max_items is None
    assert var_31.unique_items is False
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_2.String()
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

@pytest.mark.xfail(strict=True)
def test_case_54():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = module_3.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
    var_5 = 5
    var_6 = 10
    var_7 = '^test.*$'
    var_8 = 'email'
    var_9 = module_2.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length == 10
    assert var_9.min_length == 5
    assert var_9.format == 'email'
    assert var_9.coerce_types is True
    assert var_9.pattern == '^test.*$'
    assert f'{type(var_9.pattern_regex).__module__}.{type(var_9.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_1.to_json_schema(var_9)
    var_11 = False
    var_12 = 100
    var_13 = module_2.Integer(minimum=var_11, maximum=var_12, exclusive_minimum=var_11, exclusive_maximum=var_12, multiple_of=var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum is False
    assert var_13.maximum == 100
    assert var_13.exclusive_minimum is False
    assert var_13.exclusive_maximum == 100
    assert var_13.multiple_of == 5
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 0.1
    var_16 = module_2.Float(minimum=var_11, maximum=var_4, multiple_of=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Float'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum is False
    assert var_16.maximum is True
    assert var_16.exclusive_minimum is None
    assert var_16.exclusive_maximum is None
    assert var_16.multiple_of == pytest.approx(0.1, abs=0.01, rel=0.01)
    assert var_16.precision is None
    assert var_16.coerce_types is True
    var_17 = module_1.to_json_schema(var_16)
    var_18 = module_2.Boolean()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_19 = module_1.to_json_schema(var_18)
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
    var_21 = module_2.Array(var_20, var_11, var_4, var_6, unique_items=var_4)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Array'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.items).__module__}.{type(var_21.items).__qualname__}' == 'typesystem.fields.String'
    assert var_21.additional_items is False
    assert var_21.min_items is True
    assert var_21.max_items == 10
    assert var_21.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_2.String()
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
    var_25 = [var_23, var_24]
    var_26 = module_2.Boolean()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.coerce_types is True
    var_27 = module_2.Array(var_25, var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Array'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.items).__module__}.{type(var_27.items).__qualname__}' == 'builtins.list'
    assert len(var_27.items) == 2
    assert f'{type(var_27.additional_items).__module__}.{type(var_27.additional_items).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_27.min_items == 2
    assert var_27.max_items is None
    assert var_27.unique_items is False
    var_28 = module_1.to_json_schema(var_27)
    var_29 = 'items'
    var_30 = var_28[var_29]
    var_31 = var_28[var_29]
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = 'name'
    var_34 = 'age'
    var_35 = module_2.String()
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
    var_36 = module_2.Integer()
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
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = module_2.Boolean()
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.coerce_types is True
    var_39 = {var_7: var_38}
    var_40 = '^[a-z]+$'
    var_41 = module_2.String(pattern=var_40)
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
    assert var_41.pattern == '^[a-z]+$'
    assert f'{type(var_41.pattern_regex).__module__}.{type(var_41.pattern_regex).__qualname__}' == 're.Pattern'
    var_42 = [var_33]
    var_43 = module_2.Object(properties=var_37, pattern_properties=var_39, additional_properties=var_11, property_names=var_41, min_properties=var_4, max_properties=var_5, required=var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Object'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert f'{type(var_43.properties).__module__}.{type(var_43.properties).__qualname__}' == 'builtins.dict'
    assert len(var_43.properties) == 2
    assert f'{type(var_43.pattern_properties).__module__}.{type(var_43.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_43.pattern_properties) == 1
    assert var_43.additional_properties is False
    assert f'{type(var_43.property_names).__module__}.{type(var_43.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_43.min_properties is True
    assert var_43.max_properties == 5
    assert var_43.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_44 = module_1.to_json_schema(var_43)
    var_45 = module_2.String()
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
    var_46 = module_2.Integer()
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Integer'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert var_46.minimum is None
    assert var_46.maximum is None
    assert var_46.exclusive_minimum is None
    assert var_46.exclusive_maximum is None
    assert var_46.multiple_of is None
    assert var_46.precision is None
    assert var_46.coerce_types is True
    var_47 = {var_33: var_45, var_34: var_46}
    var_48 = [var_33]
    var_49 = module_4.Schema(var_47)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert f'{type(var_49.fields).__module__}.{type(var_49.fields).__qualname__}' == 'builtins.dict'
    assert len(var_49.fields) == 2
    assert var_49.required == ['name', 'age']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_50 = module_1.to_json_schema(var_49)
    var_51 = 'A'
    var_52 = (var_51, var_51)
    var_53 = 'B'
    var_54 = (var_53, var_53)
    var_55 = 'C'
    var_56 = (var_55, var_55)
    var_57 = [var_52, var_54, var_56]
    var_58 = module_2.Choice(choices=var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.Choice'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.choices == [('A', 'A'), ('B', 'B'), ('C', 'C')]
    assert var_58.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_59 = module_1.to_json_schema(var_58)
    var_60 = 'fixed_value'
    var_61 = module_2.Const(var_60)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.fields.Const'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert var_61.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_62 = module_1.to_json_schema(var_61)
    var_63 = module_2.String()
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
    var_64 = module_2.Integer()
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.Integer'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert var_64.minimum is None
    assert var_64.maximum is None
    assert var_64.exclusive_minimum is None
    assert var_64.exclusive_maximum is None
    assert var_64.multiple_of is None
    assert var_64.precision is None
    assert var_64.coerce_types is True
    var_65 = module_2.Boolean()
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.coerce_types is True
    var_66 = [var_63, var_64, var_65]
    var_67 = module_2.Union(var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.fields.Union'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    assert f'{type(var_67.any_of).__module__}.{type(var_67.any_of).__qualname__}' == 'builtins.list'
    assert len(var_67.any_of) == 3
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_68 = module_1.to_json_schema(var_67)
    var_69 = 'anyOf'
    var_70 = var_68[var_69]
    var_71 = len(var_70)
    assert var_71 == 3
    var_72 = module_2.String()
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
    assert var_72.pattern is None
    assert var_72.pattern_regex is None
    var_73 = module_2.Integer()
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.Integer'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert var_73.minimum is None
    assert var_73.maximum is None
    assert var_73.exclusive_minimum is None
    assert var_73.exclusive_maximum is None
    assert var_73.multiple_of is None
    assert var_73.precision is None
    assert var_73.coerce_types is True
    var_74 = [var_72, var_73]
    var_75 = module_3.OneOf(var_74)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert f'{type(var_75.one_of).__module__}.{type(var_75.one_of).__qualname__}' == 'builtins.list'
    assert len(var_75.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_76 = module_1.to_json_schema(var_75)
    var_77 = 'oneOf'
    var_78 = var_76[var_77]
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = module_2.String(min_length=var_5)
    assert f'{type(var_80).__module__}.{type(var_80).__qualname__}' == 'typesystem.fields.String'
    assert var_80.title == ''
    assert var_80.description == ''
    assert var_80.allow_null is False
    assert var_80.read_only is False
    assert var_80.allow_blank is False
    assert var_80.trim_whitespace is True
    assert var_80.max_length is None
    assert var_80.min_length == 5
    assert var_80.format is None
    assert var_80.coerce_types is True
    assert var_80.pattern is None
    assert var_80.pattern_regex is None
    var_81 = module_2.String(max_length=var_6)
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'typesystem.fields.String'
    assert var_81.title == ''
    assert var_81.description == ''
    assert var_81.allow_null is False
    assert var_81.read_only is False
    assert var_81.allow_blank is False
    assert var_81.trim_whitespace is True
    assert var_81.max_length == 10
    assert var_81.min_length is None
    assert var_81.format is None
    assert var_81.coerce_types is True
    assert var_81.pattern is None
    assert var_81.pattern_regex is None
    var_82 = [var_80, var_81]
    var_83 = module_3.AllOf(var_82)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert f'{type(var_83.all_of).__module__}.{type(var_83.all_of).__qualname__}' == 'builtins.list'
    assert len(var_83.all_of) == 2
    var_84 = module_1.to_json_schema(var_83)
    var_85 = 'allOf'
    var_86 = var_84[var_85]
    var_87 = len(var_86)
    assert var_87 == 2
    var_88 = module_2.String(pattern=var_7)
    assert f'{type(var_88).__module__}.{type(var_88).__qualname__}' == 'typesystem.fields.String'
    assert var_88.title == ''
    assert var_88.description == ''
    assert var_88.allow_null is False
    assert var_88.read_only is False
    assert var_88.allow_blank is False
    assert var_88.trim_whitespace is True
    assert var_88.max_length is None
    assert var_88.min_length is None
    assert var_88.format is None
    assert var_88.coerce_types is True
    assert var_88.pattern == '^test.*$'
    assert f'{type(var_88.pattern_regex).__module__}.{type(var_88.pattern_regex).__qualname__}' == 're.Pattern'
    var_89 = module_2.Integer(minimum=var_11)
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.fields.Integer'
    assert var_89.title == ''
    assert var_89.description == ''
    assert var_89.allow_null is False
    assert var_89.read_only is False
    assert var_89.minimum is False
    assert var_89.maximum is None
    assert var_89.exclusive_minimum is None
    assert var_89.exclusive_maximum is None
    assert var_89.multiple_of is None
    assert var_89.precision is None
    assert var_89.coerce_types is True
    var_90 = module_2.Boolean()
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_90.title == ''
    assert var_90.description == ''
    assert var_90.allow_null is False
    assert var_90.read_only is False
    assert var_90.coerce_types is True
    var_91 = module_3.IfThenElse(var_88, var_89, var_90)
    assert f'{type(var_91).__module__}.{type(var_91).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_91.title == ''
    assert var_91.description == ''
    assert var_91.allow_null is False
    assert var_91.read_only is False
    assert f'{type(var_91.if_clause).__module__}.{type(var_91.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_91.then_clause).__module__}.{type(var_91.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_91.else_clause).__module__}.{type(var_91.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_92 = module_1.to_json_schema(var_91)
    var_93 = module_2.String(pattern=var_7)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.String'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert var_93.allow_blank is False
    assert var_93.trim_whitespace is True
    assert var_93.max_length is None
    assert var_93.min_length is None
    assert var_93.format is None
    assert var_93.coerce_types is True
    assert var_93.pattern == '^test.*$'
    assert f'{type(var_93.pattern_regex).__module__}.{type(var_93.pattern_regex).__qualname__}' == 're.Pattern'
    var_94 = module_3.Not(var_93)
    assert f'{type(var_94).__module__}.{type(var_94).__qualname__}' == 'typesystem.composites.Not'
    assert var_94.title == ''
    assert var_94.description == ''
    assert var_94.allow_null is False
    assert var_94.read_only is False
    assert f'{type(var_94.negated).__module__}.{type(var_94.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_95 = module_1.to_json_schema(var_94)
    var_96 = module_4.Definitions()
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_96) == 0
    var_97 = module_2.String()
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.fields.String'
    assert var_97.title == ''
    assert var_97.description == ''
    assert var_97.allow_null is False
    assert var_97.read_only is False
    assert var_97.allow_blank is False
    assert var_97.trim_whitespace is True
    assert var_97.max_length is None
    assert var_97.min_length is None
    assert var_97.format is None
    assert var_97.coerce_types is True
    assert var_97.pattern is None
    assert var_97.pattern_regex is None
    var_98 = 'Person'
    var_99 = module_4.Reference(var_98, var_96)
    assert f'{type(var_99).__module__}.{type(var_99).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_99.title == ''
    assert var_99.description == ''
    assert var_99.allow_null is False
    assert var_99.read_only is False
    assert var_99.to == 'Person'
    assert f'{type(var_99.definitions).__module__}.{type(var_99.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_99.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    module_1.to_json_schema(var_99)

def test_case_55():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_1.to_json_schema(var_0)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = 'f'
    var_3 = {var_2: var_2}
    var_4 = var_0.clear()
    var_5 = None
    var_6 = module_2.Field(read_only=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Field'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_7 = module_1.from_json_schema(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Any'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    with pytest.raises(ValueError):
        module_1.to_json_schema(var_6, var_5)

def test_case_56():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = module_4.Definitions()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Float'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.minimum == 0
    assert var_9.maximum == 10
    assert var_9.exclusive_minimum is None
    assert var_9.exclusive_maximum is None
    assert var_9.multiple_of is None
    assert var_9.precision is None
    assert var_9.coerce_types is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_10 = 'exclusiveMinimum'
    var_11 = 'multipleOf'
    var_12 = 'integer'
    var_13 = 5
    var_14 = 2
    var_15 = {var_0: var_12, var_10: var_13, var_11: var_14}
    var_16 = True
    var_17 = module_4.Definitions()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_17) == 0
    var_18 = module_1.from_json_schema_type(var_15, var_12, var_16, var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Integer'
    assert var_18.default is None
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is True
    assert var_18.read_only is False
    assert var_18.minimum is None
    assert var_18.maximum is None
    assert var_18.exclusive_minimum == 5
    assert var_18.exclusive_maximum is None
    assert var_18.multiple_of == 2
    assert var_18.precision is None
    assert var_18.coerce_types is False
    var_19 = 'minLength'
    var_20 = 'maxLength'
    var_21 = 'pattern'
    var_22 = 'string'
    var_23 = 3
    var_24 = '^a.*z$'
    var_25 = {var_0: var_22, var_19: var_23, var_20: var_5, var_21: var_24}
    var_26 = False
    var_27 = module_4.Definitions()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_27) == 0
    var_28 = module_1.from_json_schema_type(var_25, var_22, var_26, var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.String'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.allow_blank is False
    assert var_28.trim_whitespace is True
    assert var_28.max_length == 10
    assert var_28.min_length == 3
    assert var_28.format is None
    assert var_28.coerce_types is False
    assert var_28.pattern == '^a.*z$'
    assert f'{type(var_28.pattern_regex).__module__}.{type(var_28.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_29 = 'default'
    var_30 = 'boolean'
    var_31 = {var_0: var_30, var_29: var_16}
    var_32 = module_4.Definitions()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_32) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_33 = module_1.from_json_schema_type(var_31, var_30, var_16, var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_33.default is True
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is True
    assert var_33.read_only is False
    assert var_33.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_34 = 'items'
    var_35 = 'minItems'
    var_36 = 'array'
    var_37 = {var_0: var_22}
    var_38 = {var_0: var_36, var_34: var_37, var_35: var_16}
    var_39 = False
    var_40 = module_4.Definitions()
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_40) == 0
    var_41 = module_1.from_json_schema_type(var_38, var_36, var_39, var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Array'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.items).__module__}.{type(var_41.items).__qualname__}' == 'typesystem.fields.String'
    assert var_41.additional_items is True
    assert var_41.min_items is True
    assert var_41.max_items is None
    assert var_41.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_42 = var_41.items
    var_43 = {var_0: var_22}
    var_44 = {var_0: var_3}
    var_45 = [var_43, var_44]
    var_46 = {var_0: var_36, var_34: var_45}
    var_47 = False
    var_48 = module_4.Definitions()
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_48) == 0
    var_49 = module_1.from_json_schema_type(var_46, var_36, var_47, var_48)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.fields.Array'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert f'{type(var_49.items).__module__}.{type(var_49.items).__qualname__}' == 'builtins.list'
    assert len(var_49.items) == 2
    assert var_49.additional_items is True
    assert var_49.min_items == 0
    assert var_49.max_items is None
    assert var_49.unique_items is False
    var_50 = var_49.items
    var_51 = var_49.items
    var_52 = len(var_51)
    assert var_52 == 2
    var_53 = var_49.items[var_47]
    var_54 = var_49.items[var_16]
    var_55 = 'properties'
    var_56 = 'required'
    var_57 = 'object'
    var_58 = 'name'
    var_59 = 'age'
    var_60 = {var_0: var_22}
    var_61 = {var_0: var_12}
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = [var_58]
    var_64 = {var_0: var_57, var_55: var_62, var_56: var_63}
    var_65 = False
    var_66 = module_4.Definitions()
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_66) == 0
    var_67 = module_1.from_json_schema_type(var_64, var_57, var_65, var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.fields.Object'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    assert f'{type(var_67.properties).__module__}.{type(var_67.properties).__qualname__}' == 'builtins.dict'
    assert len(var_67.properties) == 2
    assert var_67.pattern_properties == {}
    assert var_67.additional_properties is None
    assert var_67.property_names is None
    assert var_67.min_properties is None
    assert var_67.max_properties is None
    assert var_67.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_68 = var_67.properties[var_58]
    var_69 = var_67.properties[var_59]
    var_70 = 'patternProperties'
    var_71 = '^S_'
    var_72 = {var_0: var_22}
    var_73 = {var_71: var_72}
    var_74 = {var_0: var_57, var_70: var_73}
    var_75 = False
    var_76 = module_4.Definitions()
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_76) == 0
    var_77 = module_1.from_json_schema_type(var_74, var_57, var_75, var_76)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.Object'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert var_77.properties == {}
    assert f'{type(var_77.pattern_properties).__module__}.{type(var_77.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_77.pattern_properties) == 1
    assert var_77.additional_properties is None
    assert var_77.property_names is None
    assert var_77.min_properties is None
    assert var_77.max_properties is None
    assert var_77.required == []
    var_78 = 'additionalProperties'
    var_79 = False
    var_80 = {var_0: var_57, var_78: var_79}
    var_81 = False
    var_82 = module_4.Definitions()
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_82) == 0
    var_83 = module_1.from_json_schema_type(var_80, var_57, var_81, var_82)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.fields.Object'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert var_83.properties == {}
    assert var_83.pattern_properties == {}
    assert var_83.additional_properties is False
    assert var_83.property_names is None
    assert var_83.min_properties is None
    assert var_83.max_properties is None
    assert var_83.required == []
    var_84 = 'propertyNames'
    var_85 = '^[a-z]+$'
    var_86 = {var_21: var_85}
    var_87 = {var_0: var_57, var_84: var_86}
    var_88 = False
    var_89 = module_4.Definitions()
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_89) == 0
    var_90 = module_1.from_json_schema_type(var_87, var_57, var_88, var_89)
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'typesystem.fields.Object'
    assert var_90.title == ''
    assert var_90.description == ''
    assert var_90.allow_null is False
    assert var_90.read_only is False
    assert var_90.properties == {}
    assert var_90.pattern_properties == {}
    assert var_90.additional_properties is None
    assert f'{type(var_90.property_names).__module__}.{type(var_90.property_names).__qualname__}' == 'typesystem.fields.Union'
    assert var_90.min_properties is None
    assert var_90.max_properties is None
    assert var_90.required == []
    var_91 = var_90.property_names
    var_92 = 'test'
    var_93 = {var_0: var_22, var_29: var_92}
    var_94 = False
    var_95 = module_4.Definitions()
    assert f'{type(var_95).__module__}.{type(var_95).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_95) == 0
    var_96 = module_1.from_json_schema_type(var_93, var_22, var_94, var_95)
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.fields.String'
    assert var_96.default == 'test'
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
    var_97 = {var_0: var_3}
    var_98 = module_4.Definitions()
    assert f'{type(var_98).__module__}.{type(var_98).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_98) == 0
    var_99 = module_1.from_json_schema_type(var_97, var_3, var_16, var_98)
    assert f'{type(var_99).__module__}.{type(var_99).__qualname__}' == 'typesystem.fields.Float'
    assert var_99.default is None
    assert var_99.title == ''
    assert var_99.description == ''
    assert var_99.allow_null is True
    assert var_99.read_only is False
    assert var_99.minimum is None
    assert var_99.maximum is None
    assert var_99.exclusive_minimum is None
    assert var_99.exclusive_maximum is None
    assert var_99.multiple_of is None
    assert var_99.precision is None
    assert var_99.coerce_types is False
    var_100 = 'additionalItems'
    var_101 = {var_0: var_22}
    var_102 = {var_0: var_36, var_100: var_101}
    var_103 = False
    var_104 = module_4.Definitions()
    assert f'{type(var_104).__module__}.{type(var_104).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_104) == 0
    var_105 = module_1.from_json_schema_type(var_102, var_36, var_103, var_104)
    assert f'{type(var_105).__module__}.{type(var_105).__qualname__}' == 'typesystem.fields.Array'
    assert var_105.title == ''
    assert var_105.description == ''
    assert var_105.allow_null is False
    assert var_105.read_only is False
    assert var_105.items is None
    assert f'{type(var_105.additional_items).__module__}.{type(var_105.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_105.min_items == 0
    assert var_105.max_items is None
    assert var_105.unique_items is False
    var_106 = var_105.additional_items
    var_107 = {var_0: var_3}
    var_108 = {var_0: var_57, var_78: var_107}
    var_109 = False
    var_110 = module_4.Definitions()
    assert f'{type(var_110).__module__}.{type(var_110).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_110) == 0
    var_111 = module_1.from_json_schema_type(var_108, var_57, var_109, var_110)
    assert f'{type(var_111).__module__}.{type(var_111).__qualname__}' == 'typesystem.fields.Object'
    assert var_111.title == ''
    assert var_111.description == ''
    assert var_111.allow_null is False
    assert var_111.read_only is False
    assert var_111.properties == {}
    assert var_111.pattern_properties == {}
    assert f'{type(var_111.additional_properties).__module__}.{type(var_111.additional_properties).__qualname__}' == 'typesystem.fields.Float'
    assert var_111.property_names is None
    assert var_111.min_properties is None
    assert var_111.max_properties is None
    assert var_111.required == []
    var_112 = var_111.additional_properties

def test_case_57():
    var_0 = True
    var_1 = module_1.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = module_1.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.default == ''
    assert var_5.allow_blank is True
    assert var_5.trim_whitespace is True
    assert var_5.max_length is None
    assert var_5.min_length is None
    assert var_5.format is None
    assert var_5.coerce_types is False
    assert var_5.pattern is None
    assert var_5.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = 'integer'
    var_7 = {var_2: var_6}
    var_8 = module_1.from_json_schema(var_7)
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
    assert var_8.coerce_types is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_9 = 'boolean'
    var_10 = {var_2: var_9}
    var_11 = module_1.from_json_schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_12 = 'array'
    var_13 = {var_2: var_12}
    var_14 = module_1.from_json_schema(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Array'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.items is None
    assert var_14.additional_items is True
    assert var_14.min_items == 0
    assert var_14.max_items is None
    assert var_14.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_15 = 'object'
    var_16 = {var_2: var_15}
    var_17 = module_1.from_json_schema(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Object'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.properties == {}
    assert var_17.pattern_properties == {}
    assert var_17.additional_properties is None
    assert var_17.property_names is None
    assert var_17.min_properties is None
    assert var_17.max_properties is None
    assert var_17.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_18 = 'enum'
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'c'
    var_22 = [var_19, var_20, var_21]
    var_23 = {var_18: var_22}
    var_24 = module_1.from_json_schema(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Choice'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_24.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_25 = 'const'
    var_26 = 'fixed_value'
    var_27 = {var_25: var_26}
    var_28 = module_1.from_json_schema(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Const'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_29 = 'allOf'
    var_30 = {var_2: var_3}
    var_31 = 'minLength'
    var_32 = {var_31: var_0}
    var_33 = [var_30, var_32]
    var_34 = {var_29: var_33}
    var_35 = module_1.from_json_schema(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.all_of).__module__}.{type(var_35.all_of).__qualname__}' == 'builtins.list'
    assert len(var_35.all_of) == 2
    var_36 = 'anyOf'
    var_37 = {var_2: var_3}
    var_38 = {var_2: var_6}
    var_39 = [var_37, var_38]
    var_40 = {var_36: var_39}
    var_41 = module_1.from_json_schema(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Union'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.any_of).__module__}.{type(var_41.any_of).__qualname__}' == 'builtins.list'
    assert len(var_41.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_42 = 'oneOf'
    var_43 = {var_2: var_3}
    var_44 = {var_2: var_6}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_1.from_json_schema(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert f'{type(var_47.one_of).__module__}.{type(var_47.one_of).__qualname__}' == 'builtins.list'
    assert len(var_47.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_48 = 'not'
    var_49 = {var_2: var_3}
    var_50 = {var_48: var_49}
    var_51 = module_1.from_json_schema(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.composites.Not'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.negated).__module__}.{type(var_51.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_52 = 'if'
    var_53 = 'then'
    var_54 = 'else'
    var_55 = {var_2: var_3}
    var_56 = {var_31: var_0}
    var_57 = {var_2: var_6}
    var_58 = {var_52: var_55, var_53: var_56, var_54: var_57}
    var_59 = module_1.from_json_schema(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.if_clause).__module__}.{type(var_59.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_59.then_clause).__module__}.{type(var_59.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_59.else_clause).__module__}.{type(var_59.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_60 = 'maxLength'
    var_61 = 10
    var_62 = {var_2: var_3, var_31: var_0, var_60: var_61}
    var_63 = module_1.from_json_schema(var_62)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.fields.String'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert var_63.allow_blank is False
    assert var_63.trim_whitespace is True
    assert var_63.max_length == 10
    assert var_63.min_length is None
    assert var_63.format is None
    assert var_63.coerce_types is False
    assert var_63.pattern is None
    assert var_63.pattern_regex is None
    var_64 = {}
    var_65 = module_1.from_json_schema(var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Any'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    var_66 = module_4.Definitions()
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_66) == 0
    var_67 = '$ref'
    var_68 = '#/definitions/MyType'
    var_69 = {var_67: var_68}
    var_70 = module_1.from_json_schema(var_69, var_66)
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert var_70.to == '#/definitions/MyType'
    assert f'{type(var_70.definitions).__module__}.{type(var_70.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_70.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_71 = 'components'
    var_72 = 'schemas'
    var_73 = 'User'
    var_74 = 'properties'
    var_75 = 'name'
    var_76 = {var_2: var_3}
    var_77 = {var_75: var_76}
    var_78 = {var_2: var_15, var_74: var_77}
    var_79 = {var_73: var_78}
    var_80 = {var_72: var_79}
    var_81 = {var_71: var_80}
    var_82 = module_1.from_json_schema(var_81)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.Any'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False

def test_case_58():
    var_0 = True
    var_1 = module_1.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_1.TYPE_CONSTRAINTS == {'exclusiveMaximum', 'required', 'minProperties', 'uniqueItems', 'boolean_schema', 'maxProperties', 'properties', 'minItems', 'maxItems', 'dependencies', 'multipleOf', 'maxLength', 'patternProperties', 'maximum', 'pattern', 'items', 'type', 'exclusiveMinimum', 'additionalProperties', 'minimum', 'minLength', 'contains', 'additionalItems', 'propertyNames'}
    assert f'{type(module_1.definitions).__module__}.{type(module_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_1.definitions) == 1
    assert f'{type(module_1.JSONSchema).__module__}.{type(module_1.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_1.JSONSchema.title == ''
    assert module_1.JSONSchema.description == ''
    assert module_1.JSONSchema.allow_null is False
    assert module_1.JSONSchema.read_only is False
    assert f'{type(module_1.JSONSchema.any_of).__module__}.{type(module_1.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_1.JSONSchema.any_of) == 2
    var_2 = False
    var_3 = module_1.from_json_schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = module_1.from_json_schema(var_6)
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
    var_8 = 'number'
    var_9 = {var_4: var_8}
    var_10 = module_1.from_json_schema(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Float'
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
    var_11 = 'integer'
    var_12 = {var_4: var_11}
    var_13 = module_1.from_json_schema(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
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
    var_16 = module_1.from_json_schema(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_17 = 'array'
    var_18 = {var_4: var_17}
    var_19 = module_1.from_json_schema(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Array'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.items is None
    assert var_19.additional_items is True
    assert var_19.min_items == 0
    assert var_19.max_items is None
    assert var_19.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_20 = 'object'
    var_21 = {var_4: var_20}
    var_22 = module_1.from_json_schema(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Object'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.properties == {}
    assert var_22.pattern_properties == {}
    assert var_22.additional_properties is None
    assert var_22.property_names is None
    assert var_22.min_properties is None
    assert var_22.max_properties is None
    assert var_22.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_23 = 'minLength'
    var_24 = 5
    var_25 = {var_4: var_5, var_23: var_24}
    var_26 = module_1.from_json_schema(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.String'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.allow_blank is False
    assert var_26.trim_whitespace is True
    assert var_26.max_length is None
    assert var_26.min_length == 5
    assert var_26.format is None
    assert var_26.coerce_types is False
    assert var_26.pattern is None
    assert var_26.pattern_regex is None
    var_27 = 'enum'
    var_28 = 'a'
    var_29 = 'b'
    var_30 = 'c'
    var_31 = [var_28, var_29, var_30]
    var_32 = {var_27: var_31}
    var_33 = module_1.from_json_schema(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Choice'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_33.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_34 = 'const'
    var_35 = 'fixed_value'
    var_36 = {var_34: var_35}
    var_37 = module_1.from_json_schema(var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Const'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_38 = 'allOf'
    var_39 = 3
    var_40 = {var_4: var_5, var_23: var_39}
    var_41 = 'maxLength'
    var_42 = 10
    var_43 = {var_4: var_5, var_41: var_42}
    var_44 = [var_40, var_43]
    var_45 = {var_38: var_44}
    var_46 = module_1.from_json_schema(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.all_of).__module__}.{type(var_46.all_of).__qualname__}' == 'builtins.list'
    assert len(var_46.all_of) == 2
    var_47 = 'anyOf'
    var_48 = {var_4: var_5}
    var_49 = {var_4: var_8}
    var_50 = [var_48, var_49]
    var_51 = {var_47: var_50}
    var_52 = module_1.from_json_schema(var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Union'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert f'{type(var_52.any_of).__module__}.{type(var_52.any_of).__qualname__}' == 'builtins.list'
    assert len(var_52.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_53 = 'oneOf'
    var_54 = {var_4: var_5}
    var_55 = {var_4: var_11}
    var_56 = [var_54, var_55]
    var_57 = {var_53: var_56}
    var_58 = module_1.from_json_schema(var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert f'{type(var_58.one_of).__module__}.{type(var_58.one_of).__qualname__}' == 'builtins.list'
    assert len(var_58.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_59 = 'not'
    var_60 = {var_4: var_5}
    var_61 = {var_59: var_60}
    var_62 = module_1.from_json_schema(var_61)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.composites.Not'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert f'{type(var_62.negated).__module__}.{type(var_62.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_63 = 'if'
    var_64 = 'then'
    var_65 = 'else'
    var_66 = {var_4: var_5}
    var_67 = {var_23: var_24}
    var_68 = {var_4: var_8}
    var_69 = {var_63: var_66, var_64: var_67, var_65: var_68}
    var_70 = module_1.from_json_schema(var_69)
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert f'{type(var_70.if_clause).__module__}.{type(var_70.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_70.then_clause).__module__}.{type(var_70.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_70.else_clause).__module__}.{type(var_70.else_clause).__qualname__}' == 'typesystem.fields.Float'
    var_71 = '$ref'
    var_72 = '#/components/schemas/User'
    var_73 = {var_71: var_72}
    var_74 = module_1.from_json_schema(var_73)
    assert f'{type(var_74).__module__}.{type(var_74).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_74.title == ''
    assert var_74.description == ''
    assert var_74.allow_null is False
    assert var_74.read_only is False
    assert var_74.to == '#/components/schemas/User'
    assert f'{type(var_74.definitions).__module__}.{type(var_74.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_74.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_75 = '^[a-z]+$'
    var_76 = {var_4: var_5, var_23: var_39, var_41: var_42, var_59: var_75}
    var_77 = module_1.from_json_schema(var_76)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert f'{type(var_77.all_of).__module__}.{type(var_77.all_of).__qualname__}' == 'builtins.list'
    assert len(var_77.all_of) == 2
    var_78 = module_5.compile(var_75)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 're.Pattern'
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
    assert f'{type(module_5.Pattern.pattern).__module__}.{type(module_5.Pattern.pattern).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.Pattern.flags).__module__}.{type(module_5.Pattern.flags).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.Pattern.groups).__module__}.{type(module_5.Pattern.groups).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.Pattern.groupindex).__module__}.{type(module_5.Pattern.groupindex).__qualname__}' == 'builtins.getset_descriptor'
    var_79 = {}
    var_80 = module_1.from_json_schema(var_79)
    assert f'{type(var_80).__module__}.{type(var_80).__qualname__}' == 'typesystem.fields.Any'
    assert var_80.title == ''
    assert var_80.description == ''
    assert var_80.allow_null is False
    assert var_80.read_only is False
    var_81 = module_4.Definitions()
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_81) == 0
    var_82 = 'components'
    var_83 = 'schemas'
    var_84 = 'User'
    var_85 = 'properties'
    var_86 = 'name'
    var_87 = {var_4: var_5}
    var_88 = {var_86: var_87}
    var_89 = {var_4: var_20, var_85: var_88}
    var_90 = {var_84: var_89}
    var_91 = {var_83: var_90}
    var_92 = {var_82: var_91}
    var_93 = module_1.from_json_schema(var_92, var_81)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.Any'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False