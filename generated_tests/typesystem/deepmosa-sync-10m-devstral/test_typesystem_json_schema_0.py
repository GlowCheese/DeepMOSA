# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import typesystem.fields as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3
import enum as module_4
import re as module_5

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_1 = {}
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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

def test_case_3():
    var_0 = None
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.to_json_schema(var_0)

def test_case_5():
    pass

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

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.not_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

def test_case_12():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = {}
    var_3 = module_2.IfThenElse(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.if_clause).__module__}.{type(var_3.if_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_4 = module_0.to_json_schema(var_3)

def test_case_13():
    var_0 = {}
    var_1 = module_0.type_from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is True
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = {}
    module_0.to_json_schema(var_0, var_0)

def test_case_15():
    var_0 = []
    var_1 = module_2.AllOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.all_of == []
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
def test_case_16():
    var_0 = None
    var_1 = module_2.OneOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.one_of is None
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    module_0.to_json_schema(var_1, var_0)

def test_case_17():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_19():
    var_0 = True
    var_1 = None
    var_2 = module_1.Integer(maximum=var_0, exclusive_minimum=var_1, exclusive_maximum=var_0, precision=var_1, coerce_types=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is True
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is True
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = None
    var_3 = {}
    var_4 = None
    var_5 = module_0.type_from_json_schema(var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert f'{type(var_5.any_of).__module__}.{type(var_5.any_of).__qualname__}' == 'builtins.list'
    assert len(var_5.any_of) == 5
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_6 = module_1.Decimal(minimum=var_2, maximum=var_0, exclusive_minimum=var_0, precision=var_2, multiple_of=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.minimum is None
    assert var_6.maximum is False
    assert var_6.exclusive_minimum is False
    assert var_6.exclusive_maximum is None
    assert var_6.multiple_of is False
    assert var_6.precision is None
    assert var_6.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_7 = module_0.to_json_schema(var_6)

def test_case_21():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_3 = module_2.IfThenElse(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.if_clause).__module__}.{type(var_3.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_4 = module_0.to_json_schema(var_3)

def test_case_22():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = True
    var_3 = module_1.Array(additional_items=var_2, max_items=var_0, exact_items=var_2, unique_items=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.items is None
    assert var_3.additional_items is True
    assert var_3.min_items is True
    assert var_3.max_items is True
    assert var_3.unique_items is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = module_2.Not(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.Not'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.negated is None
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    module_0.to_json_schema(var_1)

def test_case_25():
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0}
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_4 = module_0.from_json_schema(var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5

def test_case_26():
    var_0 = None
    var_1 = None
    var_2 = module_1.String(min_length=var_1, pattern=var_0)
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
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_1 = {}
    var_2 = module_3.Reference(var_1, var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == {}
    assert var_2.definitions is None
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = module_0.from_json_schema(var_1, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_0.get_standard_properties(var_3)
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    module_0.to_json_schema(var_2)

def test_case_28():
    var_0 = None
    var_1 = {}
    var_2 = module_2.AllOf(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.all_of == {}
    var_3 = module_0.from_json_schema(var_1, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_0.to_json_schema(var_2)
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.all_of == []

def test_case_29():
    var_0 = {}
    var_1 = module_3.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_3 = module_2.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.Not'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.negated).__module__}.{type(var_5.negated).__qualname__}' == 'typesystem.fields.Union'

def test_case_31():
    var_0 = {}
    var_1 = module_1.String(**var_0)
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
    var_2 = {}
    var_3 = module_1.Integer(**var_2)
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
    var_4 = [var_1, var_3]
    var_5 = module_2.OneOf(var_4, **var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.one_of).__module__}.{type(var_5.one_of).__qualname__}' == 'builtins.list'
    assert len(var_5.one_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = bool(var_6 == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]})

def test_case_32():
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_3 = module_2.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_0.from_json_schema(var_1, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    var_5 = 'c%KS"lNk"C(0YER3}'
    var_6 = True
    var_7 = module_1.Field(title=var_5, description=var_5, allow_null=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Field'
    assert var_7.default is None
    assert var_7.title == 'c%KS"lNk"C(0YER3}'
    assert var_7.description == 'c%KS"lNk"C(0YER3}'
    assert var_7.allow_null is True
    assert var_7.read_only is False
    assert module_1.Field.errors == {}
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_7, var_1)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_1.Const(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Const'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.const == {None: None}
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_3 = module_2.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.fields.Const'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_0.from_json_schema(var_1, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_0.to_json_schema(var_3)
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.Not'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.negated).__module__}.{type(var_6.negated).__qualname__}' == 'typesystem.fields.Const'
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_3 = module_2.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_2.AllOf(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.all_of == {None: None}
    module_0.to_json_schema(var_5)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = {}
    var_1 = module_0.type_from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Union'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is True
    assert var_1.read_only is False
    assert f'{type(var_1.any_of).__module__}.{type(var_1.any_of).__qualname__}' == 'builtins.list'
    assert len(var_1.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_2 = True
    var_3 = None
    var_4 = module_1.String(trim_whitespace=var_2, max_length=var_3, min_length=var_2, coerce_types=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is True
    assert var_4.format is None
    assert var_4.coerce_types is None
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_0.to_json_schema(var_4)
    var_4.validate(var_4)

def test_case_36():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_1.String(**var_1)
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
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Object(properties=var_3, **var_4)
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
    assert var_5.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = bool(var_6 == {'type': 'object', 'properties': {'name': {'type': 'string'}}})

def test_case_37():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_1.String(**var_1)
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
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_3.Schema(var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['name']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = bool(var_6 == {'type': 'object', 'properties': {'name': {'type': 'string'}}})

def test_case_38():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_1.Choice(choices=var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Choice'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.choices == [('a', 1), ('b', 2)]
    assert var_8.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_9 = module_0.to_json_schema(var_8)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_39():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_1.String(**var_2)
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4 == {'type': ['string', 'null']})

def test_case_40():
    var_0 = 10
    var_1 = {}
    var_2 = module_1.String(max_length=var_0, **var_1)
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = bool(var_3 == {'type': 'string', 'maxLength': 10})

def test_case_41():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_1.String(format=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = bool(var_2 == {'type': 'string', 'format': 'email'})

def test_case_42():
    var_0 = {}
    var_1 = module_1.Integer(**var_0)
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = bool(var_2 == {'type': 'integer'})
    assert var_3 is True

def test_case_43():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_1.Integer(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4 == {'type': ['integer', 'null']})

def test_case_44():
    var_0 = 0
    var_1 = {}
    var_2 = module_1.Integer(minimum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum == 0
    assert var_2.maximum is None
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = bool(var_3 == {'type': 'integer', 'minimum': 0})
    assert var_4 is True

def test_case_45():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_1.Boolean(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4 == {'type': ['boolean', 'null']})

def test_case_46():
    var_0 = {}
    var_1 = module_1.Array(**var_0)
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_0 = {}
    var_1 = module_1.String(**var_0)
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
    var_2 = {}
    var_3 = module_1.Array(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.String'
    assert var_3.additional_items is False
    assert var_3.min_items is None
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4 == {'type': 'array', 'items': {'type': 'string'}})

def test_case_48():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.Object(required=var_1, **var_2)
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
    assert var_3.required == ['name']
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4 == {'type': 'object', 'required': ['name']})

def test_case_49():
    var_0 = None
    var_1 = module_1.Object(properties=var_0)
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_0 = []
    var_1 = {}
    var_2 = module_3.Definitions(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = '$ref'
    var_4 = '#/test'
    var_5 = {var_3: var_4}
    var_6 = module_0.from_json_schema(var_5, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.to == '#/test'
    assert f'{type(var_6.definitions).__module__}.{type(var_6.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    var_7 = var_6.to
    assert var_7 == '#/test'

def test_case_51():
    var_0 = 'type'
    var_1 = 'enum'
    var_2 = 'string'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = len(var_6)
    assert var_8 == 2

def test_case_52():
    var_0 = 'allOf'
    var_1 = 'string'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.all_of).__module__}.{type(var_6.all_of).__qualname__}' == 'builtins.list'
    assert len(var_6.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = len(var_1)

def test_case_53():
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 2

def test_case_54():
    var_0 = module_4._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.String(**var_0)
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
    var_2 = None
    var_3 = {}
    var_4 = module_1.Integer(**var_3)
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
    var_5 = {}
    var_6 = module_2.IfThenElse(var_1, var_2, var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.if_clause).__module__}.{type(var_6.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_6.then_clause).__module__}.{type(var_6.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_6.else_clause).__module__}.{type(var_6.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = module_0.from_json_schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.if_clause).__module__}.{type(var_8.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_8.then_clause).__module__}.{type(var_8.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_8.else_clause).__module__}.{type(var_8.else_clause).__qualname__}' == 'typesystem.fields.Integer'

def test_case_55():
    var_0 = 'components'
    var_1 = '$ref'
    var_2 = 'schemas'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_2: var_6}
    var_8 = '#/components/schemas/test'
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = module_0.from_json_schema(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.to == '#/components/schemas/test'
    assert f'{type(var_10.definitions).__module__}.{type(var_10.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_10.definitions) == 1
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'

def test_case_56():
    var_0 = 'key'
    var_1 = {}
    var_2 = module_1.String(**var_1)
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
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Object(pattern_properties=var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.properties == {}
    assert f'{type(var_5.pattern_properties).__module__}.{type(var_5.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_5.pattern_properties) == 1
    assert var_5.additional_properties is True
    assert var_5.property_names is None
    assert var_5.min_properties is None
    assert var_5.max_properties is None
    assert var_5.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_0 = 'key'
    var_1 = {}
    var_2 = module_1.String(**var_1)
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
    var_3 = {var_0: var_2}
    var_4 = module_1.Object(pattern_properties=var_3, **var_1)
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
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_6 = None
    var_7 = module_0.type_from_json_schema(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert f'{type(var_7.pattern_properties).__module__}.{type(var_7.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_7.pattern_properties) == 1
    assert var_7.additional_properties is True
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    var_8 = var_5.__contains__(var_6)
    assert var_8 is False

def test_case_58():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'patternProperties'
    var_3 = 'additionalProperties'
    var_4 = 'minProperties'
    var_5 = 'maxProperties'
    var_6 = 'required'
    var_7 = 'default'
    var_8 = 'object'
    var_9 = 'name'
    var_10 = 'age'
    var_11 = 'string'
    var_12 = {var_0: var_11}
    var_13 = 'integer'
    var_14 = {var_0: var_13}
    var_15 = {var_9: var_12, var_10: var_14}
    var_16 = '^S_'
    var_17 = {var_0: var_11}
    var_18 = {var_16: var_17}
    var_19 = False
    var_20 = 1
    var_21 = 10
    var_22 = [var_9]
    var_23 = 'John'
    var_24 = 30
    var_25 = {var_9: var_23, var_10: var_24}
    var_26 = {var_0: var_8, var_1: var_15, var_2: var_18, var_3: var_19, var_4: var_20, var_5: var_21, var_6: var_22, var_7: var_25}
    var_27 = []
    var_28 = {}
    var_29 = module_3.Definitions(*var_27, **var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_29) == 0
    var_30 = module_0.from_json_schema_type(var_26, var_8, var_19, var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Object'
    assert var_30.default == {'name': 'John', 'age': 30}
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.properties).__module__}.{type(var_30.properties).__qualname__}' == 'builtins.dict'
    assert len(var_30.properties) == 2
    assert f'{type(var_30.pattern_properties).__module__}.{type(var_30.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_30.pattern_properties) == 1
    assert var_30.additional_properties is False
    assert var_30.property_names is None
    assert var_30.min_properties == 1
    assert var_30.max_properties == 10
    assert var_30.required == ['name']
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_31 = var_30.properties[var_9]
    var_32 = var_30.properties[var_10]
    var_33 = var_30.pattern_properties[var_16]
    var_34 = var_30.additional_properties
    assert var_34 is False
    var_35 = var_30.min_properties
    assert var_35 == 1
    var_36 = var_30.max_properties
    assert var_36 == 10
    var_37 = var_30.required
    var_38 = bool(var_30.required == ['name'])
    assert var_38 is True
    var_39 = var_30.default
    var_40 = bool(var_30.default == {'name': 'John', 'age': 30})
    assert var_40 is True
    var_41 = var_30.allow_null
    assert var_41 is False

def test_case_59():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'format'
    var_4 = 'pattern'
    var_5 = 'default'
    var_6 = 'string'
    var_7 = 5
    var_8 = 100
    var_9 = 'email'
    var_10 = '^[a-zA-Z0-9]+$'
    var_11 = 'test@example.com'
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = []
    var_14 = {}
    var_15 = module_3.Definitions(*var_13, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_15) == 0
    var_16 = False
    var_17 = module_0.from_json_schema_type(var_12, var_6, var_16, var_15)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.String'
    assert var_17.default == 'test@example.com'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.allow_blank is False
    assert var_17.trim_whitespace is True
    assert var_17.max_length == 100
    assert var_17.min_length == 5
    assert var_17.format == 'email'
    assert var_17.coerce_types is False
    assert var_17.pattern == '^[a-zA-Z0-9]+$'
    assert f'{type(var_17.pattern_regex).__module__}.{type(var_17.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_18 = var_17.min_length
    assert var_18 == 5
    var_19 = var_17.max_length
    assert var_19 == 100
    var_20 = var_17.format
    assert var_20 == 'email'
    var_21 = var_17.pattern
    assert var_21 == '^[a-zA-Z0-9]+$'
    var_22 = var_17.default
    assert var_22 == 'test@example.com'
    var_23 = var_17.allow_null
    assert var_23 is False

def test_case_60():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = module_3.Definitions(*var_6, **var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_7) == 2
    var_8 = module_0.type_from_json_schema(var_5, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Array'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.items).__module__}.{type(var_8.items).__qualname__}' == 'typesystem.fields.String'
    assert var_8.additional_items is True
    assert var_8.min_items == 0
    assert var_8.max_items is None
    assert var_8.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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

def test_case_61():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_1.String(**var_1)
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
    var_3 = {var_0: var_2}
    var_4 = '^S_'
    var_5 = {}
    var_6 = module_1.String(**var_5)
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
    var_7 = {var_4: var_6}
    var_8 = False
    var_9 = {}
    var_10 = module_1.String(**var_9)
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
    var_11 = 10
    var_12 = 1
    var_13 = [var_0]
    var_14 = {}
    var_15 = module_1.Object(properties=var_3, pattern_properties=var_7, additional_properties=var_8, property_names=var_10, min_properties=var_12, max_properties=var_11, required=var_13, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Object'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.properties).__module__}.{type(var_15.properties).__qualname__}' == 'builtins.dict'
    assert len(var_15.properties) == 1
    assert f'{type(var_15.pattern_properties).__module__}.{type(var_15.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_15.pattern_properties) == 1
    assert var_15.additional_properties is False
    assert f'{type(var_15.property_names).__module__}.{type(var_15.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_15.min_properties == 1
    assert var_15.max_properties == 10
    assert var_15.required == ['name']
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_16 = module_0.to_json_schema(var_15)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_17 = bool(var_16 == {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'patternProperties': {'^S_': {'type': 'string'}}, 'additionalProperties': False, 'propertyNames': {'type': 'string'}, 'maxProperties': 10, 'minProperties': 1, 'required': ['name']})

@pytest.mark.xfail(strict=True)
def test_case_62():
    var_0 = None
    var_1 = {}
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
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_3 = module_3.Schema(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.from_json_schema(var_1, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    var_5 = module_0.to_json_schema(var_3)
    var_6 = module_0.to_json_schema(var_2)
    var_7 = module_1.Object(properties=var_1, additional_properties=var_0, property_names=var_3, min_properties=var_0, max_properties=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is None
    assert f'{type(var_7.property_names).__module__}.{type(var_7.property_names).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = module_0.to_json_schema(var_7)
    module_5.finditer(var_0, var_0)

def test_case_63():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = {}
    var_6 = module_3.Definitions(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 0
    var_7 = module_0.type_from_json_schema(var_3, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Const'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.const is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_8 = var_7.const
    assert var_8 is None

def test_case_64():
    var_0 = 'string_field'
    var_1 = {}
    var_2 = module_1.String(**var_1)
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
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_3.Definitions(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 1
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = bool(var_7 == {'components': {'schemas': {'string_field': {'type': 'string'}}}})

def test_case_65():
    var_0 = {}
    var_1 = module_1.String(**var_0)
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
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.Array(var_1, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert f'{type(var_5.items).__module__}.{type(var_5.items).__qualname__}' == 'typesystem.fields.String'
    assert var_5.additional_items is False
    assert var_5.min_items is None
    assert var_5.max_items is None
    assert var_5.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = bool(var_6 == {'type': ['array', 'null'], 'items': {'type': 'string'}})

def test_case_66():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_1.String(**var_1)
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
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_1.Object(properties=var_3, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.default is None
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is True
    assert var_7.read_only is False
    assert f'{type(var_7.properties).__module__}.{type(var_7.properties).__qualname__}' == 'builtins.dict'
    assert len(var_7.properties) == 1
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is True
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = module_0.to_json_schema(var_7)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_9 = bool(var_8 == {'type': ['object', 'null'], 'properties': {'name': {'type': 'string'}}})

def test_case_67():
    var_0 = 'if'
    var_1 = 'default'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'default_value'
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = []
    var_8 = module_3.Definitions(*var_7, **var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 1
    var_9 = module_0.if_then_else_from_json_schema(var_6, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_9.default == 'default_value'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.if_clause).__module__}.{type(var_9.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_9.then_clause).__module__}.{type(var_9.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_9.else_clause).__module__}.{type(var_9.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maximum', 'dependencies', 'additionalProperties', 'contains', 'exclusiveMaximum', 'properties', 'minLength', 'maxProperties', 'propertyNames', 'maxItems', 'minProperties', 'required', 'patternProperties', 'multipleOf', 'minimum', 'minItems', 'exclusiveMinimum', 'items', 'uniqueItems', 'additionalItems', 'maxLength', 'boolean_schema', 'type', 'pattern'}
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
    var_12 = var_9.else_clause
    var_13 = var_9.default
    assert var_13 == 'default_value'