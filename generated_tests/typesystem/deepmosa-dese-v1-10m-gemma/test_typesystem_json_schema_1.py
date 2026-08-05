# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import typesystem.composites as module_1
import enum as module_2
import typesystem.fields as module_3
import typesystem.schemas as module_4
import re as module_5

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
def test_case_1():
    var_0 = None
    module_0.from_json_schema(var_0)

def test_case_2():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_3():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.all_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.not_from_json_schema(var_0, var_0)

def test_case_9():
    var_0 = {}
    var_1 = None
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_0, var_1, var_1, var_1)

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
    var_2 = module_1.IfThenElse(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'

def test_case_11():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    module_0.any_of_from_json_schema(var_0, var_0)

def test_case_14():
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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

def test_case_16():
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = module_0.to_json_schema(var_0, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_17():
    var_0 = module_3.Integer()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = True
    var_1 = module_3.Boolean()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = module_2._EnumDict()
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.AllOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.all_of).__module__}.{type(var_1.all_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.all_of) == 0
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
def test_case_22():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_1.Not(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.negated) == 0
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    module_0.to_json_schema(var_2)

def test_case_23():
    var_0 = module_3.Integer()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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

def test_case_24():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_3 = module_0.any_of_from_json_schema(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5

def test_case_25():
    var_0 = module_3.Integer()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.from_json_schema(var_1, var_1)
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

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_3.Const(var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Const'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.const).__module__}.{type(var_1.const).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.const) == 0
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_2 = None
    var_3 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_1.validate(var_2)

def test_case_27():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_2 = module_1.IfThenElse(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_3 = module_0.to_json_schema(var_2)

def test_case_28():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_3 = module_0.from_json_schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5

def test_case_29():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.OneOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.one_of).__module__}.{type(var_1.one_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.one_of) == 0
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_2 = module_1.IfThenElse(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.composites.OneOf'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.composites.OneOf'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
def test_case_30():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.type_from_json_schema(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 5
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_4 = module_0.to_json_schema(var_1)
    assert var_4 is True
    var_5 = module_4.Definitions(**var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 2
    module_0.to_json_schema(var_5)

def test_case_31():
    var_0 = 'items'
    var_1 = 'minItems'
    var_2 = 'te'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = 'boolean'
    var_9 = {var_2: var_8}
    var_10 = 41
    var_11 = {var_0: var_7, var_2: var_9, var_1: var_10}
    var_12 = 'array'
    var_13 = False
    var_14 = None
    var_15 = module_0.from_json_schema_type(var_11, var_12, var_13, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Array'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.items).__module__}.{type(var_15.items).__qualname__}' == 'builtins.list'
    assert len(var_15.items) == 2
    assert var_15.additional_items is True
    assert var_15.min_items == 41
    assert var_15.max_items is None
    assert var_15.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_16 = var_15.items
    var_17 = var_15.items
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = var_15.additional_items

def test_case_32():
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = 'QPFRb4"/\'('
    var_2 = {var_1: var_0, var_1: var_0, var_1: var_0}
    var_3 = module_4.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ['QPFRb4"/\'(']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.properties).__module__}.{type(var_5.properties).__qualname__}' == 'builtins.dict'
    assert len(var_5.properties) == 1
    assert var_5.pattern_properties == {}
    assert var_5.additional_properties is None
    assert var_5.property_names is None
    assert var_5.min_properties is None
    assert var_5.max_properties is None
    assert var_5.required == ['QPFRb4"/\'(']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = module_1.IfThenElse(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.if_clause).__module__}.{type(var_6.if_clause).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_6.then_clause).__module__}.{type(var_6.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_6.else_clause).__module__}.{type(var_6.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_7 = module_0.to_json_schema(var_6)

def test_case_33():
    var_0 = 'properties'
    var_1 = 'patternProperties'
    var_2 = 'additionalProperties'
    var_3 = 'propertyNames'
    var_4 = 'id'
    var_5 = 'type'
    var_6 = 'integer'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = False
    var_10 = 'minLength'
    var_11 = 3
    var_12 = {var_10: var_11}
    var_13 = {var_0: var_8, var_1: var_8, var_2: var_9, var_3: var_12}
    var_14 = 'object'
    var_15 = True
    var_16 = None
    var_17 = module_0.from_json_schema_type(var_13, var_14, var_15, var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Object'
    assert var_17.default is None
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is True
    assert var_17.read_only is False
    assert f'{type(var_17.properties).__module__}.{type(var_17.properties).__qualname__}' == 'builtins.dict'
    assert len(var_17.properties) == 1
    assert f'{type(var_17.pattern_properties).__module__}.{type(var_17.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_17.pattern_properties) == 1
    assert var_17.additional_properties is False
    assert f'{type(var_17.property_names).__module__}.{type(var_17.property_names).__qualname__}' == 'typesystem.fields.Union'
    assert var_17.min_properties is None
    assert var_17.max_properties is None
    assert var_17.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}

def test_case_34():
    var_0 = 'minLength'
    var_1 = 'format'
    var_2 = 'pattern'
    var_3 = 'default'
    var_4 = 5
    var_5 = 10
    var_6 = 'email'
    var_7 = '^[a-z]+$'
    var_8 = 'test'
    var_9 = {var_0: var_4, var_7: var_5, var_1: var_6, var_2: var_7, var_3: var_8}
    var_10 = 'string'
    var_11 = False
    var_12 = None
    var_13 = module_0.from_json_schema_type(var_9, var_10, var_11, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.String'
    assert var_13.default == 'test'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.allow_blank is False
    assert var_13.trim_whitespace is True
    assert var_13.max_length is None
    assert var_13.min_length == 5
    assert var_13.format == 'email'
    assert var_13.coerce_types is False
    assert var_13.pattern == '^[a-z]+$'
    assert f'{type(var_13.pattern_regex).__module__}.{type(var_13.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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

def test_case_35():
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = True
    var_2 = module_3.Array(var_0, min_items=var_1, unique_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.String'
    assert var_2.additional_items is False
    assert var_2.min_items is True
    assert var_2.max_items is None
    assert var_2.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = None
    var_4 = module_0.to_json_schema(var_2, var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.items).__module__}.{type(var_5.items).__qualname__}' == 'typesystem.fields.String'
    assert var_5.additional_items is False
    assert var_5.min_items is True
    assert var_5.max_items is None
    assert var_5.unique_items is True

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = 'name'
    var_1 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_3.Integer()
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
    var_3 = module_3.Object(properties=var_1, required=var_0)
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
    assert var_3.required == 'name'
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.properties == {}
    assert var_5.pattern_properties == {}
    assert f'{type(var_5.additional_properties).__module__}.{type(var_5.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_5.property_names is None
    assert var_5.min_properties is None
    assert var_5.max_properties is None
    assert var_5.required == 'name'
    var_6 = None
    module_0.to_json_schema(var_6, var_6)

def test_case_37():
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = True
    var_2 = module_3.Array(additional_items=var_0, max_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert f'{type(var_2.additional_items).__module__}.{type(var_2.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_2.min_items is None
    assert var_2.max_items is True
    assert var_2.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = None
    var_4 = module_0.to_json_schema(var_2, var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.items is None
    assert f'{type(var_5.additional_items).__module__}.{type(var_5.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_5.min_items == 0
    assert var_5.max_items is True
    assert var_5.unique_items is False

def test_case_38():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_3.Integer()
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
    var_5 = [var_0]
    var_6 = module_3.Object(properties=var_4, required=var_5)
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
    assert var_6.required == ['name']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Object'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.properties).__module__}.{type(var_8.properties).__qualname__}' == 'builtins.dict'
    assert len(var_8.properties) == 2
    assert var_8.pattern_properties == {}
    assert var_8.additional_properties is True
    assert var_8.property_names is None
    assert var_8.min_properties is None
    assert var_8.max_properties is None
    assert var_8.required == ['name']

def test_case_39():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.AllOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.all_of).__module__}.{type(var_1.all_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.all_of) == 0
    var_2 = module_1.IfThenElse(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'

def test_case_40():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_3.Integer()
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
    var_5 = [var_0]
    var_6 = module_3.Object(properties=var_4, required=var_5)
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
    assert var_6.required == ['name']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = 'a'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = module_3.Choice(choices=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Choice'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.choices == [('a', None), ('b', None)]
    assert var_6.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = 5
    var_1 = True
    var_2 = module_3.String(max_length=var_1, min_length=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is True
    assert var_2.min_length == 5
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = module_3.Integer(minimum=var_0, maximum=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum == 0
    assert var_3.maximum == 100
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = 5
    var_1 = 100
    var_2 = None
    var_3 = module_3.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, multiple_of=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum == 5
    assert var_3.maximum == 100
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of == 100
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = module_3.Array(var_0)
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
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = module_3.Integer()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_1 = True
    var_2 = module_3.Array(var_0, min_items=var_1, unique_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.additional_items is False
    assert var_2.min_items is True
    assert var_2.max_items is None
    assert var_2.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = False
    var_1 = module_3.Object()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = 1
    var_2 = 5
    var_3 = True
    var_4 = module_3.Array(var_0, min_items=var_1, max_items=var_2, unique_items=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.items).__module__}.{type(var_4.items).__qualname__}' == 'typesystem.fields.String'
    assert var_4.additional_items is False
    assert var_4.min_items == 1
    assert var_4.max_items == 5
    assert var_4.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = module_3.Array()
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
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_2 = module_1.IfThenElse(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_3 = module_0.to_json_schema(var_2)
    var_4 = module_0.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'

@pytest.mark.xfail(strict=True)
def test_case_51():
    var_0 = ' nxa2@\\!nHz%2/{b`'
    var_1 = None
    var_2 = module_4.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == ' nxa2@\\!nHz%2/{b`'
    assert var_2.definitions is None
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = [var_2, var_2, var_2, var_2]
    var_4 = module_1.OneOf(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.one_of).__module__}.{type(var_4.one_of).__qualname__}' == 'builtins.list'
    assert len(var_4.one_of) == 4
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    module_0.to_json_schema(var_4, var_1)

def test_case_52():
    var_0 = 'email'
    var_1 = False
    var_2 = True
    var_3 = module_3.String(allow_blank=var_2, format=var_0)
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
    assert var_3.format == 'email'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_2 = module_1.IfThenElse(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_3 = module_0.to_json_schema(var_2)
    var_4 = module_0.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_5 = None
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
    var_7 = module_0.get_standard_properties(var_2)
    var_8 = module_3.Object(pattern_properties=var_5, additional_properties=var_2, property_names=var_4, min_properties=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Object'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.properties == {}
    assert var_8.pattern_properties == {}
    assert f'{type(var_8.additional_properties).__module__}.{type(var_8.additional_properties).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert f'{type(var_8.property_names).__module__}.{type(var_8.property_names).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_8.min_properties is None
    assert var_8.max_properties is None
    assert var_8.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_9 = module_0.to_json_schema(var_8)

@pytest.mark.xfail(strict=True)
def test_case_54():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = {}
    var_2 = None
    var_3 = module_0.type_from_json_schema(var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_4 = module_1.IfThenElse(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_5 = None
    var_6 = '$he'
    var_7 = {var_6: var_3}
    var_8 = module_3.Object(properties=var_2, pattern_properties=var_7, **var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Object'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.properties == {}
    assert f'{type(var_8.pattern_properties).__module__}.{type(var_8.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_8.pattern_properties) == 1
    assert var_8.additional_properties is True
    assert var_8.property_names is None
    assert var_8.min_properties is None
    assert var_8.max_properties is None
    assert var_8.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_9 = module_0.to_json_schema(var_8, var_5)
    module_0.from_json_schema(var_2)

def test_case_55():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.OneOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.one_of).__module__}.{type(var_1.one_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.one_of) == 0
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_2 = module_1.IfThenElse(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.composites.OneOf'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.composites.OneOf'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.composites.OneOf'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.composites.OneOf'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'

def test_case_56():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_0: var_4, var_1: var_4}
    var_6 = module_4.Definitions()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 0
    var_7 = module_0.if_then_else_from_json_schema(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.if_clause).__module__}.{type(var_7.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_7.then_clause).__module__}.{type(var_7.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_7.else_clause).__module__}.{type(var_7.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = 'if'
    var_1 = 'else'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = module_4.Definitions()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_7) == 0
    var_8 = module_0.if_then_else_from_json_schema(var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.if_clause).__module__}.{type(var_8.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_8.then_clause).__module__}.{type(var_8.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_8.else_clause).__module__}.{type(var_8.else_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'default'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'const'
    var_8 = {}
    var_9 = 'no'
    var_10 = {var_7: var_9}
    var_11 = 'maybe'
    var_12 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11}
    var_13 = module_4.Definitions()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13) == 0
    var_14 = module_0.if_then_else_from_json_schema(var_12, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_14.default == 'maybe'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.if_clause).__module__}.{type(var_14.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_14.then_clause).__module__}.{type(var_14.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_14.else_clause).__module__}.{type(var_14.else_clause).__qualname__}' == 'typesystem.fields.Const'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_59():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
    var_2 = None
    var_3 = module_3.Number(precision=var_2, multiple_of=var_2)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Number.numeric_type is None
    assert module_3.Number.errors == {'type': 'Must be a number.', 'null': 'May not be null.', 'integer': 'Must be an integer.', 'finite': 'Must be finite.', 'minimum': 'Must be greater than or equal to {minimum}.', 'exclusive_minimum': 'Must be greater than {exclusive_minimum}.', 'maximum': 'Must be less than or equal to {maximum}.', 'exclusive_maximum': 'Must be less than {exclusive_maximum}.', 'multiple_of': 'Must be a multiple of {multiple_of}.'}
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_3)

def test_case_60():
    var_0 = 5
    var_1 = module_3.Object(min_properties=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Object'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.properties == {}
    assert var_1.pattern_properties == {}
    assert var_1.additional_properties is True
    assert var_1.property_names is None
    assert var_1.min_properties == 5
    assert var_1.max_properties is None
    assert var_1.required == []
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_61():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = '#/components/schemas/MyType'
    var_3 = {var_1: var_2}
    var_4 = module_0.from_json_schema(var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == '#/components/schemas/MyType'
    assert f'{type(var_4.definitions).__module__}.{type(var_4.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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

def test_case_62():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'type'
    var_2 = 'enum'
    var_3 = 'string'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.from_json_schema(var_7, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.all_of).__module__}.{type(var_8.all_of).__qualname__}' == 'builtins.list'
    assert len(var_8.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
def test_case_63():
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = [var_0, var_0, var_0]
    var_2 = module_1.AllOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.all_of).__module__}.{type(var_2.all_of).__qualname__}' == 'builtins.list'
    assert len(var_2.all_of) == 3
    var_3 = None
    var_4 = module_1.IfThenElse(var_1, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'builtins.list'
    assert len(var_4.if_clause) == 3
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_5 = module_0.to_json_schema(var_2, var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.all_of).__module__}.{type(var_6.all_of).__qualname__}' == 'builtins.list'
    assert len(var_6.all_of) == 3
    module_1.IfThenElse(var_3, else_clause=var_4, **var_3)

def test_case_64():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'not'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.Not'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.negated).__module__}.{type(var_6.negated).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Not.errors == {'negated': 'Must not match.'}

def test_case_65():
    var_0 = 0
    var_1 = 100
    var_2 = 5
    var_3 = module_3.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum == 0
    assert var_3.maximum == 100
    assert var_3.exclusive_minimum == 5
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_66():
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = module_3.Integer()
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
    var_2 = (var_0, var_1)
    var_3 = module_3.Array(var_2)
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
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = 'items'
    var_6 = var_4[var_5]

def test_case_67():
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = 5
    var_2 = None
    var_3 = module_3.Integer(exclusive_maximum=var_1, coerce_types=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum == 5
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is None
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_68():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'oneOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = 'some_value'
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = module_0.one_of_from_json_schema(var_10, var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_11.default == 'some_value'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.one_of).__module__}.{type(var_11.one_of).__qualname__}' == 'builtins.list'
    assert len(var_11.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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

def test_case_69():
    var_0 = {}
    var_1 = {}
    var_2 = True
    var_3 = None
    var_4 = 5
    var_5 = []
    var_6 = module_3.Object(properties=var_0, pattern_properties=var_1, additional_properties=var_2, property_names=var_3, min_properties=var_3, max_properties=var_4, required=var_5)
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
    assert var_6.max_properties == 5
    assert var_6.required == []
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
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
def test_case_70():
    var_0 = 'components'
    var_1 = '$ref'
    var_2 = 'User'
    var_3 = 'type'
    var_4 = 'id'
    var_5 = 'integer'
    var_6 = {var_3: var_5}
    var_7 = {var_2: var_6}
    var_8 = {var_0: var_7, var_1: var_4}
    var_9 = module_4.Definitions()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_9) == 0
    module_0.from_json_schema(var_8, var_9)

@pytest.mark.xfail(strict=True)
def test_case_71():
    var_0 = module_3.String()
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = [var_0, var_0, var_0]
    var_2 = module_1.AllOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.all_of).__module__}.{type(var_2.all_of).__qualname__}' == 'builtins.list'
    assert len(var_2.all_of) == 3
    var_3 = None
    var_4 = module_1.IfThenElse(var_1, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'builtins.list'
    assert len(var_4.if_clause) == 3
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_5 = module_0.to_json_schema(var_2, var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'dependencies', 'propertyNames', 'items', 'minimum', 'multipleOf', 'required', 'maxLength', 'minProperties', 'boolean_schema', 'patternProperties', 'minLength', 'properties', 'pattern', 'maxItems', 'additionalItems', 'maxProperties', 'type', 'exclusiveMaximum', 'contains', 'minItems', 'maximum', 'additionalProperties', 'exclusiveMinimum', 'uniqueItems'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    module_0.from_json_schema(var_3)