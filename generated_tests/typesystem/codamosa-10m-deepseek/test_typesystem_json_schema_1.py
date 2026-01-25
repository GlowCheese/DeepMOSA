# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.composites as module_3
import re as module_4

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
def test_case_1():
    var_0 = None
    module_0.from_json_schema(var_0)

def test_case_2():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = '3/K*#YpAPiULR'
    var_3 = True
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_1, var_2, var_3, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = {}
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = None
    module_0.enum_from_json_schema(var_2, var_2)

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
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_10():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.get_standard_properties(var_1)

def test_case_11():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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

def test_case_12():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    var_2 = None
    var_3 = module_0.to_json_schema(var_1, var_2)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

def test_case_14():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    var_2 = None
    var_3 = module_0.to_json_schema(var_1, var_2)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_4 = module_0.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5

def test_case_15():
    var_0 = True
    var_1 = module_1.String()
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_16():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_3.Not(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.fields.Any'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_0.to_json_schema(var_2)

def test_case_18():
    var_0 = {}
    var_1 = module_3.AllOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.all_of == {}
    var_2 = module_0.to_json_schema(var_1, var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.to_json_schema(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_3.Not(var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.fields.Any'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_3 = None
    var_4 = None
    var_5 = None
    var_6 = module_0.to_json_schema(var_2, var_3)
    var_7 = module_0.from_json_schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.Not'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.negated).__module__}.{type(var_7.negated).__qualname__}' == 'typesystem.fields.Any'
    module_0.to_json_schema(var_5, var_4)

def test_case_20():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
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
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = module_3.IfThenElse(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_5 = module_3.Not(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.Not'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.negated).__module__}.{type(var_5.negated).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_6 = module_0.to_json_schema(var_5)
    var_7 = module_0.get_valid_types(var_0)
    var_8 = [var_3, var_1, var_1]
    var_9 = module_3.OneOf(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.one_of).__module__}.{type(var_9.one_of).__qualname__}' == 'builtins.list'
    assert len(var_9.one_of) == 3
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_10 = module_0.to_json_schema(var_9)

def test_case_21():
    var_0 = module_3.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = True
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
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_1.Integer()
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
    var_6 = module_0.to_json_schema(var_3)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = None
    var_3 = module_1.Const(var_2, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Const'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.const is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_4 = None
    var_5 = module_0.to_json_schema(var_3, var_4)
    var_6 = var_3.serialize(var_4)
    var_7 = None
    var_8 = module_0.to_json_schema(var_3)
    var_9 = {}
    module_0.to_json_schema(var_7, var_9)

def test_case_23():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    var_2 = None
    var_3 = module_3.IfThenElse(var_1, var_2, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.if_clause).__module__}.{type(var_3.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.if_clause).__module__}.{type(var_5.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_5.then_clause).__module__}.{type(var_5.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_5.else_clause).__module__}.{type(var_5.else_clause).__qualname__}' == 'typesystem.fields.Any'

def test_case_24():
    var_0 = module_2.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
def test_case_25():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
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
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = None
    var_5 = module_0.to_json_schema(var_3, var_4)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_6 = module_0.to_json_schema(var_1)
    assert var_6 is True
    var_7 = module_0.from_json_schema(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Union'
    assert var_7.default is None
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.any_of).__module__}.{type(var_7.any_of).__qualname__}' == 'builtins.list'
    assert len(var_7.any_of) == 5
    var_8 = module_2.Definitions(**var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 2
    module_0.to_json_schema(var_8, var_5)

def test_case_26():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = '$ref'
    var_3 = '#/components/schemas/Example'
    var_4 = {var_2: var_3}
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.to == '#/components/schemas/Example'
    assert f'{type(var_5.definitions).__module__}.{type(var_5.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5.definitions) == 0
    assert module_2.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_2.Reference.target).__module__}.{type(module_2.Reference.target).__qualname__}' == 'builtins.property'
    var_6 = 'type'
    var_7 = 'minLength'
    var_8 = 'string'
    var_9 = 5
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.from_json_schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.String'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.allow_blank is False
    assert var_11.trim_whitespace is True
    assert var_11.max_length is None
    assert var_11.min_length == 5
    assert var_11.format is None
    assert var_11.coerce_types is False
    assert var_11.pattern is None
    assert var_11.pattern_regex is None
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_12 = 'enum'
    var_13 = 'red'
    var_14 = 'green'
    var_15 = 'blue'
    var_16 = [var_13, var_14, var_15]
    var_17 = {var_12: var_16}
    var_18 = module_0.from_json_schema(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Choice'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.choices == [('red', 'red'), ('green', 'green'), ('blue', 'blue')]
    assert var_18.coerce_types is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_19 = 'const'
    var_20 = {var_19: var_14}
    var_21 = module_0.from_json_schema(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Const'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.const == 'green'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_22 = 'allOf'
    var_23 = {var_6: var_8}
    var_24 = 3
    var_25 = {var_7: var_24}
    var_26 = [var_23, var_25]
    var_27 = {var_22: var_26}
    var_28 = module_0.from_json_schema(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.all_of).__module__}.{type(var_28.all_of).__qualname__}' == 'builtins.list'
    assert len(var_28.all_of) == 2
    var_29 = 'anyOf'
    var_30 = 'number'
    var_31 = {var_29: var_6}
    var_32 = module_0.from_json_schema(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Union'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.any_of).__module__}.{type(var_32.any_of).__qualname__}' == 'builtins.list'
    assert len(var_32.any_of) == 4
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_33 = 'oneOf'
    var_34 = {var_6: var_8}
    var_35 = {var_6: var_30}
    var_36 = [var_34, var_35]
    var_37 = {var_33: var_36}
    var_38 = module_0.from_json_schema(var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert f'{type(var_38.one_of).__module__}.{type(var_38.one_of).__qualname__}' == 'builtins.list'
    assert len(var_38.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_39 = 'not'
    var_40 = {var_6: var_8}
    var_41 = {var_39: var_40}
    var_42 = module_0.from_json_schema(var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.composites.Not'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert f'{type(var_42.negated).__module__}.{type(var_42.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_43 = 'if'
    var_44 = 'then'
    var_45 = 'else'
    var_46 = {var_6: var_8}
    var_47 = {var_7: var_9}
    var_48 = {var_6: var_30}
    var_49 = {var_43: var_46, var_44: var_47, var_45: var_48}
    var_50 = module_0.from_json_schema(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert f'{type(var_50.if_clause).__module__}.{type(var_50.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_50.then_clause).__module__}.{type(var_50.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_50.else_clause).__module__}.{type(var_50.else_clause).__qualname__}' == 'typesystem.fields.Float'
    var_51 = 'hello'
    var_52 = 'world'
    var_53 = [var_51, var_52]
    var_54 = {var_6: var_8, var_7: var_9, var_12: var_53}
    var_55 = module_0.from_json_schema(var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert f'{type(var_55.all_of).__module__}.{type(var_55.all_of).__qualname__}' == 'builtins.list'
    assert len(var_55.all_of) == 2

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = {}
    var_1 = module_3.AllOf(var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.all_of == {}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.all_of == []
    var_4 = None
    var_5 = None
    module_0.to_json_schema(var_5, var_4)

def test_case_28():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_3.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = module_1.String()
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_1.Integer()
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
    var_6 = module_0.to_json_schema(var_5)
    var_7 = module_1.Float()
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
    var_9 = module_1.Boolean()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_1.String()
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
    var_12 = module_1.Array(var_11)
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
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_13 = module_0.to_json_schema(var_12)
    var_14 = 'name'
    var_15 = module_1.String()
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
    var_16 = {var_14: var_15}
    var_17 = module_1.Object(properties=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Object'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.properties).__module__}.{type(var_17.properties).__qualname__}' == 'builtins.dict'
    assert len(var_17.properties) == 1
    assert var_17.pattern_properties == {}
    assert var_17.additional_properties is True
    assert var_17.property_names is None
    assert var_17.min_properties is None
    assert var_17.max_properties is None
    assert var_17.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_18 = module_0.to_json_schema(var_17)
    var_19 = module_1.Const(var_14)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Const'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.const == 'name'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_20 = module_0.to_json_schema(var_19)
    var_21 = module_1.String()
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
    var_22 = module_1.Integer()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Integer'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.minimum is None
    assert var_22.maximum is None
    assert var_22.exclusive_minimum is None
    assert var_22.exclusive_maximum is None
    assert var_22.multiple_of is None
    assert var_22.precision is None
    assert var_22.coerce_types is True
    var_23 = module_1.String()
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
    var_24 = module_1.Integer()
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
    var_26 = module_3.OneOf(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.one_of).__module__}.{type(var_26.one_of).__qualname__}' == 'builtins.list'
    assert len(var_26.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_27 = module_0.to_json_schema(var_26)
    var_28 = module_1.String()
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
    var_29 = module_1.Integer()
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
    var_30 = [var_28, var_29]
    var_31 = module_3.AllOf(var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert f'{type(var_31.all_of).__module__}.{type(var_31.all_of).__qualname__}' == 'builtins.list'
    assert len(var_31.all_of) == 2
    var_32 = module_1.Integer()
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
    var_33 = module_3.IfThenElse(var_4, var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.if_clause).__module__}.{type(var_33.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_33.then_clause).__module__}.{type(var_33.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_33.else_clause).__module__}.{type(var_33.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_34 = module_0.to_json_schema(var_33)
    var_35 = module_1.String()
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
    var_36 = module_3.Not(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.composites.Not'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert f'{type(var_36.negated).__module__}.{type(var_36.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_37 = module_0.from_json_schema(var_20)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Const'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.const == 'name'

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_2.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = module_0.type_from_json_schema(var_3, var_0)
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    var_5 = 'null'
    var_6 = [var_2, var_5]
    var_7 = {var_1: var_6}
    var_8 = module_0.type_from_json_schema(var_7, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.default is None
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is True
    assert var_8.read_only is False
    assert var_8.allow_blank is True
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format is None
    assert var_8.coerce_types is False
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_9 = []
    var_10 = {var_1: var_9}
    var_11 = module_0.type_from_json_schema(var_10, var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Union'
    assert var_11.default is None
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is True
    assert var_11.read_only is False
    assert f'{type(var_11.any_of).__module__}.{type(var_11.any_of).__qualname__}' == 'builtins.list'
    assert len(var_11.any_of) == 5
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_12 = [var_5]
    var_13 = {var_1: var_12}
    var_14 = module_0.type_from_json_schema(var_13, var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Const'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.const is None
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_15 = 'invalid'
    var_16 = {var_1: var_15}
    module_0.type_from_json_schema(var_16, var_0)

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
    var_1 = module_1.Float()
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_1.Boolean()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_1.String()
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
    var_6 = module_1.Array(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.items).__module__}.{type(var_6.items).__qualname__}' == 'typesystem.fields.String'
    assert var_6.additional_items is False
    assert var_6.min_items is None
    assert var_6.max_items is None
    assert var_6.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_7 = module_0.to_json_schema(var_6)
    var_8 = 'name'
    var_9 = module_1.String()
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
    var_10 = {var_8: var_9}
    var_11 = module_1.Object(properties=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Object'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.properties).__module__}.{type(var_11.properties).__qualname__}' == 'builtins.dict'
    assert len(var_11.properties) == 1
    assert var_11.pattern_properties == {}
    assert var_11.additional_properties is True
    assert var_11.property_names is None
    assert var_11.min_properties is None
    assert var_11.max_properties is None
    assert var_11.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_12 = module_0.to_json_schema(var_11)
    var_13 = module_1.Const(var_8)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Const'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.const == 'name'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_14 = module_0.to_json_schema(var_13)
    var_15 = module_1.String()
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
    var_16 = module_1.Integer()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Integer'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum is None
    assert var_16.maximum is None
    assert var_16.exclusive_minimum is None
    assert var_16.exclusive_maximum is None
    assert var_16.multiple_of is None
    assert var_16.precision is None
    assert var_16.coerce_types is True
    var_17 = module_1.Integer()
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
    var_18 = module_1.Integer()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Integer'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.minimum is None
    assert var_18.maximum is None
    assert var_18.exclusive_minimum is None
    assert var_18.exclusive_maximum is None
    assert var_18.multiple_of is None
    assert var_18.precision is None
    assert var_18.coerce_types is True
    var_19 = module_4.purge()
    assert module_4.ASCII == module_4.RegexFlag.ASCII
    assert module_4.A == module_4.RegexFlag.ASCII
    assert module_4.IGNORECASE == module_4.RegexFlag.IGNORECASE
    assert module_4.I == module_4.RegexFlag.IGNORECASE
    assert module_4.LOCALE == module_4.RegexFlag.LOCALE
    assert module_4.L == module_4.RegexFlag.LOCALE
    assert module_4.UNICODE == module_4.RegexFlag.UNICODE
    assert module_4.U == module_4.RegexFlag.UNICODE
    assert module_4.MULTILINE == module_4.RegexFlag.MULTILINE
    assert module_4.M == module_4.RegexFlag.MULTILINE
    assert module_4.DOTALL == module_4.RegexFlag.DOTALL
    assert module_4.S == module_4.RegexFlag.DOTALL
    assert module_4.VERBOSE == module_4.RegexFlag.VERBOSE
    assert module_4.X == module_4.RegexFlag.VERBOSE
    assert module_4.TEMPLATE == module_4.RegexFlag.TEMPLATE
    assert module_4.T == module_4.RegexFlag.TEMPLATE
    assert module_4.DEBUG == module_4.RegexFlag.DEBUG
    var_20 = module_1.Integer()
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
    var_21 = module_3.IfThenElse(var_0, var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.if_clause).__module__}.{type(var_21.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_21.then_clause).__module__}.{type(var_21.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_21.else_clause).__module__}.{type(var_21.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_22 = module_0.from_json_schema(var_7)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Array'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.items).__module__}.{type(var_22.items).__qualname__}' == 'typesystem.fields.String'
    assert var_22.additional_items is False
    assert var_22.min_items == 0
    assert var_22.max_items is None
    assert var_22.unique_items is False

def test_case_31():
    var_0 = module_3.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_1.Integer()
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_1.Array(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.items).__module__}.{type(var_4.items).__qualname__}' == 'typesystem.fields.String'
    assert var_4.additional_items is False
    assert var_4.min_items is None
    assert var_4.max_items is None
    assert var_4.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_5 = 'name'
    var_6 = {var_5: var_3}
    var_7 = module_1.Object(properties=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
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
    var_9 = module_1.String()
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
    var_10 = module_1.Integer()
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
    var_11 = module_1.String()
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
    var_12 = module_1.Integer()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Integer'
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
    assert var_12.coerce_types is True
    var_13 = [var_11, var_12]
    var_14 = module_3.OneOf(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.one_of).__module__}.{type(var_14.one_of).__qualname__}' == 'builtins.list'
    assert len(var_14.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_15 = module_0.to_json_schema(var_14)
    var_16 = module_1.String()
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
    var_17 = module_1.Integer()
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
    var_18 = [var_16, var_17]
    var_19 = module_3.AllOf(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.all_of).__module__}.{type(var_19.all_of).__qualname__}' == 'builtins.list'
    assert len(var_19.all_of) == 2
    var_20 = module_3.IfThenElse(var_16, var_12)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.if_clause).__module__}.{type(var_20.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_20.then_clause).__module__}.{type(var_20.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_20.else_clause).__module__}.{type(var_20.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_21 = module_1.String()
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
    var_22 = module_0.from_json_schema(var_8)
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
    assert var_22.required == []

def test_case_32():
    var_0 = module_2.Definitions()
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    var_10 = 'minimum'
    var_11 = 'maximum'
    var_12 = 'integer'
    var_13 = 1
    var_14 = 100
    var_15 = {var_1: var_12, var_10: var_13, var_11: var_14}
    var_16 = module_0.from_json_schema_type(var_15, var_12, var_8, var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Integer'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum == 1
    assert var_16.maximum == 100
    assert var_16.exclusive_minimum is None
    assert var_16.exclusive_maximum is None
    assert var_16.multiple_of is None
    assert var_16.precision is None
    assert var_16.coerce_types is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_17 = 'number'
    var_18 = module_0.from_json_schema_type(var_7, var_17, var_8, var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Float'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.minimum is None
    assert var_18.maximum is None
    assert var_18.exclusive_minimum is None
    assert var_18.exclusive_maximum is None
    assert var_18.multiple_of is None
    assert var_18.precision is None
    assert var_18.coerce_types is False
    var_19 = 'boolean'
    var_20 = {var_1: var_19}
    var_21 = module_0.from_json_schema_type(var_20, var_19, var_8, var_0)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.coerce_types is False
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_22 = 'items'
    var_23 = 'minItems'
    var_24 = 'maxItems'
    var_25 = 'array'
    var_26 = {var_1: var_4}
    var_27 = {var_1: var_25, var_22: var_26, var_23: var_13, var_24: var_6}
    var_28 = module_0.from_json_schema_type(var_27, var_25, var_8, var_0)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Array'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.items).__module__}.{type(var_28.items).__qualname__}' == 'typesystem.fields.String'
    assert var_28.additional_items is True
    assert var_28.min_items == 1
    assert var_28.max_items == 10
    assert var_28.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_29 = var_28.items
    var_30 = 'properties'
    var_31 = 'object'
    var_32 = 'name'
    var_33 = {var_1: var_4}
    var_34 = {var_32: var_33}
    var_35 = {var_1: var_31, var_30: var_34}
    var_36 = module_0.from_json_schema_type(var_35, var_31, var_8, var_0)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Object'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert f'{type(var_36.properties).__module__}.{type(var_36.properties).__qualname__}' == 'builtins.dict'
    assert len(var_36.properties) == 1
    assert var_36.pattern_properties == {}
    assert var_36.additional_properties is None
    assert var_36.property_names is None
    assert var_36.min_properties is None
    assert var_36.max_properties is None
    assert var_36.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_37 = var_36.properties[var_32]

def test_case_33():
    var_0 = module_2.Definitions()
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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

def test_case_34():
    var_0 = module_2.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'oneOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'number'
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.one_of[var_12]
    with pytest.raises(TypeError):
        var_14 = var_9.one_of[var_10]

def test_case_35():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = 'boolean'
    var_9 = {var_3: var_8}
    var_10 = {var_0: var_5, var_1: var_7, var_2: var_9}
    var_11 = module_2.Definitions()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_11) == 0
    var_12 = module_0.if_then_else_from_json_schema(var_10, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.if_clause).__module__}.{type(var_12.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_12.then_clause).__module__}.{type(var_12.then_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_12.else_clause).__module__}.{type(var_12.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_13 = var_12.then_clause
    var_14 = var_12.else_clause
    var_15 = {var_3: var_4}
    var_16 = {var_3: var_6}
    var_17 = {var_0: var_15, var_1: var_16}
    var_18 = module_2.Definitions()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_18) == 0
    var_19 = module_0.if_then_else_from_json_schema(var_17, var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.if_clause).__module__}.{type(var_19.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_19.then_clause).__module__}.{type(var_19.then_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_19.else_clause).__module__}.{type(var_19.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_20 = var_19.if_clause
    var_21 = var_19.then_clause
    var_22 = {var_3: var_4}
    var_23 = {var_3: var_8}
    var_24 = {var_0: var_22, var_2: var_23}
    var_25 = module_2.Definitions()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_25) == 0
    var_26 = module_0.if_then_else_from_json_schema(var_24, var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.if_clause).__module__}.{type(var_26.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_26.then_clause).__module__}.{type(var_26.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_26.else_clause).__module__}.{type(var_26.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_27 = var_26.if_clause
    var_28 = var_26.else_clause

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = module_3.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = True
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
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_1.Integer()
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
    var_6 = module_4.purge()
    assert module_4.ASCII == module_4.RegexFlag.ASCII
    assert module_4.A == module_4.RegexFlag.ASCII
    assert module_4.IGNORECASE == module_4.RegexFlag.IGNORECASE
    assert module_4.I == module_4.RegexFlag.IGNORECASE
    assert module_4.LOCALE == module_4.RegexFlag.LOCALE
    assert module_4.L == module_4.RegexFlag.LOCALE
    assert module_4.UNICODE == module_4.RegexFlag.UNICODE
    assert module_4.U == module_4.RegexFlag.UNICODE
    assert module_4.MULTILINE == module_4.RegexFlag.MULTILINE
    assert module_4.M == module_4.RegexFlag.MULTILINE
    assert module_4.DOTALL == module_4.RegexFlag.DOTALL
    assert module_4.S == module_4.RegexFlag.DOTALL
    assert module_4.VERBOSE == module_4.RegexFlag.VERBOSE
    assert module_4.X == module_4.RegexFlag.VERBOSE
    assert module_4.TEMPLATE == module_4.RegexFlag.TEMPLATE
    assert module_4.T == module_4.RegexFlag.TEMPLATE
    assert module_4.DEBUG == module_4.RegexFlag.DEBUG
    var_7 = module_1.Float()
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
    var_9 = module_1.Boolean()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_1.Array()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Array'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.items is None
    assert var_11.additional_items is False
    assert var_11.min_items is None
    assert var_11.max_items is None
    assert var_11.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_12 = module_0.to_json_schema(var_11)
    module_0.to_json_schema(var_6)

def test_case_37():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = True
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_1.Integer()
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
    var_6 = module_0.to_json_schema(var_5)
    var_7 = module_1.Float()
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
    var_8 = module_0.from_json_schema(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format is None
    assert var_8.coerce_types is False
    assert var_8.pattern is None
    assert var_8.pattern_regex is None

def test_case_38():
    var_0 = None
    var_1 = module_1.Array(exact_items=var_0)
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
    var_2 = module_1.Object()
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
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
def test_case_39():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    var_2 = None
    var_3 = -121.72416963658142
    var_4 = module_1.Integer(maximum=var_3, coerce_types=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Integer'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum is None
    assert var_4.maximum == pytest.approx(-121.72416963658142, abs=0.01, rel=0.01)
    assert var_4.exclusive_minimum is None
    assert var_4.exclusive_maximum is None
    assert var_4.multiple_of is None
    assert var_4.precision is None
    assert var_4.coerce_types is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_5 = var_4.get_default_value()
    var_6 = module_1.Float(maximum=var_2, precision=var_2, multiple_of=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Float'
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
    var_7 = module_0.to_json_schema(var_4)
    var_8 = module_1.Boolean()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    module_0.to_json_schema(var_2)

@pytest.mark.xfail(strict=True)
def test_case_40():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = True
    var_3 = 10
    var_4 = '^[a-z]+$'
    var_5 = module_1.String(max_length=var_3, min_length=var_2, pattern=var_4, format=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length == 10
    assert var_5.min_length is True
    assert var_5.format == '^[a-z]+$'
    assert var_5.coerce_types is True
    assert var_5.pattern == '^[a-z]+$'
    assert f'{type(var_5.pattern_regex).__module__}.{type(var_5.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = 'null'
    var_7 = module_0.to_json_schema(var_5)
    var_8 = 0
    var_9 = 100
    var_10 = 5
    var_11 = module_1.Integer(minimum=var_8, maximum=var_9, multiple_of=var_10)
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
    var_13 = module_1.Boolean()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_14 = module_0.to_json_schema(var_13)
    var_15 = module_1.String()
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
    var_16 = True
    var_17 = module_1.Array(var_15, var_16, var_2, var_3, unique_items=var_2)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Array'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.items).__module__}.{type(var_17.items).__qualname__}' == 'typesystem.fields.String'
    assert var_17.additional_items is True
    assert var_17.min_items is True
    assert var_17.max_items == 10
    assert var_17.unique_items is True
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_18 = 'parray'
    var_19 = [var_18, var_6]
    var_20 = module_0.to_json_schema(var_17)
    var_21 = 'name'
    var_22 = module_1.String()
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
    var_23 = {var_21: var_22}
    var_24 = False
    var_25 = module_1.Object(properties=var_23, additional_properties=var_24, required=var_19)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Object'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert f'{type(var_25.properties).__module__}.{type(var_25.properties).__qualname__}' == 'builtins.dict'
    assert len(var_25.properties) == 1
    assert var_25.pattern_properties == {}
    assert var_25.additional_properties is False
    assert var_25.property_names is None
    assert var_25.min_properties is None
    assert var_25.max_properties is None
    assert var_25.required == ['parray', 'null']
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_26 = module_0.to_json_schema(var_25)
    var_27 = module_2.Definitions()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_27) == 0
    var_28 = module_1.String()
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
    var_29 = 'Person'
    var_30 = module_2.Reference(var_29, var_27)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.to == 'Person'
    assert f'{type(var_30.definitions).__module__}.{type(var_30.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_30.definitions) == 0
    assert module_2.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_2.Reference.target).__module__}.{type(module_2.Reference.target).__qualname__}' == 'builtins.property'
    var_31 = var_27.keys()
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_31) == 0
    module_0.to_json_schema(var_30)

def test_case_41():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = True
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
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_1.Integer()
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
    var_6 = module_0.to_json_schema(var_5)
    var_7 = module_1.Float()
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
    var_9 = module_1.Boolean()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_1.String()
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
    var_12 = module_1.Array(var_11)
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
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_13 = module_0.to_json_schema(var_12)
    var_14 = 'name'
    var_15 = module_1.String()
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
    var_16 = {var_14: var_15}
    var_17 = module_1.Object(properties=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Object'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.properties).__module__}.{type(var_17.properties).__qualname__}' == 'builtins.dict'
    assert len(var_17.properties) == 1
    assert var_17.pattern_properties == {}
    assert var_17.additional_properties is True
    assert var_17.property_names is None
    assert var_17.min_properties is None
    assert var_17.max_properties is None
    assert var_17.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_18 = module_0.to_json_schema(var_17)
    var_19 = 'a'
    var_20 = 'A'
    var_21 = (var_19, var_20)
    var_22 = 'b'
    var_23 = 'B'
    var_24 = (var_22, var_23)
    var_25 = [var_21, var_24]
    var_26 = module_1.Choice(choices=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Choice'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.choices == [('a', 'A'), ('b', 'B')]
    assert var_26.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_27 = module_0.to_json_schema(var_26)
    var_28 = 'test'
    var_29 = module_1.Const(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Const'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert var_29.const == 'test'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_30 = module_0.to_json_schema(var_29)
    var_31 = module_1.String()
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
    var_32 = module_1.Integer()
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
    var_34 = module_1.Union(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Union'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.any_of).__module__}.{type(var_34.any_of).__qualname__}' == 'builtins.list'
    assert len(var_34.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_35 = module_0.to_json_schema(var_34)
    var_36 = module_1.String()
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
    var_37 = module_1.Integer()
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Integer'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.minimum is None
    assert var_37.maximum is None
    assert var_37.exclusive_minimum is None
    assert var_37.exclusive_maximum is None
    assert var_37.multiple_of is None
    assert var_37.precision is None
    assert var_37.coerce_types is True
    var_38 = [var_36, var_37]
    var_39 = module_3.OneOf(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.one_of).__module__}.{type(var_39.one_of).__qualname__}' == 'builtins.list'
    assert len(var_39.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_40 = module_0.to_json_schema(var_39)
    var_41 = module_1.String()
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
    var_42 = module_1.Integer()
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
    var_44 = module_3.AllOf(var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert f'{type(var_44.all_of).__module__}.{type(var_44.all_of).__qualname__}' == 'builtins.list'
    assert len(var_44.all_of) == 2
    var_45 = module_0.to_json_schema(var_44)
    var_46 = module_1.String()
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.String'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert var_46.allow_blank is False
    assert var_46.trim_whitespace is True
    assert var_46.max_length is None
    assert var_46.min_length is None
    assert var_46.format is None
    assert var_46.coerce_types is True
    assert var_46.pattern is None
    assert var_46.pattern_regex is None
    var_47 = module_1.Integer()
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.Integer'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert var_47.minimum is None
    assert var_47.maximum is None
    assert var_47.exclusive_minimum is None
    assert var_47.exclusive_maximum is None
    assert var_47.multiple_of is None
    assert var_47.precision is None
    assert var_47.coerce_types is True
    var_48 = module_3.IfThenElse(var_46, var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert f'{type(var_48.if_clause).__module__}.{type(var_48.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_48.then_clause).__module__}.{type(var_48.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_48.else_clause).__module__}.{type(var_48.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_49 = module_0.to_json_schema(var_48)
    var_50 = module_1.String()
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
    var_51 = module_3.Not(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.composites.Not'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.negated).__module__}.{type(var_51.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_52 = module_0.to_json_schema(var_51)
    var_53 = 'Person'
    var_54 = module_1.String()
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
    var_55 = {var_14: var_54}
    var_56 = module_1.Object(properties=var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Object'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.properties).__module__}.{type(var_56.properties).__qualname__}' == 'builtins.dict'
    assert len(var_56.properties) == 1
    assert var_56.pattern_properties == {}
    assert var_56.additional_properties is True
    assert var_56.property_names is None
    assert var_56.min_properties is None
    assert var_56.max_properties is None
    assert var_56.required == []
    var_57 = {var_53: var_56}
    var_58 = module_2.Reference(var_53, var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.to == 'Person'
    assert f'{type(var_58.definitions).__module__}.{type(var_58.definitions).__qualname__}' == 'builtins.dict'
    assert len(var_58.definitions) == 1
    assert module_2.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_2.Reference.target).__module__}.{type(module_2.Reference.target).__qualname__}' == 'builtins.property'
    var_59 = module_0.to_json_schema(var_58)
    var_60 = module_1.String()
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
    var_61 = {var_14: var_60}
    var_62 = [var_14]
    var_63 = module_2.Schema(var_61)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert f'{type(var_63.fields).__module__}.{type(var_63.fields).__qualname__}' == 'builtins.dict'
    assert len(var_63.fields) == 1
    assert var_63.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_64 = module_0.to_json_schema(var_63)
    var_65 = module_3.NeverMatch()
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_66 = module_0.to_json_schema(var_65)
    assert var_66 is False
    var_67 = module_1.Any()
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.fields.Any'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    var_68 = module_0.to_json_schema(var_67)
    assert var_68 is True

def test_case_42():
    var_0 = module_2.Definitions()
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    var_5 = '$ref'
    var_6 = 'http://example.com/schema#/User'
    var_7 = {var_5: var_6}
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_7, var_0)

def test_case_43():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_3.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = module_1.String()
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_0.to_json_schema(var_4)
    var_6 = module_1.Integer()
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
    var_8 = module_1.Float()
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
    var_10 = module_1.Boolean()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_11 = module_0.to_json_schema(var_10)
    var_12 = module_1.String()
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
    var_13 = module_1.Array(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Array'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.items).__module__}.{type(var_13.items).__qualname__}' == 'typesystem.fields.String'
    assert var_13.additional_items is False
    assert var_13.min_items is None
    assert var_13.max_items is None
    assert var_13.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_14 = module_0.to_json_schema(var_13)
    var_15 = 'name'
    var_16 = module_1.String()
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
    var_17 = {var_15: var_16}
    var_18 = module_1.Object(properties=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Object'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert f'{type(var_18.properties).__module__}.{type(var_18.properties).__qualname__}' == 'builtins.dict'
    assert len(var_18.properties) == 1
    assert var_18.pattern_properties == {}
    assert var_18.additional_properties is True
    assert var_18.property_names is None
    assert var_18.min_properties is None
    assert var_18.max_properties is None
    assert var_18.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_19 = module_0.to_json_schema(var_18)
    var_20 = 'a'
    var_21 = (var_20, var_20)
    var_22 = [var_21]
    var_23 = module_1.Choice(choices=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Choice'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.choices == [('a', 'a')]
    assert var_23.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_1.Const(var_20)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Const'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.const == 'a'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_26 = module_0.to_json_schema(var_25)
    var_27 = module_1.String()
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
    var_28 = module_1.Integer()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Integer'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.minimum is None
    assert var_28.maximum is None
    assert var_28.exclusive_minimum is None
    assert var_28.exclusive_maximum is None
    assert var_28.multiple_of is None
    assert var_28.precision is None
    assert var_28.coerce_types is True
    var_29 = [var_27, var_28]
    var_30 = module_1.Union(var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Union'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.any_of).__module__}.{type(var_30.any_of).__qualname__}' == 'builtins.list'
    assert len(var_30.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_31 = module_0.to_json_schema(var_30)
    var_32 = module_1.String()
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
    var_33 = module_1.Integer()
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
    var_35 = module_3.OneOf(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.one_of).__module__}.{type(var_35.one_of).__qualname__}' == 'builtins.list'
    assert len(var_35.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_36 = module_0.to_json_schema(var_35)
    var_37 = module_1.String()
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
    var_38 = module_1.Integer()
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Integer'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.minimum is None
    assert var_38.maximum is None
    assert var_38.exclusive_minimum is None
    assert var_38.exclusive_maximum is None
    assert var_38.multiple_of is None
    assert var_38.precision is None
    assert var_38.coerce_types is True
    var_39 = [var_37, var_38]
    var_40 = module_3.AllOf(var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert f'{type(var_40.all_of).__module__}.{type(var_40.all_of).__qualname__}' == 'builtins.list'
    assert len(var_40.all_of) == 2
    var_41 = module_1.Integer()
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Integer'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.minimum is None
    assert var_41.maximum is None
    assert var_41.exclusive_minimum is None
    assert var_41.exclusive_maximum is None
    assert var_41.multiple_of is None
    assert var_41.precision is None
    assert var_41.coerce_types is True
    var_42 = module_3.IfThenElse(var_4, var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert f'{type(var_42.if_clause).__module__}.{type(var_42.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_42.then_clause).__module__}.{type(var_42.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_42.else_clause).__module__}.{type(var_42.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_43 = module_0.to_json_schema(var_42)
    var_44 = module_1.String()
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
    var_45 = module_3.Not(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.composites.Not'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert f'{type(var_45.negated).__module__}.{type(var_45.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_46 = module_0.to_json_schema(var_45)

def test_case_44():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_3.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = module_1.String()
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_1.Integer()
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
    var_6 = module_0.to_json_schema(var_5)
    var_7 = module_1.Float()
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
    var_9 = module_1.Boolean()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_1.String()
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
    var_12 = module_1.Array(var_11)
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
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_13 = module_0.to_json_schema(var_12)
    var_14 = 'name'
    var_15 = module_1.String()
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
    var_16 = {var_14: var_15}
    var_17 = module_1.Object(properties=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Object'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.properties).__module__}.{type(var_17.properties).__qualname__}' == 'builtins.dict'
    assert len(var_17.properties) == 1
    assert var_17.pattern_properties == {}
    assert var_17.additional_properties is True
    assert var_17.property_names is None
    assert var_17.min_properties is None
    assert var_17.max_properties is None
    assert var_17.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_18 = module_0.to_json_schema(var_17)
    var_19 = module_1.Const(var_14)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Const'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.const == 'name'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_20 = module_0.to_json_schema(var_19)
    var_21 = module_1.String()
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
    var_22 = module_1.Integer()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Integer'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.minimum is None
    assert var_22.maximum is None
    assert var_22.exclusive_minimum is None
    assert var_22.exclusive_maximum is None
    assert var_22.multiple_of is None
    assert var_22.precision is None
    assert var_22.coerce_types is True
    var_23 = [var_21, var_22]
    var_24 = module_1.Union(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Union'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.any_of).__module__}.{type(var_24.any_of).__qualname__}' == 'builtins.list'
    assert len(var_24.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_25 = module_0.to_json_schema(var_24)
    var_26 = module_1.String()
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
    var_27 = module_1.Integer()
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
    var_29 = module_3.OneOf(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.one_of).__module__}.{type(var_29.one_of).__qualname__}' == 'builtins.list'
    assert len(var_29.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_30 = module_0.to_json_schema(var_29)
    var_31 = module_1.String()
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
    var_32 = module_1.Integer()
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
    var_34 = module_3.AllOf(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.all_of).__module__}.{type(var_34.all_of).__qualname__}' == 'builtins.list'
    assert len(var_34.all_of) == 2
    var_35 = module_0.to_json_schema(var_34)
    var_36 = module_1.String()
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
    var_37 = module_1.Integer()
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Integer'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.minimum is None
    assert var_37.maximum is None
    assert var_37.exclusive_minimum is None
    assert var_37.exclusive_maximum is None
    assert var_37.multiple_of is None
    assert var_37.precision is None
    assert var_37.coerce_types is True
    var_38 = module_3.IfThenElse(var_36, var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert f'{type(var_38.if_clause).__module__}.{type(var_38.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_38.then_clause).__module__}.{type(var_38.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_38.else_clause).__module__}.{type(var_38.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_39 = module_0.to_json_schema(var_38)
    var_40 = module_1.String()
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.String'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert var_40.allow_blank is False
    assert var_40.trim_whitespace is True
    assert var_40.max_length is None
    assert var_40.min_length is None
    assert var_40.format is None
    assert var_40.coerce_types is True
    assert var_40.pattern is None
    assert var_40.pattern_regex is None
    var_41 = module_3.Not(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.composites.Not'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.negated).__module__}.{type(var_41.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_42 = module_0.to_json_schema(var_41)

def test_case_45():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
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
    var_2 = None
    var_3 = module_3.IfThenElse(var_1, else_clause=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.if_clause).__module__}.{type(var_3.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_4 = None
    var_5 = 'H R'
    var_6 = {var_5: var_3}
    var_7 = module_2.Schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 1
    assert var_7.required == ['H R']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = module_0.to_json_schema(var_7)
    var_9 = module_0.to_json_schema(var_3, var_0)
    var_10 = module_0.from_json_schema(var_6, var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Any'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False

@pytest.mark.xfail(strict=True)
def test_case_46():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_3.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
    var_5 = 10
    var_6 = '^[a-z]*$'
    var_7 = 'email'
    var_8 = module_1.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
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
    assert var_8.pattern == '^[a-z]*$'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = 100
    var_11 = 2
    var_12 = module_1.Integer(minimum=var_4, maximum=var_10, multiple_of=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Integer'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum is True
    assert var_12.maximum == 100
    assert var_12.exclusive_minimum is None
    assert var_12.exclusive_maximum is None
    assert var_12.multiple_of == 2
    assert var_12.precision is None
    assert var_12.coerce_types is True
    var_13 = module_0.to_json_schema(var_12)
    var_14 = module_1.Boolean()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_15 = module_0.to_json_schema(var_14)
    var_16 = module_1.String()
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
    var_17 = False
    var_18 = module_1.Array(var_16, var_17, var_4, var_5, unique_items=var_4)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Array'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert f'{type(var_18.items).__module__}.{type(var_18.items).__qualname__}' == 'typesystem.fields.String'
    assert var_18.additional_items is False
    assert var_18.min_items is True
    assert var_18.max_items == 10
    assert var_18.unique_items is True
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_19 = module_0.to_json_schema(var_18)
    var_20 = 'name'
    var_21 = module_1.String()
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
    var_23 = [var_20]
    var_24 = module_1.Object(properties=var_22, min_properties=var_4, max_properties=var_11, required=var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Object'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.properties).__module__}.{type(var_24.properties).__qualname__}' == 'builtins.dict'
    assert len(var_24.properties) == 1
    assert var_24.pattern_properties == {}
    assert var_24.additional_properties is True
    assert var_24.property_names is None
    assert var_24.min_properties is True
    assert var_24.max_properties == 2
    assert var_24.required == ['name']
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_25 = module_0.to_json_schema(var_24)
    var_26 = 'a'
    var_27 = 'A'
    var_28 = (var_26, var_27)
    var_29 = 'b'
    var_30 = 'B'
    var_31 = (var_29, var_30)
    var_32 = [var_28, var_31]
    var_33 = module_1.Choice(choices=var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Choice'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.choices == [('a', 'A'), ('b', 'B')]
    assert var_33.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_34 = module_0.to_json_schema(var_33)
    var_35 = 'test'
    var_36 = module_1.Const(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Const'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.const == 'test'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_37 = module_0.to_json_schema(var_36)
    var_38 = module_1.String()
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
    var_39 = module_1.Integer()
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
    var_40 = [var_38, var_39]
    var_41 = module_1.Union(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Union'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.any_of).__module__}.{type(var_41.any_of).__qualname__}' == 'builtins.list'
    assert len(var_41.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_42 = module_0.to_json_schema(var_41)
    var_43 = module_1.String(min_length=var_4)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.String'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.allow_blank is False
    assert var_43.trim_whitespace is True
    assert var_43.max_length is None
    assert var_43.min_length is True
    assert var_43.format is None
    assert var_43.coerce_types is True
    assert var_43.pattern is None
    assert var_43.pattern_regex is None
    var_44 = module_1.String(max_length=var_5)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.String'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert var_44.allow_blank is False
    assert var_44.trim_whitespace is True
    assert var_44.max_length == 10
    assert var_44.min_length is None
    assert var_44.format is None
    assert var_44.coerce_types is True
    assert var_44.pattern is None
    assert var_44.pattern_regex is None
    var_45 = [var_43, var_44]
    var_46 = module_3.AllOf(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.all_of).__module__}.{type(var_46.all_of).__qualname__}' == 'builtins.list'
    assert len(var_46.all_of) == 2
    var_47 = module_0.to_json_schema(var_46)
    var_48 = module_1.String()
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
    var_49 = module_1.Integer()
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
    var_50 = module_1.Boolean()
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert var_50.coerce_types is True
    var_51 = module_3.IfThenElse(var_48, var_49, var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.if_clause).__module__}.{type(var_51.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_51.then_clause).__module__}.{type(var_51.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_51.else_clause).__module__}.{type(var_51.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_52 = module_0.to_json_schema(var_51)
    var_53 = module_1.String()
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.String'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert var_53.allow_blank is False
    assert var_53.trim_whitespace is True
    assert var_53.max_length is None
    assert var_53.min_length is None
    assert var_53.format is None
    assert var_53.coerce_types is True
    assert var_53.pattern is None
    assert var_53.pattern_regex is None
    var_54 = module_3.Not(var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.composites.Not'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert f'{type(var_54.negated).__module__}.{type(var_54.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_55 = module_0.to_json_schema(var_54)
    var_56 = module_1.String()
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
    var_33.validate(var_12)

def test_case_47():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'dependencies', 'items', 'uniqueItems', 'minProperties', 'propertyNames', 'additionalProperties', 'maximum', 'type', 'exclusiveMaximum', 'pattern', 'maxItems', 'minimum', 'additionalItems', 'boolean_schema', 'maxLength', 'properties', 'minLength', 'required', 'exclusiveMinimum', 'minItems', 'maxProperties', 'patternProperties', 'multipleOf'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
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
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_1.Float()
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
    var_7 = module_1.Boolean()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_8 = module_0.to_json_schema(var_7)
    var_9 = module_1.String()
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
    var_10 = module_1.Array(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Array'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.items).__module__}.{type(var_10.items).__qualname__}' == 'typesystem.fields.String'
    assert var_10.additional_items is False
    assert var_10.min_items is None
    assert var_10.max_items is None
    assert var_10.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_11 = module_0.to_json_schema(var_10)
    var_12 = 'name'
    var_13 = module_1.String()
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
    assert var_13.pattern is None
    assert var_13.pattern_regex is None
    var_14 = module_1.Object(properties=var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Object'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.properties == {}
    assert var_14.pattern_properties == {}
    assert f'{type(var_14.additional_properties).__module__}.{type(var_14.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_14.property_names is None
    assert var_14.min_properties is None
    assert var_14.max_properties is None
    assert var_14.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_15 = module_0.to_json_schema(var_14)
    var_16 = module_1.Const(var_12)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Const'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.const == 'name'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_17 = module_0.to_json_schema(var_16)
    var_18 = module_1.String()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.String'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.allow_blank is False
    assert var_18.trim_whitespace is True
    assert var_18.max_length is None
    assert var_18.min_length is None
    assert var_18.format is None
    assert var_18.coerce_types is True
    assert var_18.pattern is None
    assert var_18.pattern_regex is None
    var_19 = module_1.Integer()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Integer'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.minimum is None
    assert var_19.maximum is None
    assert var_19.exclusive_minimum is None
    assert var_19.exclusive_maximum is None
    assert var_19.multiple_of is None
    assert var_19.precision is None
    assert var_19.coerce_types is True
    var_20 = module_1.String()
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
    var_21 = module_1.Integer()
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
    var_22 = module_1.String()
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
    var_23 = module_1.Integer()
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
    var_24 = [var_22, var_23]
    var_25 = module_3.AllOf(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert f'{type(var_25.all_of).__module__}.{type(var_25.all_of).__qualname__}' == 'builtins.list'
    assert len(var_25.all_of) == 2
    var_26 = module_1.Integer()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Integer'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.minimum is None
    assert var_26.maximum is None
    assert var_26.exclusive_minimum is None
    assert var_26.exclusive_maximum is None
    assert var_26.multiple_of is None
    assert var_26.precision is None
    assert var_26.coerce_types is True
    var_27 = module_3.IfThenElse(var_2, var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.if_clause).__module__}.{type(var_27.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_27.then_clause).__module__}.{type(var_27.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_27.else_clause).__module__}.{type(var_27.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_28 = module_0.to_json_schema(var_27)
    var_29 = module_3.Not(var_13)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.composites.Not'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.negated).__module__}.{type(var_29.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_30 = module_0.from_json_schema(var_15)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Object'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.properties == {}
    assert var_30.pattern_properties == {}
    assert f'{type(var_30.additional_properties).__module__}.{type(var_30.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_30.property_names is None
    assert var_30.min_properties is None
    assert var_30.max_properties is None
    assert var_30.required == []