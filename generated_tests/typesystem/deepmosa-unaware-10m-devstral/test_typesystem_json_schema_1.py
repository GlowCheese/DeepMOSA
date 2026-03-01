# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import typesystem.composites as module_1
import typesystem.fields as module_2
import enum as module_3
import re as module_4
import typesystem.schemas as module_5

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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

def test_case_1():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_1 = {var_0: var_0}
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_0 = module_3._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_4():
    var_0 = {}
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_4.purge()
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

def test_case_10():
    var_0 = 'allOf'
    var_1 = 'string'
    var_2 = 5
    var_3 = {var_0: var_1, var_1: var_2}
    var_4 = module_0.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.all_of).__module__}.{type(var_4.all_of).__qualname__}' == 'builtins.list'
    assert len(var_4.all_of) == 6
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
def test_case_11():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

def test_case_12():
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_13():
    var_0 = module_2.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_1 = module_0.get_standard_properties(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
def test_case_14():
    var_0 = None
    module_0.any_of_from_json_schema(var_0, var_0)

def test_case_15():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.to_json_schema(var_1, var_0)
    assert var_2 is True

def test_case_16():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_3._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0

def test_case_17():
    var_0 = module_2.Object()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
def test_case_19():
    var_0 = None
    var_1 = {var_0: var_0}
    var_2 = module_2.Const(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Const'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.const == {None: None}
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_3 = module_1.NeverMatch()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_4 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    module_0.get_standard_properties(var_0)

def test_case_20():
    var_0 = None
    var_1 = {var_0: var_0}
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7

def test_case_21():
    var_0 = None
    var_1 = module_2.Integer(exclusive_minimum=var_0)
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_0 = None
    var_1 = {var_0: var_0}
    var_2 = module_1.OneOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.one_of == {None: None}
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    module_0.to_json_schema(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_2.Object()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.fields.Object'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_3 = var_0.has_default()
    assert var_3 is False
    var_4 = module_0.to_json_schema(var_2, var_3)
    var_5 = module_0.type_from_json_schema(var_1, var_3)
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
    var_1.pop(var_3)

def test_case_24():
    var_0 = None
    var_1 = {var_0: var_0}
    var_2 = module_0.from_json_schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_4 = module_2.Field()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_4, var_0)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = {var_0: var_0}
    var_2 = module_1.AllOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.all_of == {None: None}
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    module_0.to_json_schema(var_2)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = {var_0: var_0}
    var_2 = module_5.Reference(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == {None: None}
    assert var_2.definitions is None
    assert module_5.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_5.Reference.target).__module__}.{type(module_5.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = module_0.type_from_json_schema(var_1, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is True
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    module_0.to_json_schema(var_2)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_3._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_1.OneOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.one_of).__module__}.{type(var_2.one_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.one_of) == 0
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_4 = module_0.to_json_schema(var_2, var_1)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_5 = module_0.get_standard_properties(var_2)
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = {var_0: var_0}
    var_2 = module_0.from_json_schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = module_0.type_from_json_schema(var_1, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is True
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_0.to_json_schema(var_2)
    assert var_5 is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_6 = module_0.get_standard_properties(var_2)
    var_7 = module_0.to_json_schema(var_4, var_0)
    var_8 = module_0.to_json_schema(var_2)
    assert var_8 is True
    var_9 = '&'
    var_10 = {var_9: var_0, var_9: var_0, var_9: var_0, var_9: var_0}
    var_11 = module_5.Definitions(**var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_11) == 1
    module_0.to_json_schema(var_11, var_0)

def test_case_29():
    var_0 = True
    var_1 = 'L%HiL?WJq2e'
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is True
    assert var_2.min_length is True
    assert var_2.format == 'L%HiL?WJq2e'
    assert var_2.coerce_types is True
    assert var_2.pattern == 'L%HiL?WJq2e'
    assert f'{type(var_2.pattern_regex).__module__}.{type(var_2.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_0 = False
    var_1 = 'L%HiL?WJq2e'
    var_2 = module_2.String(allow_blank=var_0, trim_whitespace=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace == 'L%HiL?WJq2e'
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
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_31():
    var_0 = True
    var_1 = 'L%HiL?WJq2e'
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is True
    assert var_2.min_length is True
    assert var_2.format == 'L%HiL?WJq2e'
    assert var_2.coerce_types is True
    assert var_2.pattern == 'L%HiL?WJq2e'
    assert f'{type(var_2.pattern_regex).__module__}.{type(var_2.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = None
    var_5 = module_0.from_json_schema(var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length is True
    assert var_5.min_length is None
    assert var_5.format == 'L%HiL?WJq2e'
    assert var_5.coerce_types is False
    assert var_5.pattern == 'L%HiL?WJq2e'
    assert f'{type(var_5.pattern_regex).__module__}.{type(var_5.pattern_regex).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = {var_0: var_0}
    var_2 = module_0.from_json_schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_4 = module_0.to_json_schema(var_3, var_0)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.any_of).__module__}.{type(var_5.any_of).__qualname__}' == 'builtins.list'
    assert len(var_5.any_of) == 5
    module_0.type_from_json_schema(var_3, var_0)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'allOf'
    var_2 = None
    var_3 = module_0.to_json_schema(var_0, var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = 'minLength'
    var_5 = 'string'
    var_6 = 5
    var_7 = {var_1: var_5, var_4: var_6}
    var_8 = None
    var_9 = True
    var_10 = module_2.Field(default=var_8, allow_null=var_4, read_only=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Field'
    assert var_10.default is None
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null == 'minLength'
    assert var_10.read_only is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_11 = module_0.get_standard_properties(var_10)
    var_12 = {var_1: var_5, var_5: var_6}
    var_13 = [var_7, var_12]
    var_14 = {var_1: var_13}
    var_15 = module_0.all_of_from_json_schema(var_14, var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.all_of).__module__}.{type(var_15.all_of).__qualname__}' == 'builtins.list'
    assert len(var_15.all_of) == 2
    var_16 = 0
    var_17 = var_15.all_of[var_16]
    var_18 = 'properties'
    var_19 = module_0.to_json_schema(var_17, var_14)
    var_20 = 'object'
    var_21 = module_0.from_json_schema(var_7)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.all_of).__module__}.{type(var_21.all_of).__qualname__}' == 'builtins.list'
    assert len(var_21.all_of) == 2
    var_22 = var_17.serialize(var_17)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.all_of).__module__}.{type(var_22.all_of).__qualname__}' == 'builtins.list'
    assert len(var_22.all_of) == 2
    var_23 = {var_1: var_20, var_18: var_7}
    var_24 = [var_12, var_23]
    var_25 = {var_1: var_24}
    module_0.all_of_from_json_schema(var_25, var_0)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 'oneOf'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = [var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_5.Definitions()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 0
    var_7 = module_0.one_of_from_json_schema(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.one_of).__module__}.{type(var_7.one_of).__qualname__}' == 'builtins.list'
    assert len(var_7.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_8 = var_7.one_of
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 0
    var_11 = var_7.one_of[var_10]
    var_12 = 1
    var_13 = var_7.one_of[var_12]
    var_14 = module_5.Definitions()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_14) == 0
    var_15 = module_0.one_of_from_json_schema(var_3, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.one_of).__module__}.{type(var_15.one_of).__qualname__}' == 'builtins.list'
    assert len(var_15.one_of) == 6
    var_16 = 'properties'
    var_17 = 'object'
    var_18 = 'name'
    var_19 = {var_18: var_3}
    var_20 = {var_1: var_17, var_16: var_19}
    var_21 = 'items'
    var_22 = 'array'
    var_23 = {var_1: var_22, var_21: var_20}
    var_24 = [var_20, var_23]
    var_25 = {var_0: var_24}
    var_26 = module_5.Definitions()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_26) == 0
    var_27 = module_0.one_of_from_json_schema(var_25, var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.one_of).__module__}.{type(var_27.one_of).__qualname__}' == 'builtins.list'
    assert len(var_27.one_of) == 2
    var_28 = var_27.one_of
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = var_27.one_of[var_12]
    var_31 = module_5.Definitions()
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_31) == 0
    var_32 = '$ref'
    var_33 = '#/components/schemas/StringSchema'
    var_34 = {var_32: var_33}
    var_35 = {var_16: var_13}
    var_36 = [var_34, var_35]
    var_37 = {var_0: var_36}
    module_0.one_of_from_json_schema(var_37, var_31)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = 'oneOf'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_5.Definitions()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = module_0.one_of_from_json_schema(var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.one_of).__module__}.{type(var_5.one_of).__qualname__}' == 'builtins.list'
    assert len(var_5.one_of) == 6
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_6 = var_5.one_of
    var_7 = len(var_6)
    var_8 = 0
    var_9 = var_5.one_of[var_8]
    var_10 = 1
    var_11 = var_5.one_of[var_10]
    var_12 = {var_11: var_1}
    var_13 = {var_9: var_0}
    var_14 = [var_12, var_13]
    var_15 = 'default_value'
    var_16 = {var_0: var_14, var_11: var_15}
    var_17 = module_5.Definitions()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_17) == 0
    var_18 = module_0.one_of_from_json_schema(var_16, var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert f'{type(var_18.one_of).__module__}.{type(var_18.one_of).__qualname__}' == 'builtins.list'
    assert len(var_18.one_of) == 2
    var_19 = 'properties'
    var_20 = 'items'
    var_21 = 'array'
    var_22 = {var_1: var_21, var_20: var_3}
    var_23 = [var_13, var_22]
    var_24 = {var_0: var_23}
    var_25 = module_5.Definitions()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_25) == 0
    var_26 = module_0.one_of_from_json_schema(var_24, var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.one_of).__module__}.{type(var_26.one_of).__qualname__}' == 'builtins.list'
    assert len(var_26.one_of) == 2
    var_27 = var_26.one_of
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = var_26.one_of[var_10]
    var_30 = module_5.Definitions()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_30) == 0
    var_31 = '$ref'
    var_32 = '#/components/schemas/StringSchema'
    var_33 = {var_31: var_32}
    var_34 = {var_19: var_11}
    var_35 = [var_33, var_34]
    var_36 = {var_0: var_35}
    module_0.one_of_from_json_schema(var_36, var_30)

def test_case_36():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_5.Definitions()
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = {var_0: var_12, var_8: var_10}
    var_14 = module_5.Definitions()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_14) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_15 = module_0.enum_from_json_schema(var_13, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Choice'
    assert var_15.default == 2
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.choices == [(1, 1), (2, 2), (3, 3)]
    assert var_15.coerce_types is True
    var_16 = module_5.Definitions()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_16) == 0

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = 'anyOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = 'number'
    var_4 = -8
    var_5 = 1
    var_6 = 'default'
    var_7 = 'minLength'
    var_8 = {var_1: var_2, var_7: var_5}
    var_9 = 'minimum'
    var_10 = {var_1: var_3, var_9: var_4}
    var_11 = [var_8, var_10]
    var_12 = 'test'
    var_13 = {var_0: var_11, var_6: var_12}
    var_14 = module_5.Definitions()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_14) == 0
    var_15 = module_0.any_of_from_json_schema(var_13, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Union'
    assert var_15.default == 'test'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.any_of).__module__}.{type(var_15.any_of).__qualname__}' == 'builtins.list'
    assert len(var_15.any_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_16 = var_15.any_of
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = module_5.Definitions()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_18) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_19 = '$ref'
    var_20 = 'aXc*ukC?'
    var_21 = {var_19: var_20}
    var_22 = {var_1: var_3}
    var_23 = [var_21, var_22]
    var_24 = {var_0: var_23}
    module_0.any_of_from_json_schema(var_24, var_18)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'allOf'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = None
    var_7 = True
    var_8 = module_2.Field(default=var_6, allow_null=var_2, read_only=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Field'
    assert var_8.default is None
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null == 'minLength'
    assert var_8.read_only is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_9 = var_0.clear()
    var_10 = {var_1: var_3, var_3: var_4}
    var_11 = [var_5, var_10]
    var_12 = {var_1: var_11}
    var_13 = module_0.all_of_from_json_schema(var_12, var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.all_of).__module__}.{type(var_13.all_of).__qualname__}' == 'builtins.list'
    assert len(var_13.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_14 = var_13.all_of
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 0
    var_17 = var_13.all_of[var_16]
    var_18 = module_0.to_json_schema(var_17, var_12)
    var_19 = module_0.from_json_schema(var_5)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.all_of).__module__}.{type(var_19.all_of).__qualname__}' == 'builtins.list'
    assert len(var_19.all_of) == 2
    var_20 = var_17.serialize(var_14)
    var_21 = 'integer'
    module_0.from_json_schema_type(var_4, var_21, var_20, var_17)

def test_case_39():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'allOf'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.all_of).__module__}.{type(var_6.all_of).__qualname__}' == 'builtins.list'
    assert len(var_6.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
def test_case_40():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'allOf'
    var_2 = None
    var_3 = module_0.to_json_schema(var_0, var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = 'minLength'
    var_5 = 'string'
    var_6 = 5
    var_7 = {var_1: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 'properties'
    var_10 = 'object'
    var_11 = module_0.from_json_schema(var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.all_of).__module__}.{type(var_11.all_of).__qualname__}' == 'builtins.list'
    assert len(var_11.all_of) == 2
    var_12 = {var_10: var_10, var_9: var_8}
    var_13 = 'age'
    var_14 = {var_13: var_12}
    var_15 = {var_1: var_10, var_9: var_14}
    var_16 = [var_12, var_15]
    var_17 = {var_1: var_16}
    module_0.all_of_from_json_schema(var_17, var_0)

def test_case_41():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'allOf'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = 5
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_1: var_3, var_3: var_5}
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_8}
    var_10 = module_0.all_of_from_json_schema(var_9, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.all_of).__module__}.{type(var_10.all_of).__qualname__}' == 'builtins.list'
    assert len(var_10.all_of) == 2
    var_11 = var_10.all_of
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 0
    var_14 = var_10.all_of[var_13]
    var_15 = module_0.to_json_schema(var_14, var_9)
    var_16 = module_0.from_json_schema(var_6)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert f'{type(var_16.all_of).__module__}.{type(var_16.all_of).__qualname__}' == 'builtins.list'
    assert len(var_16.all_of) == 2
    var_17 = [var_9, var_15]
    var_18 = {var_1: var_17}
    var_19 = module_0.all_of_from_json_schema(var_18, var_0)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.all_of).__module__}.{type(var_19.all_of).__qualname__}' == 'builtins.list'
    assert len(var_19.all_of) == 2
    with pytest.raises(AttributeError):
        var_20 = var_1.all_of

def test_case_42():
    var_0 = 'oneOf'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = [var_2, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_5.Definitions()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 0
    var_6 = module_0.one_of_from_json_schema(var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.one_of).__module__}.{type(var_6.one_of).__qualname__}' == 'builtins.list'
    assert len(var_6.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_7 = len(var_1)
    var_8 = 0
    var_9 = var_6.one_of[var_8]
    var_10 = 1
    var_11 = var_6.one_of[var_10]
    var_12 = {var_11: var_1}
    var_13 = {var_9: var_0}
    var_14 = [var_12, var_13]
    var_15 = {var_0: var_14, var_11: var_7}
    var_16 = module_5.Definitions()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_16) == 0
    var_17 = module_0.one_of_from_json_schema(var_15, var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.one_of).__module__}.{type(var_17.one_of).__qualname__}' == 'builtins.list'
    assert len(var_17.one_of) == 2
    var_18 = 'items'
    var_19 = '=yJ\r\\;/"0'
    var_20 = {var_1: var_19, var_18: var_2}
    var_21 = [var_13, var_20]
    var_22 = {var_0: var_21}
    var_23 = module_5.Definitions()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_23) == 0
    var_24 = module_0.one_of_from_json_schema(var_22, var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.one_of).__module__}.{type(var_24.one_of).__qualname__}' == 'builtins.list'
    assert len(var_24.one_of) == 2
    with pytest.raises(AttributeError):
        var_25 = var_8.one_of

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'default'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'number'
    var_8 = {var_4: var_7}
    var_9 = 42
    var_10 = {var_0: var_6, var_1: var_8, var_2: var_8, var_3: var_9}
    var_11 = module_5.Definitions()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_11) == 0
    var_12 = module_0.if_then_else_from_json_schema(var_10, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_12.default == 42
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.if_clause).__module__}.{type(var_12.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_12.then_clause).__module__}.{type(var_12.then_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_12.else_clause).__module__}.{type(var_12.else_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_18 = 3.14
    var_19 = {var_0: var_16, var_1: var_17, var_3: var_18}
    var_20 = module_0.if_then_else_from_json_schema(var_19, var_11)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_20.default == pytest.approx(3.14, abs=0.01, rel=0.01)
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.if_clause).__module__}.{type(var_20.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_20.then_clause).__module__}.{type(var_20.then_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_20.else_clause).__module__}.{type(var_20.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_21 = var_20.if_clause
    var_22 = var_20.then_clause
    var_23 = {var_4: var_5}
    var_24 = True
    var_25 = {var_0: var_23, var_2: var_10, var_3: var_24}
    var_26 = module_0.if_then_else_from_json_schema(var_25, var_11)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_26.default is True
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.if_clause).__module__}.{type(var_26.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_26.then_clause).__module__}.{type(var_26.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_26.else_clause).__module__}.{type(var_26.else_clause).__qualname__}' == 'typesystem.composites.IfThenElse'
    var_27 = var_26.if_clause
    var_28 = var_26.else_clause
    module_0.if_then_else_from_json_schema(var_8, var_11)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = 10
    var_1 = 'phi*OP$a/ArN+'
    var_2 = 'number'
    var_3 = 'items'
    var_4 = module_2.Integer()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_5 = module_4.purge()
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
    var_6 = {var_3: var_2, var_1: var_0}
    var_7 = module_0.from_json_schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Union'
    assert var_7.default is None
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is True
    assert var_7.read_only is False
    assert f'{type(var_7.any_of).__module__}.{type(var_7.any_of).__qualname__}' == 'builtins.list'
    assert len(var_7.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_5.validate(var_5)

def test_case_45():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 'number'
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
    var_12 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11}
    var_13 = False
    var_14 = module_5.Definitions()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_14) == 0
    var_15 = module_0.from_json_schema_type(var_12, var_7, var_13, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Float'
    assert var_15.default == 50
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum == 0
    assert var_15.maximum == 100
    assert var_15.exclusive_minimum == 0
    assert var_15.exclusive_maximum == 100
    assert var_15.multiple_of == 2
    assert var_15.precision is None
    assert var_15.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_16 = 'integer'
    var_17 = module_5.Definitions()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_17) == 0
    var_18 = var_14.__iter__()
    var_19 = 'minLength'
    var_20 = 'maxLength'
    var_21 = 'format'
    var_22 = 'pattern'
    var_23 = 'string'
    var_24 = 5
    var_25 = 'email'
    var_26 = '^[a-zA-Z0-9]+$'
    var_27 = 'test'
    var_28 = {var_0: var_23, var_19: var_24, var_20: var_9, var_21: var_25, var_22: var_26, var_6: var_27}
    var_29 = False
    var_30 = module_5.Definitions()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_30) == 0
    var_31 = module_0.from_json_schema_type(var_28, var_23, var_29, var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.String'
    assert var_31.default == 'test'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.allow_blank is False
    assert var_31.trim_whitespace is True
    assert var_31.max_length == 100
    assert var_31.min_length == 5
    assert var_31.format == 'email'
    assert var_31.coerce_types is False
    assert var_31.pattern == '^[a-zA-Z0-9]+$'
    assert f'{type(var_31.pattern_regex).__module__}.{type(var_31.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_32 = 'boolean'
    var_33 = True
    var_34 = {var_0: var_32, var_6: var_33}
    var_35 = False
    var_36 = module_5.Definitions()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_36) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_37 = module_0.from_json_schema_type(var_34, var_32, var_35, var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_37.default is True
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_38 = 'items'
    var_39 = 'additionalItems'
    var_40 = 'minItems'
    var_41 = 'maxItems'
    var_42 = 'uniqueItems'
    var_43 = 'array'
    var_44 = {var_0: var_23}
    var_45 = False
    var_46 = 10
    var_47 = [var_27]
    var_48 = {var_0: var_43, var_38: var_44, var_39: var_45, var_40: var_33, var_41: var_46, var_42: var_33, var_6: var_47}
    var_49 = False
    var_50 = module_5.Definitions()
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_50) == 0
    var_51 = module_0.from_json_schema_type(var_48, var_43, var_49, var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Array'
    assert var_51.default == ['test']
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.items).__module__}.{type(var_51.items).__qualname__}' == 'typesystem.fields.String'
    assert var_51.additional_items is False
    assert var_51.min_items is True
    assert var_51.max_items == 10
    assert var_51.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_52 = var_51.items
    var_53 = 'B1\n'
    var_54 = 'patternProperties'
    var_55 = 'additionalProperties'
    var_56 = 'propertyNames'
    var_57 = 'minProperties'
    var_58 = 'required'
    var_59 = 'object'
    var_60 = 'name'
    var_61 = 'age'
    var_62 = {var_0: var_23}
    var_63 = {var_0: var_16}
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = '^S_'
    var_66 = '^I_'
    var_67 = {var_0: var_23}
    var_68 = {var_0: var_16}
    var_69 = {var_65: var_67, var_66: var_68}
    var_70 = False
    var_71 = {var_0: var_23}
    var_72 = [var_60]
    var_73 = 25
    var_74 = {var_60: var_27, var_61: var_73}
    var_75 = {var_0: var_59, var_53: var_64, var_54: var_69, var_55: var_70, var_56: var_71, var_57: var_33, var_25: var_46, var_58: var_72, var_6: var_74}
    var_76 = False
    var_77 = module_5.Definitions()
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_77) == 0
    var_78 = module_0.from_json_schema_type(var_75, var_59, var_76, var_77)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.Object'
    assert var_78.default == {'name': 'test', 'age': 25}
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert var_78.properties == {}
    assert f'{type(var_78.pattern_properties).__module__}.{type(var_78.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_78.pattern_properties) == 2
    assert var_78.additional_properties is False
    assert f'{type(var_78.property_names).__module__}.{type(var_78.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_78.min_properties is True
    assert var_78.max_properties is None
    assert var_78.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    with pytest.raises(KeyError):
        var_79 = var_78.properties[var_60]

def test_case_46():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_17 = 'array'
    var_18 = {var_4: var_17}
    var_19 = module_0.from_json_schema(var_18)
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
    var_22 = module_0.from_json_schema(var_21)
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
    var_23 = 'enum'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'c'
    var_27 = [var_24, var_25, var_26]
    var_28 = {var_23: var_27}
    var_29 = module_0.from_json_schema(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Choice'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert var_29.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_29.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_30 = 'const'
    var_31 = 'fixed_value'
    var_32 = {var_30: var_31}
    var_33 = module_0.from_json_schema(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Const'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_34 = 'allOf'
    var_35 = 'minLength'
    var_36 = 5
    var_37 = {var_4: var_5, var_35: var_36}
    var_38 = 'maxLength'
    var_39 = 10
    var_40 = {var_4: var_5, var_38: var_39}
    var_41 = [var_37, var_40]
    var_42 = {var_34: var_41}
    var_43 = module_0.from_json_schema(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert f'{type(var_43.all_of).__module__}.{type(var_43.all_of).__qualname__}' == 'builtins.list'
    assert len(var_43.all_of) == 2
    with pytest.raises(AttributeError):
        var_44 = var_43.schemas

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = None
    var_1 = module_3._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_0.from_json_schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = module_0.type_from_json_schema(var_1, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is True
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_0.to_json_schema(var_2)
    assert var_5 is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_6 = module_0.to_json_schema(var_4, var_0)
    var_7 = module_2.Object()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is True
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = module_1.Not(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.Not'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.negated).__module__}.{type(var_8.negated).__qualname__}' == 'typesystem.fields.Object'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_9 = var_4.get_default_value()
    var_10 = module_0.to_json_schema(var_8, var_0)
    var_11 = module_2.Float(minimum=var_9, exclusive_minimum=var_5, precision=var_0, coerce_types=var_6, **var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Float'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.minimum is None
    assert var_11.maximum is None
    assert var_11.exclusive_minimum is True
    assert var_11.exclusive_maximum is None
    assert var_11.multiple_of is None
    assert var_11.precision is None
    assert var_11.coerce_types == {'anyOf': [{'type': 'boolean'}, {'type': 'object'}, {'type': 'number'}, {'type': 'array', 'minItems': 0, 'additionalItems': True}, {'type': 'string', 'default': ''}], 'default': None}
    var_12 = var_8.validate(var_0)
    var_13 = var_8.validate(var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Float'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum is None
    assert var_13.maximum is None
    assert var_13.exclusive_minimum is True
    assert var_13.exclusive_maximum is None
    assert var_13.multiple_of is None
    assert var_13.precision is None
    assert var_13.coerce_types == {'anyOf': [{'type': 'boolean'}, {'type': 'object'}, {'type': 'number'}, {'type': 'array', 'minItems': 0, 'additionalItems': True}, {'type': 'string', 'default': ''}], 'default': None}
    var_14 = module_0.to_json_schema(var_13)
    module_0.type_from_json_schema(var_11, var_9)

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = None
    var_1 = module_3._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_0.from_json_schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = var_2.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = module_0.type_from_json_schema(var_1, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is True
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_0.to_json_schema(var_2)
    assert var_5 is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_6 = module_5.Definitions()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 0
    var_7 = module_2.Object()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is True
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = module_0.to_json_schema(var_7)
    var_9 = module_1.Not(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.Not'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.negated).__module__}.{type(var_9.negated).__qualname__}' == 'typesystem.fields.Object'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_10 = module_4.RegexFlag.DEBUG
    var_11 = module_0.to_json_schema(var_9, var_0)
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
    var_12 = module_2.Float(minimum=var_10, exclusive_minimum=var_5, precision=var_0, coerce_types=var_5, **var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Float'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum == module_4.RegexFlag.DEBUG
    assert var_12.maximum is None
    assert var_12.exclusive_minimum is True
    assert var_12.exclusive_maximum is None
    assert var_12.multiple_of is None
    assert var_12.precision is None
    assert var_12.coerce_types is True
    var_13 = module_3._EnumDict()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'enum._EnumDict'
    assert len(var_13) == 0
    var_14 = var_9.validate(var_0)
    var_15 = var_9.validate(var_12)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Float'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum == module_4.RegexFlag.DEBUG
    assert var_15.maximum is None
    assert var_15.exclusive_minimum is True
    assert var_15.exclusive_maximum is None
    assert var_15.multiple_of is None
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
    module_0.type_from_json_schema(var_12, var_10)

def test_case_49():
    var_0 = module_3._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_5.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.fields).__module__}.{type(var_1.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.fields) == 0
    assert var_1.required == []
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = module_0.to_json_schema(var_1, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.to_json_schema(var_1)

def test_case_50():
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    assert var_6.min_length is True
    assert var_6.format == 'email'
    assert var_6.coerce_types is True
    assert var_6.pattern == '^[a-z]+$'
    assert f'{type(var_6.pattern_regex).__module__}.{type(var_6.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = module_0.to_json_schema(var_6)
    var_8 = False
    var_9 = 100
    var_10 = module_2.Integer(minimum=var_8, maximum=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Integer'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.minimum is False
    assert var_10.maximum == 100
    assert var_10.exclusive_minimum is None
    assert var_10.exclusive_maximum is None
    assert var_10.multiple_of is None
    assert var_10.precision is None
    assert var_10.coerce_types is True
    var_11 = module_0.to_json_schema(var_10)
    var_12 = module_2.Float(multiple_of=var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Float'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum is None
    assert var_12.maximum is None
    assert var_12.exclusive_minimum is None
    assert var_12.exclusive_maximum is None
    assert var_12.multiple_of == 100
    assert var_12.precision is None
    assert var_12.coerce_types is True
    var_13 = module_0.to_json_schema(var_12)
    var_14 = module_2.Boolean()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_15 = module_0.to_json_schema(var_14)
    var_16 = module_2.String()
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
    var_17 = 5
    var_18 = module_2.Array(var_16, min_items=var_2, max_items=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Array'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert f'{type(var_18.items).__module__}.{type(var_18.items).__qualname__}' == 'typesystem.fields.String'
    assert var_18.additional_items is False
    assert var_18.min_items is True
    assert var_18.max_items == 5
    assert var_18.unique_items is False
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
    var_23 = [var_20]
    var_24 = module_2.Object(properties=var_22, required=var_23)
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
    assert var_24.min_properties is None
    assert var_24.max_properties is None
    assert var_24.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_25 = module_3._EnumDict()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'enum._EnumDict'
    assert len(var_25) == 0
    var_26 = module_0.to_json_schema(var_24)
    var_27 = 'a'
    var_28 = (var_27, var_27)
    var_29 = '_'
    var_30 = (var_29, var_29)
    var_31 = [var_28, var_30]
    var_32 = module_2.Choice(choices=var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Choice'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.choices == [('a', 'a'), ('_', '_')]
    assert var_32.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_33 = module_0.to_json_schema(var_32)
    var_34 = 'fixed_value'
    var_35 = module_2.Const(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Const'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_36 = module_0.to_json_schema(var_35)
    var_37 = module_2.String()
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
    var_38 = module_2.Integer()
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
    var_40 = module_2.Union(var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Union'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert f'{type(var_40.any_of).__module__}.{type(var_40.any_of).__qualname__}' == 'builtins.list'
    assert len(var_40.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}

@pytest.mark.xfail(strict=True)
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
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
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
    assert var_8.min_length == 1
    assert var_8.format == 'email'
    assert var_8.coerce_types is True
    assert var_8.pattern == '[a-z]+'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = 0
    var_11 = 100
    var_12 = True
    var_13 = module_2.Integer(minimum=var_10, maximum=var_11, exclusive_maximum=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum == 0
    assert var_13.maximum == 100
    assert var_13.exclusive_minimum is None
    assert var_13.exclusive_maximum is True
    assert var_13.multiple_of is None
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = True
    var_15 = module_0.to_json_schema(var_13)
    var_16 = 0.5
    var_17 = module_2.Float(multiple_of=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Float'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum is None
    assert var_17.maximum is None
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of == pytest.approx(0.5, abs=0.01, rel=0.01)
    assert var_17.precision is None
    assert var_17.coerce_types is True
    var_18 = module_0.to_json_schema(var_17)
    var_19 = module_2.Boolean()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_20 = module_0.to_json_schema(var_19)
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
    var_22 = 5
    var_23 = True
    var_24 = module_2.Array(var_21, min_items=var_14, max_items=var_22, unique_items=var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Array'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.items).__module__}.{type(var_24.items).__qualname__}' == 'typesystem.fields.String'
    assert var_24.additional_items is False
    assert var_24.min_items is True
    assert var_24.max_items == 5
    assert var_24.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_25 = True
    var_26 = module_0.to_json_schema(var_24)
    var_27 = 'name'
    var_28 = 'fixed_value'
    var_29 = module_2.Const(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Const'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert var_29.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_30 = module_0.to_json_schema(var_29)
    var_31 = module_2.String()
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
    var_32 = module_2.Integer()
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
    var_34 = module_2.Union(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Union'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.any_of).__module__}.{type(var_34.any_of).__qualname__}' == 'builtins.list'
    assert len(var_34.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_35 = module_0.to_json_schema(var_34)
    var_36 = module_2.String(min_length=var_25)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.String'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.allow_blank is False
    assert var_36.trim_whitespace is True
    assert var_36.max_length is None
    assert var_36.min_length is True
    assert var_36.format is None
    assert var_36.coerce_types is True
    assert var_36.pattern is None
    assert var_36.pattern_regex is None
    var_37 = module_2.String(max_length=var_5)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.String'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.allow_blank is False
    assert var_37.trim_whitespace is True
    assert var_37.max_length == 10
    assert var_37.min_length is None
    assert var_37.format is None
    assert var_37.coerce_types is True
    assert var_37.pattern is None
    assert var_37.pattern_regex is None
    var_38 = [var_36, var_37]
    var_39 = module_1.AllOf(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.all_of).__module__}.{type(var_39.all_of).__qualname__}' == 'builtins.list'
    assert len(var_39.all_of) == 2
    var_40 = module_0.to_json_schema(var_39)
    var_41 = module_2.String()
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
    var_42 = module_2.Integer()
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
    var_44 = module_1.OneOf(var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert f'{type(var_44.one_of).__module__}.{type(var_44.one_of).__qualname__}' == 'builtins.list'
    assert len(var_44.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_45 = module_0.to_json_schema(var_44)
    var_46 = module_2.String()
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
    var_47 = module_1.Not(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.composites.Not'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert f'{type(var_47.negated).__module__}.{type(var_47.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_48 = module_0.to_json_schema(var_47)
    var_49 = module_2.String()
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
    var_50 = module_2.Integer()
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
    var_51 = module_2.Boolean()
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert var_51.coerce_types is True
    var_52 = module_1.IfThenElse(var_49, var_50, var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert f'{type(var_52.if_clause).__module__}.{type(var_52.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_52.then_clause).__module__}.{type(var_52.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_52.else_clause).__module__}.{type(var_52.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_53 = module_0.to_json_schema(var_52)
    var_54 = module_2.String()
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
    var_55 = {var_27: var_54}
    var_56 = module_2.Object(properties=var_55)
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
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
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
    var_58 = {var_27: var_57}
    var_59 = module_5.Schema(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.fields).__module__}.{type(var_59.fields).__qualname__}' == 'builtins.dict'
    assert len(var_59.fields) == 1
    assert var_59.required == ['name']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_60 = module_0.to_json_schema(var_59)
    var_61 = 'Address'
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
    var_47.validate(var_61)

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
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    assert var_6.min_length is True
    assert var_6.format == 'email'
    assert var_6.coerce_types is True
    assert var_6.pattern == '^[a-z]+$'
    assert f'{type(var_6.pattern_regex).__module__}.{type(var_6.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = module_0.to_json_schema(var_6)
    var_8 = False
    var_9 = 100
    var_10 = module_2.Integer(minimum=var_8, maximum=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Integer'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.minimum is False
    assert var_10.maximum == 100
    assert var_10.exclusive_minimum is None
    assert var_10.exclusive_maximum is None
    assert var_10.multiple_of is None
    assert var_10.precision is None
    assert var_10.coerce_types is True
    var_11 = module_0.to_json_schema(var_10)
    var_12 = module_2.Float(multiple_of=var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Float'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum is None
    assert var_12.maximum is None
    assert var_12.exclusive_minimum is None
    assert var_12.exclusive_maximum is None
    assert var_12.multiple_of == 100
    assert var_12.precision is None
    assert var_12.coerce_types is True
    var_13 = module_0.to_json_schema(var_12)
    var_14 = module_2.Boolean()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_15 = module_0.to_json_schema(var_14)
    var_16 = module_2.String()
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
    var_17 = 5
    var_18 = module_2.Array(var_16, min_items=var_2, max_items=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Array'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert f'{type(var_18.items).__module__}.{type(var_18.items).__qualname__}' == 'typesystem.fields.String'
    assert var_18.additional_items is False
    assert var_18.min_items is True
    assert var_18.max_items == 5
    assert var_18.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_19 = module_0.to_json_schema(var_18)
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
    var_21 = module_3._EnumDict()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'enum._EnumDict'
    assert len(var_21) == 0
    var_22 = '_'
    var_23 = (var_22, var_22)
    var_24 = [var_23, var_23]
    var_25 = module_2.Choice(choices=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Choice'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.choices == [('_', '_'), ('_', '_')]
    assert var_25.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_26 = module_0.to_json_schema(var_25)
    var_27 = 'fixed_value'
    var_28 = module_2.Const(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Const'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
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
    var_31 = module_2.Integer()
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
    var_32 = [var_30, var_31]
    var_33 = module_2.Union(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Union'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.any_of).__module__}.{type(var_33.any_of).__qualname__}' == 'builtins.list'
    assert len(var_33.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}

@pytest.mark.xfail(strict=True)
def test_case_53():
    var_0 = 10
    var_1 = '[a-z]+'
    var_2 = 'uA7il'
    var_3 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length == 10
    assert var_3.min_length == 10
    assert var_3.format == 'uA7il'
    assert var_3.coerce_types is True
    assert var_3.pattern == '[a-z]+'
    assert f'{type(var_3.pattern_regex).__module__}.{type(var_3.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = 4
    var_5 = module_2.Float(multiple_of=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Float'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum is None
    assert var_5.maximum is None
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of == 4
    assert var_5.precision is None
    assert var_5.coerce_types is True
    var_6 = False
    var_7 = module_2.Integer()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Integer'
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
    var_8 = [var_3, var_7]
    var_9 = module_2.Union(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Union'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.any_of).__module__}.{type(var_9.any_of).__qualname__}' == 'builtins.list'
    assert len(var_9.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_10 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_11 = module_1.IfThenElse(var_3, var_6, var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.if_clause).__module__}.{type(var_11.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert var_11.then_clause is False
    assert var_11.else_clause is False
    module_0.to_json_schema(var_11)

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
    var_1 = module_0.get_standard_properties(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_5 = module_2.String(max_length=var_3, min_length=var_2, pattern=var_4, format=var_4)
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = 'null'
    var_7 = module_0.to_json_schema(var_5)
    var_8 = False
    var_9 = 100
    var_10 = module_2.Integer(minimum=var_8, maximum=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Integer'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.minimum is False
    assert var_10.maximum == 100
    assert var_10.exclusive_minimum is None
    assert var_10.exclusive_maximum is None
    assert var_10.multiple_of is None
    assert var_10.precision is None
    assert var_10.coerce_types is True
    var_11 = module_0.to_json_schema(var_10)
    var_12 = module_2.Float(multiple_of=var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Float'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum is None
    assert var_12.maximum is None
    assert var_12.exclusive_minimum is None
    assert var_12.exclusive_maximum is None
    assert var_12.multiple_of == 100
    assert var_12.precision is None
    assert var_12.coerce_types is True
    var_13 = 'number'
    var_14 = [var_13, var_6]
    var_15 = module_0.to_json_schema(var_12)
    var_16 = module_2.Boolean()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_17 = module_0.to_json_schema(var_16)
    var_18 = module_2.String()
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
    module_0.to_json_schema(var_14)

@pytest.mark.xfail(strict=True)
def test_case_55():
    var_0 = True
    var_1 = 10
    var_2 = '^[a-z]+$'
    var_3 = module_2.String(max_length=var_1, min_length=var_0, pattern=var_2, format=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length == 10
    assert var_3.min_length is True
    assert var_3.format == '^[a-z]+$'
    assert var_3.coerce_types is True
    assert var_3.pattern == '^[a-z]+$'
    assert f'{type(var_3.pattern_regex).__module__}.{type(var_3.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = 100
    var_6 = 'minik(mum'
    var_7 = module_2.Float(multiple_of=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Float'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.minimum is None
    assert var_7.maximum is None
    assert var_7.exclusive_minimum is None
    assert var_7.exclusive_maximum is None
    assert var_7.multiple_of == 100
    assert var_7.precision is None
    assert var_7.coerce_types is True
    var_8 = module_0.to_json_schema(var_7)
    var_9 = var_3.serialize(var_6)
    assert var_9 == 'minik(mum'
    module_0.to_json_schema(var_9)

def test_case_56():
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = var_0.has_default()
    assert var_1 is False
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_2.Array(var_2, min_items=var_1, max_items=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.String'
    assert var_3.additional_items is False
    assert var_3.min_items is False
    assert var_3.max_items is False
    assert var_3.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'uA7il'
    var_8 = module_2.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length == 10
    assert var_8.min_length == 1
    assert var_8.format == 'uA7il'
    assert var_8.coerce_types is True
    assert var_8.pattern == '[a-z]+'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_3._EnumDict()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'enum._EnumDict'
    assert len(var_9) == 0
    var_10 = 0.5
    var_11 = module_2.Float(multiple_of=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Float'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.minimum is None
    assert var_11.maximum is None
    assert var_11.exclusive_minimum is None
    assert var_11.exclusive_maximum is None
    assert var_11.multiple_of == pytest.approx(0.5, abs=0.01, rel=0.01)
    assert var_11.precision is None
    assert var_11.coerce_types is True
    var_12 = 14
    var_13 = True
    var_14 = module_2.Array(var_8, min_items=var_3, max_items=var_12, unique_items=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Array'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.items).__module__}.{type(var_14.items).__qualname__}' == 'typesystem.fields.String'
    assert var_14.additional_items is False
    assert var_14.min_items is False
    assert var_14.max_items == 14
    assert var_14.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_15 = False
    var_16 = module_0.to_json_schema(var_14)
    var_17 = 'name'
    var_18 = {var_17: var_8}
    var_19 = False
    var_20 = var_8.get_default_value()
    var_21 = module_2.Object(properties=var_18, additional_properties=var_19, required=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Object'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.properties).__module__}.{type(var_21.properties).__qualname__}' == 'builtins.dict'
    assert len(var_21.properties) == 1
    assert var_21.pattern_properties == {}
    assert var_21.additional_properties is False
    assert var_21.property_names is None
    assert var_21.min_properties is None
    assert var_21.max_properties is None
    assert var_21.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_22 = var_14.has_default()
    assert var_22 is False
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
    var_26 = module_2.Union(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Union'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.any_of).__module__}.{type(var_26.any_of).__qualname__}' == 'builtins.list'
    assert len(var_26.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_27 = module_0.to_json_schema(var_26)
    var_28 = [var_8, var_23]
    var_29 = module_1.AllOf(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.all_of).__module__}.{type(var_29.all_of).__qualname__}' == 'builtins.list'
    assert len(var_29.all_of) == 2
    var_30 = module_0.to_json_schema(var_29)
    var_31 = module_2.String(min_length=var_15)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.String'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.allow_blank is False
    assert var_31.trim_whitespace is True
    assert var_31.max_length is None
    assert var_31.min_length is False
    assert var_31.format is None
    assert var_31.coerce_types is True
    assert var_31.pattern is None
    assert var_31.pattern_regex is None
    var_32 = module_2.Integer()
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
    var_33 = module_1.IfThenElse(var_31, var_32, var_20)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.if_clause).__module__}.{type(var_33.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_33.then_clause).__module__}.{type(var_33.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_33.else_clause).__module__}.{type(var_33.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_34 = module_0.to_json_schema(var_33)
    var_35 = module_4.purge()
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
    var_36 = module_1.Not(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.composites.Not'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.negated is None
    assert module_1.Not.errors == {'negated': 'Must not match.'}

@pytest.mark.xfail(strict=True)
def test_case_58():
    var_0 = 10
    var_1 = '[a-z]+'
    var_2 = 'uA7il'
    var_3 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length == 10
    assert var_3.min_length == 10
    assert var_3.format == 'uA7il'
    assert var_3.coerce_types is True
    assert var_3.pattern == '[a-z]+'
    assert f'{type(var_3.pattern_regex).__module__}.{type(var_3.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = 4
    var_5 = module_2.Float(multiple_of=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Float'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum is None
    assert var_5.maximum is None
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of == 4
    assert var_5.precision is None
    assert var_5.coerce_types is True
    var_6 = False
    var_7 = module_2.Integer()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Integer'
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
    var_8 = [var_3, var_7]
    var_9 = module_2.Union(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Union'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.any_of).__module__}.{type(var_9.any_of).__qualname__}' == 'builtins.list'
    assert len(var_9.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_10 = module_4.purge()
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
    var_11 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_12 = module_1.IfThenElse(var_3, var_10, var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.if_clause).__module__}.{type(var_12.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_12.then_clause).__module__}.{type(var_12.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert var_12.else_clause is False
    module_0.to_json_schema(var_12)

def test_case_59():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_5.Definitions()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 0
    var_4 = module_0.type_from_json_schema(var_2, var_3)
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_5 = 'number'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_5.Definitions()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_9 = module_0.type_from_json_schema(var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Union'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.any_of).__module__}.{type(var_9.any_of).__qualname__}' == 'builtins.list'
    assert len(var_9.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_10 = var_9.any_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 'null'
    var_13 = [var_1, var_12]
    var_14 = {var_0: var_13}
    var_15 = module_5.Definitions()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_15) == 0
    var_16 = module_0.type_from_json_schema(var_14, var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.String'
    assert var_16.default is None
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is True
    assert var_16.read_only is False
    assert var_16.allow_blank is True
    assert var_16.trim_whitespace is True
    assert var_16.max_length is None
    assert var_16.min_length is None
    assert var_16.format is None
    assert var_16.coerce_types is False
    assert var_16.pattern is None
    assert var_16.pattern_regex is None
    var_17 = {}
    var_18 = module_5.Definitions()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_18) == 0
    var_19 = module_0.type_from_json_schema(var_17, var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Union'
    assert var_19.default is None
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is True
    assert var_19.read_only is False
    assert f'{type(var_19.any_of).__module__}.{type(var_19.any_of).__qualname__}' == 'builtins.list'
    assert len(var_19.any_of) == 5
    var_20 = {var_0: var_12}
    var_21 = module_5.Definitions()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_21) == 0
    var_22 = module_0.type_from_json_schema(var_20, var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Const'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.const is None
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_23 = 'minLength'
    var_24 = 5
    var_25 = {var_0: var_1, var_23: var_24}
    var_26 = module_5.Definitions()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_26) == 0
    var_27 = module_0.type_from_json_schema(var_25, var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.String'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.allow_blank is False
    assert var_27.trim_whitespace is True
    assert var_27.max_length is None
    assert var_27.min_length == 5
    assert var_27.format is None
    assert var_27.coerce_types is False
    assert var_27.pattern is None
    assert var_27.pattern_regex is None
    var_28 = 'minimum'
    var_29 = 'maximum'
    var_30 = 0
    var_31 = 100
    var_32 = {var_0: var_5, var_28: var_30, var_29: var_31}
    var_33 = module_5.Definitions()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_33) == 0
    var_34 = module_0.type_from_json_schema(var_32, var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Float'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.minimum == 0
    assert var_34.maximum == 100
    assert var_34.exclusive_minimum is None
    assert var_34.exclusive_maximum is None
    assert var_34.multiple_of is None
    assert var_34.precision is None
    assert var_34.coerce_types is False

@pytest.mark.xfail(strict=True)
def test_case_60():
    var_0 = module_2.Object()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.fields.Object'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_3 = var_0.has_default()
    assert var_3 is False
    var_4 = module_0.to_json_schema(var_2, var_3)
    var_5 = var_0.get_default_value()
    var_6 = module_0.from_json_schema(var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.Not'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.negated).__module__}.{type(var_6.negated).__qualname__}' == 'typesystem.fields.Object'
    var_7 = module_0.type_from_json_schema(var_1, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is True
    assert var_7.property_names is None
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    var_8 = var_2.__or__(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Union'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.any_of).__module__}.{type(var_8.any_of).__qualname__}' == 'builtins.list'
    assert len(var_8.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_9 = module_0.to_json_schema(var_0)
    module_0.ref_from_json_schema(var_5, var_5)

def test_case_61():
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
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
    assert var_8.min_length == 1
    assert var_8.format == 'email'
    assert var_8.coerce_types is True
    assert var_8.pattern == '[a-z]+'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = 0
    var_11 = 100
    var_12 = module_2.Integer(minimum=var_10, maximum=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Integer'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum == 0
    assert var_12.maximum == 100
    assert var_12.exclusive_minimum is None
    assert var_12.exclusive_maximum is None
    assert var_12.multiple_of is None
    assert var_12.precision is None
    assert var_12.coerce_types is True
    var_13 = module_0.to_json_schema(var_12)
    var_14 = 0.5
    var_15 = module_2.Float(multiple_of=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Float'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is None
    assert var_15.maximum is None
    assert var_15.exclusive_minimum is None
    assert var_15.exclusive_maximum is None
    assert var_15.multiple_of == pytest.approx(0.5, abs=0.01, rel=0.01)
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_2.Boolean()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_18 = module_0.to_json_schema(var_17)
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
    var_20 = 5
    var_21 = True
    var_22 = module_2.Array(var_19, min_items=var_4, max_items=var_20, unique_items=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Array'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.items).__module__}.{type(var_22.items).__qualname__}' == 'typesystem.fields.String'
    assert var_22.additional_items is False
    assert var_22.min_items == 1
    assert var_22.max_items == 5
    assert var_22.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_23 = True
    var_24 = module_0.to_json_schema(var_22)
    var_25 = 'name'
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
    var_27 = {var_25: var_26}
    var_28 = [var_25]
    var_29 = module_2.Object(properties=var_27, min_properties=var_23, max_properties=var_20, required=var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Object'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.properties).__module__}.{type(var_29.properties).__qualname__}' == 'builtins.dict'
    assert len(var_29.properties) == 1
    assert var_29.pattern_properties == {}
    assert var_29.additional_properties is True
    assert var_29.property_names is None
    assert var_29.min_properties is True
    assert var_29.max_properties == 5
    assert var_29.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_30 = module_0.to_json_schema(var_29)
    var_31 = 'a'
    var_32 = (var_31, var_31)
    var_33 = 'b'
    var_34 = (var_33, var_33)
    var_35 = [var_32, var_34]
    var_36 = module_2.Choice(choices=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Choice'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.choices == [('a', 'a'), ('b', 'b')]
    assert var_36.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_37 = module_0.to_json_schema(var_36)
    var_38 = 'fixed_value'
    var_39 = module_2.Const(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Const'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_40 = module_0.to_json_schema(var_39)
    var_41 = module_2.String()
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
    var_42 = module_2.Integer()
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
    var_44 = module_2.Union(var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Union'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert f'{type(var_44.any_of).__module__}.{type(var_44.any_of).__qualname__}' == 'builtins.list'
    assert len(var_44.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_45 = module_0.to_json_schema(var_44)
    var_46 = module_2.String()
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
    var_47 = 'test'
    var_48 = module_2.Const(var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.Const'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert var_48.const == 'test'
    var_49 = [var_46, var_48]
    var_50 = module_1.AllOf(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert f'{type(var_50.all_of).__module__}.{type(var_50.all_of).__qualname__}' == 'builtins.list'
    assert len(var_50.all_of) == 2
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
    var_53 = module_2.String()
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
    var_54 = module_2.Integer()
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Integer'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert var_54.minimum is None
    assert var_54.maximum is None
    assert var_54.exclusive_minimum is None
    assert var_54.exclusive_maximum is None
    assert var_54.multiple_of is None
    assert var_54.precision is None
    assert var_54.coerce_types is True
    var_55 = module_2.Boolean()
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.coerce_types is True
    var_56 = module_1.IfThenElse(var_53, var_54, var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.if_clause).__module__}.{type(var_56.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_56.then_clause).__module__}.{type(var_56.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_56.else_clause).__module__}.{type(var_56.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_57 = module_0.to_json_schema(var_56)
    var_58 = module_2.String()
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.String'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.allow_blank is False
    assert var_58.trim_whitespace is True
    assert var_58.max_length is None
    assert var_58.min_length is None
    assert var_58.format is None
    assert var_58.coerce_types is True
    assert var_58.pattern is None
    assert var_58.pattern_regex is None
    var_59 = module_1.Not(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.Not'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.negated).__module__}.{type(var_59.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_60 = module_0.to_json_schema(var_59)

def test_case_62():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 'number'
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
    var_12 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11}
    var_13 = False
    var_14 = module_5.Definitions()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_14) == 0
    var_15 = module_0.from_json_schema_type(var_12, var_7, var_13, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Float'
    assert var_15.default == 50
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum == 0
    assert var_15.maximum == 100
    assert var_15.exclusive_minimum == 0
    assert var_15.exclusive_maximum == 100
    assert var_15.multiple_of == 2
    assert var_15.precision is None
    assert var_15.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_16 = 'integer'
    var_17 = {var_0: var_16, var_1: var_13, var_2: var_9, var_3: var_13, var_4: var_9, var_5: var_10, var_6: var_11}
    var_18 = module_0.from_json_schema_type(var_17, var_16, var_13, var_14)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Integer'
    assert var_18.default == 50
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.minimum is False
    assert var_18.maximum == 100
    assert var_18.exclusive_minimum is False
    assert var_18.exclusive_maximum == 100
    assert var_18.multiple_of == 2
    assert var_18.precision is None
    assert var_18.coerce_types is False
    var_19 = 'minLength'
    var_20 = 'maxLength'
    var_21 = 'format'
    var_22 = 'pattern'
    var_23 = 'string'
    var_24 = 5
    var_25 = 'edCPl'
    var_26 = '^[a-zA-Z0-9]+$'
    var_27 = 'test'
    var_28 = {var_0: var_23, var_19: var_24, var_20: var_9, var_21: var_25, var_22: var_26, var_6: var_27}
    var_29 = False
    var_30 = module_5.Definitions()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_30) == 0
    var_31 = module_0.from_json_schema_type(var_28, var_23, var_29, var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.String'
    assert var_31.default == 'test'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.allow_blank is False
    assert var_31.trim_whitespace is True
    assert var_31.max_length == 100
    assert var_31.min_length == 5
    assert var_31.format == 'edCPl'
    assert var_31.coerce_types is False
    assert var_31.pattern == '^[a-zA-Z0-9]+$'
    assert f'{type(var_31.pattern_regex).__module__}.{type(var_31.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_32 = 'boolean'
    var_33 = True
    var_34 = {var_0: var_32, var_6: var_33}
    var_35 = False
    var_36 = module_5.Definitions()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_36) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_37 = module_0.from_json_schema_type(var_34, var_32, var_35, var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_37.default is True
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_38 = 'items'
    var_39 = 'additionalItems'
    var_40 = 'minItems'
    var_41 = 'maxItems'
    var_42 = 'uniqueItems'
    var_43 = 'array'
    var_44 = {var_0: var_23}
    var_45 = {var_0: var_7}
    var_46 = 10
    var_47 = [var_27]
    var_48 = {var_0: var_43, var_38: var_44, var_39: var_45, var_40: var_33, var_41: var_46, var_42: var_33, var_6: var_47}
    var_49 = False
    var_50 = module_5.Definitions()
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_50) == 0
    var_51 = module_0.from_json_schema_type(var_48, var_43, var_49, var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Array'
    assert var_51.default == ['test']
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.items).__module__}.{type(var_51.items).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_51.additional_items).__module__}.{type(var_51.additional_items).__qualname__}' == 'typesystem.fields.Float'
    assert var_51.min_items is True
    assert var_51.max_items == 10
    assert var_51.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_52 = var_51.items
    var_53 = var_51.additional_items
    var_54 = 'properties'
    var_55 = 'patternProperties'
    var_56 = 'additionalProperties'
    var_57 = 'propertyNames'
    var_58 = 'minProperties'
    var_59 = 'maxProperties'
    var_60 = 'required'
    var_61 = 'oje'
    var_62 = 'name'
    var_63 = 'age'
    var_64 = {var_0: var_23}
    var_65 = {var_0: var_16}
    var_66 = {var_62: var_64, var_63: var_65}
    var_67 = '^S_'
    var_68 = '^I_'
    var_69 = {var_0: var_23}
    var_70 = {var_0: var_16}
    var_71 = {var_67: var_69, var_68: var_70}
    var_72 = {var_0: var_32}
    var_73 = {var_0: var_23}
    var_74 = [var_62]
    var_75 = 30
    var_76 = {var_62: var_27, var_63: var_75}
    var_77 = {var_0: var_61, var_54: var_66, var_55: var_71, var_56: var_72, var_57: var_73, var_58: var_33, var_59: var_46, var_60: var_74, var_6: var_76}
    var_78 = False
    var_79 = module_5.Definitions()
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_79) == 0
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_77, var_61, var_78, var_79)

def test_case_63():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'default'
    var_6 = 'number'
    var_7 = 0
    var_8 = 100
    var_9 = 2
    var_10 = 50
    var_11 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_7, var_4: var_8, var_2: var_9, var_5: var_10}
    var_12 = False
    var_13 = module_5.Definitions()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13) == 0
    var_14 = module_0.from_json_schema_type(var_11, var_6, var_12, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Float'
    assert var_14.default == 50
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum == 0
    assert var_14.maximum == 2
    assert var_14.exclusive_minimum == 0
    assert var_14.exclusive_maximum == 100
    assert var_14.multiple_of is None
    assert var_14.precision is None
    assert var_14.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_15 = 'integer'
    var_16 = {var_0: var_15, var_1: var_12, var_2: var_8, var_3: var_12, var_4: var_8, var_3: var_9, var_5: var_10}
    var_17 = module_5.Definitions()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_17) == 0
    var_18 = module_0.from_json_schema_type(var_16, var_15, var_12, var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Integer'
    assert var_18.default == 50
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.minimum is False
    assert var_18.maximum == 100
    assert var_18.exclusive_minimum == 2
    assert var_18.exclusive_maximum == 100
    assert var_18.multiple_of is None
    assert var_18.precision is None
    assert var_18.coerce_types is False
    var_19 = 'minLength'
    var_20 = 'maxLength'
    var_21 = 'format'
    var_22 = 'pattern'
    var_23 = 'string'
    var_24 = 5
    var_25 = 'email'
    var_26 = '^[a-zA-Z0-9]+$'
    var_27 = 'test'
    var_28 = {var_0: var_23, var_19: var_24, var_20: var_8, var_21: var_25, var_22: var_26, var_5: var_27}
    var_29 = False
    var_30 = module_5.Definitions()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_30) == 0
    var_31 = module_0.from_json_schema_type(var_28, var_23, var_29, var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.String'
    assert var_31.default == 'test'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.allow_blank is False
    assert var_31.trim_whitespace is True
    assert var_31.max_length == 100
    assert var_31.min_length == 5
    assert var_31.format == 'email'
    assert var_31.coerce_types is False
    assert var_31.pattern == '^[a-zA-Z0-9]+$'
    assert f'{type(var_31.pattern_regex).__module__}.{type(var_31.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_32 = 'boolean'
    var_33 = True
    var_34 = {var_0: var_32, var_5: var_33}
    var_35 = module_5.Definitions()
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_35) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_36 = module_0.from_json_schema_type(var_34, var_32, var_28, var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_36.default is True
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null == {'type': 'string', 'minLength': 5, 'maxLength': 100, 'format': 'email', 'pattern': '^[a-zA-Z0-9]+$', 'default': 'test'}
    assert var_36.read_only is False
    assert var_36.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_37 = 'items'
    var_38 = 'additionalItems'
    var_39 = 'minItems'
    var_40 = 'maxItems'
    var_41 = 'uniqueItems'
    var_42 = 'array'
    var_43 = {var_0: var_23}
    var_44 = {var_0: var_6}
    var_45 = 10
    var_46 = [var_27]
    var_47 = {var_0: var_42, var_37: var_43, var_38: var_44, var_39: var_33, var_40: var_45, var_41: var_33, var_5: var_46}
    var_48 = False
    var_49 = module_5.Definitions()
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_49) == 0
    var_50 = module_0.from_json_schema_type(var_47, var_42, var_48, var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.fields.Array'
    assert var_50.default == ['test']
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert f'{type(var_50.items).__module__}.{type(var_50.items).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_50.additional_items).__module__}.{type(var_50.additional_items).__qualname__}' == 'typesystem.fields.Float'
    assert var_50.min_items is True
    assert var_50.max_items == 10
    assert var_50.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_51 = var_50.items
    var_52 = 'properties'
    var_53 = 'patternProperties'
    var_54 = 'additionalProperties'
    var_55 = 'propertyNames'
    var_56 = 'minProperties'
    var_57 = 'maxProperties'
    var_58 = 'required'
    var_59 = 'object'
    var_60 = 'name'
    var_61 = 'age'
    var_62 = {var_0: var_23}
    var_63 = {var_0: var_15}
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = '^S_'
    var_66 = '^I_'
    var_67 = {var_0: var_23}
    var_68 = {var_0: var_15}
    var_69 = {var_65: var_67, var_66: var_68}
    var_70 = {var_0: var_32}
    var_71 = {var_0: var_23}
    var_72 = [var_60]
    var_73 = 30
    var_74 = {var_60: var_27, var_61: var_73}
    var_75 = {var_0: var_59, var_52: var_64, var_53: var_69, var_54: var_70, var_55: var_71, var_56: var_33, var_57: var_45, var_58: var_72, var_5: var_74}
    var_76 = False
    var_77 = module_5.Definitions()
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_77) == 0
    var_78 = module_0.from_json_schema_type(var_75, var_59, var_76, var_77)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.Object'
    assert var_78.default == {'name': 'test', 'age': 30}
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert f'{type(var_78.properties).__module__}.{type(var_78.properties).__qualname__}' == 'builtins.dict'
    assert len(var_78.properties) == 2
    assert f'{type(var_78.pattern_properties).__module__}.{type(var_78.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_78.pattern_properties) == 2
    assert f'{type(var_78.additional_properties).__module__}.{type(var_78.additional_properties).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_78.property_names).__module__}.{type(var_78.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_78.min_properties is True
    assert var_78.max_properties == 10
    assert var_78.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_79 = var_78.properties[var_60]
    var_80 = var_78.properties[var_61]
    var_81 = var_78.pattern_properties[var_65]
    var_82 = var_78.pattern_properties[var_66]
    var_83 = var_78.additional_properties
    var_84 = var_78.property_names
    var_85 = {}
    var_86 = False
    var_87 = module_5.Definitions()
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_87) == 0
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_85, var_79, var_86, var_87)

@pytest.mark.xfail(strict=True)
def test_case_64():
    var_0 = module_3._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_5.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.fields).__module__}.{type(var_1.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.fields) == 0
    assert var_1.required == []
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_3 = module_0.to_json_schema(var_1)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_4 = None
    var_5 = module_2.Object(properties=var_4, additional_properties=var_4, property_names=var_2, required=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Object'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.properties == {}
    assert var_5.pattern_properties == {}
    assert var_5.additional_properties is None
    assert f'{type(var_5.property_names).__module__}.{type(var_5.property_names).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.min_properties is None
    assert var_5.max_properties is None
    assert var_5.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_6 = module_0.to_json_schema(var_1, var_4)
    var_7 = module_1.Not(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.Not'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.negated is None
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_8 = module_4.purge()
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
    var_9 = module_0.to_json_schema(var_5, var_8)
    var_10 = var_1.get_default_value()
    module_0.to_json_schema(var_4)

@pytest.mark.xfail(strict=True)
def test_case_65():
    var_0 = module_3._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_5.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.fields).__module__}.{type(var_1.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.fields) == 0
    assert var_1.required == []
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
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
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
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
    var_3 = module_2.Object()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = module_2.Object(additional_properties=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert var_4.pattern_properties == {}
    assert f'{type(var_4.additional_properties).__module__}.{type(var_4.additional_properties).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == []
    var_5 = module_0.to_json_schema(var_4, var_0)
    var_6 = module_1.Not(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.Not'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.negated).__module__}.{type(var_6.negated).__qualname__}' == 'typesystem.schemas.Schema'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_7 = module_4.purge()
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
    module_0.to_json_schema(var_7, var_7)

def test_case_66():
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
    var_2 = var_0.get_default_value()
    var_3 = module_2.Array(var_2, min_items=var_1, max_items=var_1)
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
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'required', 'patternProperties', 'minItems', 'multipleOf', 'uniqueItems', 'contains', 'maxProperties', 'dependencies', 'pattern', 'exclusiveMinimum', 'minimum', 'boolean_schema', 'minLength', 'maximum', 'maxLength', 'additionalItems', 'items', 'propertyNames', 'maxItems', 'minProperties', 'additionalProperties', 'properties', 'type', 'exclusiveMaximum'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2