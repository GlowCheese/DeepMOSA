# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import typesystem.composites as module_1
import enum as module_2
import typesystem.fields as module_3
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
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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

def test_case_3():
    var_0 = {}
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_0 = module_3.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Choice'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.choices == []
    assert var_0.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_5():
    pass

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    module_0.any_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
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
    assert var_1 is True

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_0.not_from_json_schema(var_0, var_0)

def test_case_13():
    var_0 = None
    var_1 = 'xyjW*,Jj^vSb".x+'
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_0, var_0, var_1, var_0)

def test_case_14():
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.type_from_json_schema(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = module_0.enum_from_json_schema(var_2, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Choice'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.choices == []
    assert var_4.coerce_types is True
    var_5 = var_3.get_default_value()

def test_case_15():
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
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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

def test_case_16():
    var_0 = {}
    var_1 = module_3.String(**var_0)
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
def test_case_17():
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_4 = module_0.enum_from_json_schema(var_2, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Choice'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.choices == []
    assert var_4.coerce_types is True
    var_5 = module_0.get_valid_types(var_2)
    var_6 = module_0.from_json_schema(var_2, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Choice'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.choices == []
    assert var_6.coerce_types is True
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_1.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = None
    var_2 = module_3.Choice(coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Choice'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.choices == []
    assert var_2.coerce_types is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_1.Not(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.Not'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.negated).__module__}.{type(var_4.negated).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_5 = module_0.to_json_schema(var_4)
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.Not'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.negated).__module__}.{type(var_6.negated).__qualname__}' == 'typesystem.composites.NeverMatch'
    module_4.search(var_1, var_1, var_1)

def test_case_19():
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
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
def test_case_20():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_3 = module_3.Choice(coerce_types=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Choice'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.choices == []
    assert var_3.coerce_types is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_1.AllOf(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.all_of == {'enum': []}
    module_0.to_json_schema(var_5)

@pytest.mark.xfail(strict=True)
def test_case_21():
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
    var_2 = module_1.Not(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.fields.Choice'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    module_0.get_valid_types(var_0)

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
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_3 = module_0.get_valid_types(var_0)
    var_4 = module_1.AllOf(var_0, **var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.all_of).__module__}.{type(var_4.all_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_4.all_of) == 0
    var_5 = module_0.to_json_schema(var_4)
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.all_of == []
    var_7 = module_2._EnumDict()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'enum._EnumDict'
    assert len(var_7) == 0
    module_4.search(var_2, var_2)

def test_case_23():
    var_0 = []
    var_1 = {}
    var_2 = module_5.Definitions(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = 'allOf'
    var_4 = '$ref'
    var_5 = '#/components/schemas/test'
    var_6 = {var_4: var_5}
    var_7 = [var_6]
    var_8 = {var_3: var_7}
    var_9 = module_0.all_of_from_json_schema(var_8, var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.all_of).__module__}.{type(var_9.all_of).__qualname__}' == 'builtins.list'
    assert len(var_9.all_of) == 1
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_10 = var_9.all_of[0].to
    assert var_10 == '#/components/schemas/test'

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_3 = module_3.Choice(coerce_types=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Choice'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.choices == []
    assert var_3.coerce_types is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_4 = module_0.to_json_schema(var_3)
    var_5 = module_1.Not(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.Not'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.negated).__module__}.{type(var_5.negated).__qualname__}' == 'typesystem.fields.Choice'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_6 = module_0.to_json_schema(var_5)
    var_7 = module_0.from_json_schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.Not'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.negated).__module__}.{type(var_7.negated).__qualname__}' == 'typesystem.fields.Choice'
    var_8 = module_0.enum_from_json_schema(var_4, var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Choice'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.choices == []
    assert var_8.coerce_types is True
    module_0.to_json_schema(var_2)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = {}
    var_1 = module_3.Const(var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Const'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.const == {}
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_2 = None
    var_3 = module_3.Choice(coerce_types=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Choice'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.choices == []
    assert var_3.coerce_types is None
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_0.type_from_json_schema(var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert f'{type(var_5.any_of).__module__}.{type(var_5.any_of).__qualname__}' == 'builtins.list'
    assert len(var_5.any_of) == 5
    assert module_3.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_6 = module_0.to_json_schema(var_5)
    var_7 = module_0.to_json_schema(var_1)
    var_8 = module_0.from_json_schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Const'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.const == {}
    module_0.enum_from_json_schema(var_2, var_4)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = module_1.OneOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.one_of is None
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_2 = var_1.serialize(var_0)
    module_0.to_json_schema(var_1, var_0)

def test_case_27():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.Schema(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.to_json_schema(var_2)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = {}
    var_1 = None
    var_2 = module_3.Choice(coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Choice'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.choices == []
    assert var_2.coerce_types is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_3 = module_1.IfThenElse(var_2, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.if_clause).__module__}.{type(var_3.if_clause).__qualname__}' == 'typesystem.fields.Choice'
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.if_clause).__module__}.{type(var_5.if_clause).__qualname__}' == 'typesystem.fields.Choice'
    assert f'{type(var_5.then_clause).__module__}.{type(var_5.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_5.else_clause).__module__}.{type(var_5.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_6 = None
    module_0.to_json_schema(var_6)

def test_case_29():
    var_0 = {}
    var_1 = module_3.String(**var_0)
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
    var_2 = None
    var_3 = {}
    var_4 = module_1.IfThenElse(var_1, var_2, var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_6 = bool('if' in var_5)
    assert var_6 is True

def test_case_30():
    var_0 = []
    var_1 = {}
    var_2 = module_5.Definitions(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = '$ref'
    var_4 = '#/components/schemas/MyType'
    var_5 = {var_3: var_4}
    var_6 = module_0.from_json_schema(var_5, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.to == '#/components/schemas/MyType'
    assert f'{type(var_6.definitions).__module__}.{type(var_6.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_5.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_5.Reference.target).__module__}.{type(module_5.Reference.target).__qualname__}' == 'builtins.property'
    var_7 = var_6.to
    assert var_7 == '#/components/schemas/MyType'

def test_case_31():
    var_0 = []
    var_1 = {}
    var_2 = module_5.Definitions(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = 'allOf'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'enum'
    var_8 = 'a'
    var_9 = 'b'
    var_10 = [var_8, var_9]
    var_11 = {var_7: var_10}
    var_12 = [var_6, var_11]
    var_13 = {var_3: var_12, var_7: var_8}
    var_14 = module_0.all_of_from_json_schema(var_13, var_2)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.all_of).__module__}.{type(var_14.all_of).__qualname__}' == 'builtins.list'
    assert len(var_14.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_15 = var_14.all_of
    var_16 = len(var_15)
    assert var_16 == 2
    with pytest.raises(AttributeError):
        var_17 = var_14.default
    assert var_17 == 'a'

def test_case_32():
    var_0 = 'properties'
    var_1 = 'additionalProperties'
    var_2 = 'name'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = [var_2]
    var_8 = 'integer'
    var_9 = {var_3: var_8}
    var_10 = {var_0: var_6, var_4: var_7, var_1: var_9}
    var_11 = []
    var_12 = module_2._EnumDict()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'enum._EnumDict'
    assert len(var_12) == 0
    var_13 = module_5.Definitions(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13) == 0
    var_14 = 'object'
    var_15 = False
    var_16 = module_0.from_json_schema_type(var_10, var_14, var_15, var_13)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Object'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert f'{type(var_16.properties).__module__}.{type(var_16.properties).__qualname__}' == 'builtins.dict'
    assert len(var_16.properties) == 1
    assert var_16.pattern_properties == {}
    assert f'{type(var_16.additional_properties).__module__}.{type(var_16.additional_properties).__qualname__}' == 'typesystem.fields.Integer'
    assert var_16.property_names is None
    assert var_16.min_properties is None
    assert var_16.max_properties is None
    assert var_16.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_17 = var_16.properties[var_2]
    var_18 = bool('name' in var_16.required)
    var_19 = var_16.additional_properties

def test_case_33():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
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
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 2

def test_case_34():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'string'
    var_3 = 3
    var_4 = 10
    var_5 = {var_0: var_2, var_1: var_3, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_5.Definitions(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = module_0.type_from_json_schema(var_5, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length is None
    assert var_9.min_length == 10
    assert var_9.format is None
    assert var_9.coerce_types is False
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_10 = var_9.min_length
    var_11 = var_9.max_length

def test_case_35():
    var_0 = {}
    var_1 = module_3.String(**var_0)
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
    var_2 = 1
    var_3 = module_3.Array(var_1, min_items=var_2, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.String'
    assert var_3.additional_items is False
    assert var_3.min_items == 1
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_0.type_from_json_schema(var_4, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Array'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.items).__module__}.{type(var_5.items).__qualname__}' == 'typesystem.fields.String'
    assert var_5.additional_items is False
    assert var_5.min_items == 1
    assert var_5.max_items is None
    assert var_5.unique_items is False

def test_case_36():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_3.String(**var_1)
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
    var_3 = {}
    var_4 = module_3.Integer(**var_3)
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
    var_5 = {var_0: var_2, var_0: var_4}
    var_6 = [var_0]
    var_7 = {}
    var_8 = module_3.Object(properties=var_5, required=var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Object'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.properties).__module__}.{type(var_8.properties).__qualname__}' == 'builtins.dict'
    assert len(var_8.properties) == 1
    assert var_8.pattern_properties == {}
    assert var_8.additional_properties is True
    assert var_8.property_names is None
    assert var_8.min_properties is None
    assert var_8.max_properties is None
    assert var_8.required == ['name']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_9 = module_0.to_json_schema(var_8)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_10 = var_9['type']
    assert var_10 == 'object'
    var_11 = var_9['properties']['name']
    var_12 = bool(var_9['properties']['name'] == {'type': 'string', 'minLength': 1})
    assert var_12 is True
    with pytest.raises(KeyError):
        var_13 = bool(var_9['properties']['age'] == {'type': 'integer'})
    assert var_13 is True

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = []
    var_1 = {}
    var_2 = module_5.Definitions(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    module_0.all_of_from_json_schema(var_3, var_2)

def test_case_38():
    var_0 = 'User'
    var_1 = {}
    var_2 = module_3.String(**var_1)
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
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_5.Definitions(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 1
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = bool('components' in var_7)
    assert var_8 is True
    var_9 = bool('schemas' in var_7['components'])
    assert var_9 is True
    var_10 = var_7['components']['schemas']['User']

def test_case_39():
    var_0 = True
    var_1 = 5
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_3.String(min_length=var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is True
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length == 5
    assert var_4.format is None
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_6 = var_5['type']
    var_7 = bool(var_5['type'] == ['string', 'null'])
    assert var_7 is True
    var_8 = var_5['minLength']
    assert var_8 == 5

def test_case_40():
    var_0 = 10
    var_1 = {}
    var_2 = module_3.Integer(exclusive_maximum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Integer'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum is None
    assert var_2.maximum is None
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum == 10
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_0 = 0
    var_1 = -1
    var_2 = {}
    var_3 = module_3.Integer(minimum=var_0, maximum=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Integer'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum == 0
    assert var_3.maximum == -1
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
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_1 == {'type': 'integer', 'minimum': 0, 'maximum': 10})

def test_case_42():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_3.Boolean(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

def test_case_43():
    var_0 = {}
    var_1 = module_3.String(**var_0)
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
    var_2 = {}
    var_3 = module_3.Array(var_1, **var_2)
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
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4 == {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'uniqueItems': True})

def test_case_44():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_3.String(**var_2)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {}
    var_5 = {}
    var_6 = None
    var_7 = []
    var_8 = module_3.Object(properties=var_4, pattern_properties=var_5, additional_properties=var_3, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_7, **var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Object'
    assert var_8.default is None
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is True
    assert var_8.read_only is False
    assert var_8.properties == {}
    assert var_8.pattern_properties == {}
    assert f'{type(var_8.additional_properties).__module__}.{type(var_8.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_8.property_names is None
    assert var_8.min_properties is None
    assert var_8.max_properties is None
    assert var_8.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_9 = module_0.to_json_schema(var_8)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_10 = 'additionalProperties'
    var_11 = var_9[var_10]

def test_case_45():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = {}
    var_6 = module_5.Definitions(*var_4, **var_5)
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
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_8 = var_7.const
    assert var_8 is None

def test_case_46():
    var_0 = None
    var_1 = True
    var_2 = module_3.Field(default=var_0, read_only=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Field.errors == {}
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_2)

def test_case_47():
    var_0 = 'date-time'
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_3.String(format=var_0, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length is None
    assert var_4.min_length is None
    assert var_4.format == 'date-time'
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_6 = var_5['format']
    assert var_6 == 'date-time'

def test_case_48():
    var_0 = 10
    var_1 = {}
    var_2 = module_3.String(max_length=var_0, **var_1)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = bool('maxLength' in var_3)
    assert var_4 is True
    var_5 = var_3['maxLength']
    assert var_5 == 10

def test_case_49():
    var_0 = 'a'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_3.Choice(choices=var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Choice'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.choices == [('a', None), ('b', None)]
    assert var_7.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_8 = module_0.to_json_schema(var_7)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_9 = bool(var_8 == {'enum': ['a', 'b']})
    assert var_9 is True

def test_case_50():
    var_0 = {}
    var_1 = module_3.String(**var_0)
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
    var_2 = 1
    var_3 = module_3.Array(additional_items=var_1, max_items=var_2, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.items is None
    assert f'{type(var_3.additional_items).__module__}.{type(var_3.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_3.min_items is None
    assert var_3.max_items == 1
    assert var_3.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_0.from_json_schema(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False

def test_case_51():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'minItems'
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = {var_3: var_4}
    var_6 = 'boolean'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = 'string'
    var_10 = {var_3: var_9}
    var_11 = 1
    var_12 = {var_0: var_8, var_1: var_10, var_2: var_11}
    var_13 = []
    var_14 = {}
    var_15 = module_5.Definitions(*var_13, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_15) == 0
    var_16 = 'array'
    var_17 = False
    var_18 = module_0.from_json_schema_type(var_12, var_16, var_17, var_15)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Array'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert f'{type(var_18.items).__module__}.{type(var_18.items).__qualname__}' == 'builtins.list'
    assert len(var_18.items) == 2
    assert f'{type(var_18.additional_items).__module__}.{type(var_18.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_18.min_items == 1
    assert var_18.max_items is None
    assert var_18.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_19 = var_18.items
    var_20 = var_18.items[var_17]
    var_21 = var_18.items[var_11]
    var_22 = var_18.additional_items
    var_23 = var_18.min_items
    assert var_23 == 1

def test_case_52():
    var_0 = 'properties'
    var_1 = 'patternProperties'
    var_2 = 'additionalProperties'
    var_3 = 'minProperties'
    var_4 = 'age'
    var_5 = 'type'
    var_6 = 'integer'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = '^id_'
    var_10 = 'string'
    var_11 = {var_5: var_10}
    var_12 = {var_9: var_11}
    var_13 = False
    var_14 = {var_5: var_10}
    var_15 = {var_0: var_8, var_1: var_12, var_2: var_13, var_2: var_14, var_3: var_13}
    var_16 = []
    var_17 = {}
    var_18 = module_5.Definitions(*var_16, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_18) == 0
    var_19 = 'object'
    var_20 = module_0.from_json_schema_type(var_15, var_19, var_13, var_18)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Object'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.properties).__module__}.{type(var_20.properties).__qualname__}' == 'builtins.dict'
    assert len(var_20.properties) == 1
    assert f'{type(var_20.pattern_properties).__module__}.{type(var_20.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_20.pattern_properties) == 1
    assert f'{type(var_20.additional_properties).__module__}.{type(var_20.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_20.property_names is None
    assert var_20.min_properties is False
    assert var_20.max_properties is None
    assert var_20.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_21 = var_20.properties[var_4]
    var_22 = bool('^id_' in var_20.pattern_properties)
    assert var_22 is True
    var_23 = var_20.additional_properties
    var_24 = var_20.property_names
    var_25 = var_20.min_properties

def test_case_53():
    var_0 = 'properties'
    var_1 = 'patternProperties'
    var_2 = 'additionalProperties'
    var_3 = 'propertyNames'
    var_4 = 'minProperties'
    var_5 = 'age'
    var_6 = 'type'
    var_7 = 'integer'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = '^id_'
    var_11 = 'string'
    var_12 = {var_6: var_11}
    var_13 = {var_10: var_12}
    var_14 = False
    var_15 = {var_6: var_11}
    var_16 = 1
    var_17 = {var_0: var_9, var_1: var_13, var_2: var_14, var_3: var_15, var_4: var_16}
    var_18 = []
    var_19 = {}
    var_20 = module_5.Definitions(*var_18, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_20) == 0
    var_21 = 'object'
    var_22 = module_0.from_json_schema_type(var_17, var_21, var_14, var_20)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Object'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.properties).__module__}.{type(var_22.properties).__qualname__}' == 'builtins.dict'
    assert len(var_22.properties) == 1
    assert f'{type(var_22.pattern_properties).__module__}.{type(var_22.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_22.pattern_properties) == 1
    assert var_22.additional_properties is False
    assert f'{type(var_22.property_names).__module__}.{type(var_22.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_22.min_properties == 1
    assert var_22.max_properties is None
    assert var_22.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
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
    var_23 = var_22.properties[var_5]
    var_24 = bool('^id_' in var_22.pattern_properties)
    assert var_24 is True
    var_25 = var_22.additional_properties
    assert var_25 is False
    var_26 = var_22.property_names
    var_27 = var_22.min_properties
    assert var_27 == 1

@pytest.mark.xfail(strict=True)
def test_case_54():
    var_0 = []
    var_1 = {}
    var_2 = module_5.Definitions(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = 'allOf'
    var_4 = '$ref'
    var_5 = module_0.from_json_schema(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_6 = {var_4: var_4}
    var_7 = module_0.to_json_schema(var_2)
    var_8 = [var_6]
    var_9 = {var_3: var_8}
    module_0.all_of_from_json_schema(var_9, var_2)

def test_case_55():
    var_0 = []
    var_1 = {}
    var_2 = module_5.Definitions(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = 'if'
    var_4 = 'default'
    var_5 = 'type'
    var_6 = 'number'
    var_7 = {var_5: var_6}
    var_8 = None
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = module_0.if_then_else_from_json_schema(var_9, var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_10.default is None
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.if_clause).__module__}.{type(var_10.if_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_10.then_clause).__module__}.{type(var_10.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_10.else_clause).__module__}.{type(var_10.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_11 = var_10.if_clause.numeric_type
    var_12 = var_10.then_clause
    var_13 = var_10.else_clause

def test_case_56():
    var_0 = 'components'
    var_1 = 'schemas'
    var_2 = 'Item'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Any'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_10 = bool(True)
    assert var_10 is True

def test_case_57():
    var_0 = 0
    var_1 = 10
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_3.Integer(minimum=var_0, maximum=var_1, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Integer'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert var_5.minimum == 0
    assert var_5.maximum == 10
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of is None
    assert var_5.precision is None
    assert var_5.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minProperties', 'maxLength', 'maxItems', 'additionalProperties', 'exclusiveMinimum', 'propertyNames', 'multipleOf', 'boolean_schema', 'properties', 'contains', 'additionalItems', 'minLength', 'maximum', 'items', 'exclusiveMaximum', 'minimum', 'required', 'patternProperties', 'maxProperties', 'pattern', 'uniqueItems', 'type', 'minItems', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = bool(var_6 == {'type': ['integer', 'null'], 'minimum': 0, 'maximum': 10})