# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import re as module_1
import typesystem.schemas as module_2
import typesystem.fields as module_3
import enum as module_4
import typesystem.composites as module_5

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.from_json_schema(var_0)

def test_case_1():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_0 = module_1.purge()
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    module_0.to_json_schema(var_0)
    assert var_1 is True

def test_case_4():
    pass

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = var_1.get_default_value()
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = []
    var_1 = {}
    var_2 = module_2.Definitions(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_5 = module_0.from_json_schema(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    module_0.all_of_from_json_schema(var_4, var_4)

def test_case_7():
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_2 = module_4._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_3 = module_0.to_json_schema(var_1)
    var_4 = module_0.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

def test_case_9():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}

def test_case_10():
    var_0 = {}
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_12():
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
def test_case_13():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

def test_case_14():
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
    var_2 = module_5.Not(var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.Not'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.negated).__module__}.{type(var_4.negated).__qualname__}' == 'typesystem.fields.String'

def test_case_15():
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_0 = module_4._EnumDict()
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_4 = module_0.from_json_schema(var_0, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False

def test_case_17():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_3.String(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.default == 'default_value'
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_2 = module_5.Not(var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_0 = {}
    var_1 = module_3.Array(**var_0)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_0 = None
    var_1 = module_3.Integer(exclusive_minimum=var_0)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_2 = module_0.to_json_schema(var_1, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_0 = 0.0
    var_1 = {}
    var_2 = module_3.Float(minimum=var_0, maximum=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Float'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.minimum == pytest.approx(0.0, abs=0.01, rel=0.01)
    assert var_2.maximum == pytest.approx(0.0, abs=0.01, rel=0.01)
    assert var_2.exclusive_minimum is None
    assert var_2.exclusive_maximum is None
    assert var_2.multiple_of is None
    assert var_2.precision is None
    assert var_2.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = bool(var_3 == {'type': 'number', 'minimum': 0.0, 'maximum': 100.0})

def test_case_22():
    var_0 = {}
    var_1 = module_3.Boolean(**var_0)
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = bool(var_2 == {'type': 'boolean'})
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    module_0.to_json_schema(var_3)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = {}
    var_1 = module_5.AllOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.all_of == {}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    module_0.to_json_schema(var_3)

def test_case_25():
    var_0 = module_4._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_5.NeverMatch()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = module_3.Boolean(**var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_3 = module_4._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    var_4 = module_5.IfThenElse(var_1, var_2, var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.if_clause).__module__}.{type(var_6.if_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_6.then_clause).__module__}.{type(var_6.then_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_6.else_clause).__module__}.{type(var_6.else_clause).__qualname__}' == 'typesystem.fields.Boolean'

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = {}
    var_1 = module_3.Integer(**var_0)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_5.AllOf(var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.all_of).__module__}.{type(var_4.all_of).__qualname__}' == 'builtins.list'
    assert len(var_4.all_of) == 2
    module_0.to_json_schema(var_4)

def test_case_27():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
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
    assert var_3.format is None
    assert var_3.coerce_types is False
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = {}
    var_1 = None
    var_2 = module_2.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == {}
    assert var_2.definitions is None
    assert module_2.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_2.Reference.target).__module__}.{type(module_2.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    module_0.to_json_schema(var_2)

def test_case_29():
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
    var_3 = {var_0: var_2}
    var_4 = '^S_'
    var_5 = {}
    var_6 = module_3.String(**var_5)
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
    var_8 = True
    var_9 = 19
    var_10 = -12
    var_11 = []
    var_12 = {}
    var_13 = module_3.Object(properties=var_3, pattern_properties=var_7, additional_properties=var_8, property_names=var_6, min_properties=var_10, max_properties=var_9, required=var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Object'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.properties).__module__}.{type(var_13.properties).__qualname__}' == 'builtins.dict'
    assert len(var_13.properties) == 1
    assert f'{type(var_13.pattern_properties).__module__}.{type(var_13.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_13.pattern_properties) == 1
    assert var_13.additional_properties is True
    assert f'{type(var_13.property_names).__module__}.{type(var_13.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_13.min_properties == -12
    assert var_13.max_properties == 19
    assert var_13.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_14 = module_0.to_json_schema(var_13)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_0 = {}
    var_1 = '^S_'
    var_2 = {}
    var_3 = module_3.String(**var_2)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_1: var_3}
    var_5 = True
    var_6 = 19
    var_7 = []
    var_8 = {}
    var_9 = module_3.Object(properties=var_0, pattern_properties=var_4, additional_properties=var_5, property_names=var_3, min_properties=var_6, max_properties=var_6, required=var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Object'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.properties == {}
    assert f'{type(var_9.pattern_properties).__module__}.{type(var_9.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_9.pattern_properties) == 1
    assert var_9.additional_properties is True
    assert f'{type(var_9.property_names).__module__}.{type(var_9.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_9.min_properties == 19
    assert var_9.max_properties == 19
    assert var_9.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_10 = module_0.to_json_schema(var_9)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = [var_2, var_2]
    var_4 = {}
    var_5 = module_3.Choice(choices=var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Choice'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.choices == [('a', 'Option A'), ('a', 'Option A')]
    assert var_5.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = bool(var_6 == {'enum': ['a', 'b']})

def test_case_32():
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
    var_3 = module_3.Integer(**var_2)
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
    var_5 = {}
    var_6 = module_5.OneOf(var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.one_of).__module__}.{type(var_6.one_of).__qualname__}' == 'builtins.list'
    assert len(var_6.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = bool(var_7 == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]})

def test_case_33():
    var_0 = []
    var_1 = {}
    var_2 = module_2.Definitions(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_34():
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
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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

def test_case_35():
    var_0 = -15
    var_1 = 1635
    var_2 = True
    var_3 = 2
    var_4 = {}
    var_5 = module_3.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_2, multiple_of=var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Integer'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum == -15
    assert var_5.maximum == 1635
    assert var_5.exclusive_minimum is True
    assert var_5.exclusive_maximum is True
    assert var_5.multiple_of == 2
    assert var_5.precision is None
    assert var_5.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_36():
    var_0 = 10
    var_1 = {}
    var_2 = module_3.Array(max_items=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.items is None
    assert var_2.additional_items is False
    assert var_2.min_items is None
    assert var_2.max_items == 10
    assert var_2.unique_items is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = bool(var_0 == {'type': 'array', 'maxItems': 10})

def test_case_37():
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
    var_2 = module_3.Array(var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.String'
    assert var_2.additional_items is False
    assert var_2.min_items is None
    assert var_2.max_items is None
    assert var_2.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = bool(var_3 == {'type': 'array', 'items': {'type': 'string'}})

def test_case_38():
    var_0 = 50
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
    var_3 = True
    var_4 = module_3.Array(var_2, var_3, var_0, var_3, unique_items=var_3, **var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.items).__module__}.{type(var_4.items).__qualname__}' == 'typesystem.fields.String'
    assert var_4.additional_items is True
    assert var_4.min_items == 50
    assert var_4.max_items is True
    assert var_4.unique_items is True
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_0 = {}
    var_1 = {}
    var_2 = {}
    var_3 = module_3.String(**var_2)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = True
    var_5 = 19
    var_6 = []
    var_7 = module_3.Object(properties=var_1, pattern_properties=var_0, additional_properties=var_4, property_names=var_3, min_properties=var_5, max_properties=var_5, required=var_6, **var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is True
    assert f'{type(var_7.property_names).__module__}.{type(var_7.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_7.min_properties == 19
    assert var_7.max_properties == 19
    assert var_7.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = module_0.to_json_schema(var_7)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_3 = '^S_'
    var_4 = {}
    var_5 = module_3.String(**var_4)
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
    var_6 = {var_3: var_5}
    var_7 = False
    var_8 = 10
    var_9 = [var_0]
    var_10 = {}
    var_11 = module_3.Object(properties=var_4, pattern_properties=var_6, additional_properties=var_7, property_names=var_2, min_properties=var_7, max_properties=var_8, required=var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Object'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.properties == {}
    assert f'{type(var_11.pattern_properties).__module__}.{type(var_11.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_11.pattern_properties) == 1
    assert var_11.additional_properties is False
    assert f'{type(var_11.property_names).__module__}.{type(var_11.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_11.min_properties is False
    assert var_11.max_properties == 10
    assert var_11.required == ['name']
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_12 = module_0.to_json_schema(var_11)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_13 = bool(var_0 == {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'patternProperties': {'^S_': {'type': 'string'}}, 'additionalProperties': False, 'propertyNames': {'type': 'string'}, 'maxProperties': 10, 'minProperties': 1, 'required': ['name']})

def test_case_41():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_3.Object(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is True
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4 == {'type': ['object', 'null']})

def test_case_42():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_3.Const(var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Const'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.const == 'fixed_value'
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = bool(var_3 == {'const': 'fixed_value'})
    assert var_4 is True

def test_case_43():
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_2 = module_3.Boolean(**var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.coerce_types is True
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_3.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_3.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_3 = module_4._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    var_4 = module_5.IfThenElse(var_1, var_2, var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_5 = module_0.to_json_schema(var_4)

def test_case_44():
    var_0 = '$ref'
    var_1 = '#/components/schemas/test'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_2.Definitions(*var_3, **var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 1
    var_5 = module_0.from_json_schema(var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.to == '#/components/schemas/test'
    assert f'{type(var_5.definitions).__module__}.{type(var_5.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5.definitions) == 1
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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

def test_case_45():
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = len(var_1)

def test_case_46():
    var_0 = 'allOf'
    var_1 = 'tpe'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'enum'
    var_5 = 'a'
    var_6 = ''
    var_7 = [var_5, var_6]
    var_8 = {var_4: var_7}
    var_9 = [var_3, var_8]
    var_10 = {var_0: var_9}
    var_11 = module_0.from_json_schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.all_of).__module__}.{type(var_11.all_of).__qualname__}' == 'builtins.list'
    assert len(var_11.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    with pytest.raises(TypeError):
        var_13 = len(var_11)
    assert var_13 == 2

def test_case_47():
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 2

def test_case_48():
    var_0 = {}
    var_1 = module_3.Boolean(**var_0)
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
    var_2 = module_4._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = module_5.IfThenElse(var_1, var_1, var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.if_clause).__module__}.{type(var_3.if_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    assert f'{type(var_5.if_clause).__module__}.{type(var_5.if_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_5.then_clause).__module__}.{type(var_5.then_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_5.else_clause).__module__}.{type(var_5.else_clause).__qualname__}' == 'typesystem.fields.Boolean'

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = 'if'
    var_1 = 'theD'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = {var_3: var_3}
    var_9 = {var_0: var_5, var_1: var_7, var_2: var_8}
    module_0.from_json_schema(var_9)

def test_case_50():
    var_0 = 'const'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Const'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.const == 'test'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_4 = var_3.const
    assert var_4 == 'test'

def test_case_51():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_3.Decimal(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Decimal'
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4['type'] == ['number', 'null'])
    assert var_5 is True

def test_case_52():
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
    assert module_3.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    with pytest.raises(TypeError):
        var_5 = var_0['type']
    assert var_5 == 'boolean'

def test_case_53():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_3.String(**var_0)
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
    var_3 = False
    var_4 = module_3.Array(var_2, var_3, var_3, var_3, unique_items=var_3, **var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.items).__module__}.{type(var_4.items).__qualname__}' == 'typesystem.fields.String'
    assert var_4.additional_items is False
    assert var_4.min_items is False
    assert var_4.max_items is False
    assert var_4.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_5 = module_0.to_json_schema(var_4)
    var_6 = bool(var_5 == {'type': 'array', 'minItems': 1, 'maxItems': 10, 'items': {'type': 'string'}, 'additionalItems': False, 'uniqueItems': True})
    var_7 = module_0.from_json_schema(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Array'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.items).__module__}.{type(var_7.items).__qualname__}' == 'typesystem.fields.String'
    assert var_7.additional_items is False
    assert var_7.min_items is False
    assert var_7.max_items is False
    assert var_7.unique_items is False

def test_case_54():
    var_0 = module_4._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = 1651
    var_2 = True
    var_3 = module_3.String(max_length=var_1, coerce_types=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length == 1651
    assert var_3.min_length is None
    assert var_3.format is None
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = None
    var_5 = module_5.Not(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.Not'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.negated is None
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_6 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = module_0.from_json_schema(var_0, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Any'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False

def test_case_55():
    var_0 = 5
    var_1 = {}
    var_2 = module_3.String(min_length=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
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
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_3 = module_3.Integer(**var_2)
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
    var_5 = {}
    var_6 = module_3.Array(var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.items).__module__}.{type(var_6.items).__qualname__}' == 'builtins.list'
    assert len(var_6.items) == 2
    assert var_6.additional_items is False
    assert var_6.min_items == 2
    assert var_6.max_items == 2
    assert var_6.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = bool(var_7 == {'type': 'array', 'items': [{'type': 'string'}, {'type': 'integer'}]})

def test_case_57():
    var_0 = True
    var_1 = 'name'
    var_2 = {}
    var_3 = module_3.String(**var_2)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_1: var_3}
    var_5 = 'allow_null'
    var_6 = {var_5: var_0}
    var_7 = module_2.Schema(var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.default is None
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is True
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 1
    assert var_7.required == ['name']
    assert module_2.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = module_0.to_json_schema(var_7)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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

def test_case_58():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_3.String(format=var_0, **var_1)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = bool(var_3 == {'type': 'string', 'format': 'email'})

def test_case_59():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_3.Array(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert var_3.items is None
    assert var_3.additional_items is False
    assert var_3.min_items is None
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4 == {'type': ['array', 'null']})

def test_case_60():
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
    var_3 = module_3.Array(additional_items=var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.items is None
    assert f'{type(var_3.additional_items).__module__}.{type(var_3.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_3.min_items is None
    assert var_3.max_items is None
    assert var_3.unique_items is False
    assert module_3.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4 == {'type': 'array', 'additionalItems': {'type': 'string'}})

def test_case_61():
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
    var_3 = module_3.Object(additional_properties=var_1, **var_2)
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
    assert var_3.required == []
    assert module_3.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = bool(var_4 == {'type': 'object', 'additionalProperties': {'type': 'string'}})

def test_case_62():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = 42
    var_9 = {var_0: var_5, var_1: var_7, var_2: var_8}
    var_10 = []
    var_11 = {}
    var_12 = module_2.Definitions(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_12) == 0
    var_13 = module_0.if_then_else_from_json_schema(var_9, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_13.default == 42
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.if_clause).__module__}.{type(var_13.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_13.then_clause).__module__}.{type(var_13.then_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_13.else_clause).__module__}.{type(var_13.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_17 = var_13.default
    assert var_17 == 42

def test_case_63():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'additionalProperties'
    var_3 = 'minProperties'
    var_4 = 'maxProperties'
    var_5 = 'required'
    var_6 = 'default'
    var_7 = 'object'
    var_8 = 'name'
    var_9 = 'string'
    var_10 = {var_0: var_9}
    var_11 = {var_8: var_10}
    var_12 = False
    var_13 = 1
    var_14 = 5
    var_15 = [var_8]
    var_16 = 'test'
    var_17 = {var_8: var_16}
    var_18 = {var_0: var_7, var_1: var_11, var_2: var_12, var_3: var_13, var_4: var_14, var_5: var_15, var_6: var_17}
    var_19 = []
    var_20 = {}
    var_21 = module_2.Definitions(*var_19, **var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_21) == 0
    var_22 = module_0.from_json_schema_type(var_18, var_7, var_12, var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Object'
    assert var_22.default == {'name': 'test'}
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.properties).__module__}.{type(var_22.properties).__qualname__}' == 'builtins.dict'
    assert len(var_22.properties) == 1
    assert var_22.pattern_properties == {}
    assert var_22.additional_properties is False
    assert var_22.property_names is None
    assert var_22.min_properties == 1
    assert var_22.max_properties == 5
    assert var_22.required == ['name']
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_23 = var_22.properties[var_8]
    var_24 = var_22.additional_properties
    assert var_24 is False
    var_25 = var_22.min_properties
    assert var_25 == 1
    var_26 = var_22.max_properties
    assert var_26 == 5
    var_27 = var_22.required
    var_28 = bool(var_22.required == ['name'])
    assert var_28 is True
    var_29 = bool(var_22.default == {'name': 'test'})
    assert var_29 is True
    var_30 = var_22.allow_null
    assert var_30 is False

def test_case_64():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'default'
    var_4 = 'integer'
    var_5 = 0
    var_6 = 100
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_5}
    var_8 = False
    var_9 = []
    var_10 = {}
    var_11 = module_2.Definitions(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_11) == 0
    var_12 = module_0.from_json_schema_type(var_7, var_4, var_8, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Integer'
    assert var_12.default == 0
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
    assert var_12.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_13 = var_12.minimum
    assert var_13 == 0
    var_14 = var_12.maximum
    assert var_14 == 100
    var_15 = var_12.default
    var_16 = var_12.allow_null
    assert var_16 is False
    var_17 = var_12.coerce_types
    assert var_17 is False

def test_case_65():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'format'
    var_4 = 'pattern'
    var_5 = 'default'
    var_6 = 'string'
    var_7 = 5
    var_8 = 10
    var_9 = 'email'
    var_10 = '^[a-z]+$'
    var_11 = 'test'
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = False
    var_14 = []
    var_15 = {}
    var_16 = module_2.Definitions(*var_14, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_16) == 0
    var_17 = module_0.from_json_schema_type(var_12, var_6, var_13, var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.String'
    assert var_17.default == 'test'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.allow_blank is False
    assert var_17.trim_whitespace is True
    assert var_17.max_length == 10
    assert var_17.min_length == 5
    assert var_17.format == 'email'
    assert var_17.coerce_types is False
    assert var_17.pattern == '^[a-z]+$'
    assert f'{type(var_17.pattern_regex).__module__}.{type(var_17.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_18 = var_17.min_length
    assert var_18 == 5
    var_19 = var_17.format
    assert var_19 == 'email'
    var_20 = var_17.pattern
    assert var_20 == '^[a-z]+$'
    var_21 = var_17.default
    assert var_21 == 'test'
    var_22 = var_17.allow_null
    assert var_22 is False
    var_23 = var_17.allow_blank
    assert var_23 is False

def test_case_66():
    var_0 = 'type'
    var_1 = 'additionalProperties'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_2.Definitions(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = False
    var_10 = module_0.from_json_schema_type(var_5, var_2, var_9, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Object'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.properties == {}
    assert var_10.pattern_properties == {}
    assert f'{type(var_10.additional_properties).__module__}.{type(var_10.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_10.property_names is None
    assert var_10.min_properties is None
    assert var_10.max_properties is None
    assert var_10.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_11 = var_10.additional_properties

def test_case_67():
    var_0 = 'type'
    var_1 = 'patternProperties'
    var_2 = 'object'
    var_3 = '^S_'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = []
    var_9 = {}
    var_10 = module_2.Definitions(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_10) == 0
    var_11 = False
    var_12 = module_0.from_json_schema_type(var_7, var_2, var_11, var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Object'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.properties == {}
    assert f'{type(var_12.pattern_properties).__module__}.{type(var_12.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_12.pattern_properties) == 1
    assert var_12.additional_properties is None
    assert var_12.property_names is None
    assert var_12.min_properties is None
    assert var_12.max_properties is None
    assert var_12.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_13 = var_12.pattern_properties[var_3]

def test_case_68():
    var_0 = 'type'
    var_1 = 'propertyNames'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_2.Definitions(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = False
    var_10 = module_0.from_json_schema_type(var_5, var_2, var_9, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Object'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.properties == {}
    assert var_10.pattern_properties == {}
    assert var_10.additional_properties is None
    assert f'{type(var_10.property_names).__module__}.{type(var_10.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_10.min_properties is None
    assert var_10.max_properties is None
    assert var_10.required == []
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_11 = var_10.property_names

def test_case_69():
    var_0 = 'type'
    var_1 = 'additionalItems'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_2.Definitions(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = False
    var_10 = module_0.from_json_schema_type(var_5, var_2, var_9, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Array'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.items is None
    assert f'{type(var_10.additional_items).__module__}.{type(var_10.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_10.min_items == 0
    assert var_10.max_items is None
    assert var_10.unique_items is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
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
    var_11 = var_10.additional_items

def test_case_70():
    var_0 = 'string'
    var_1 = 'integer'
    var_2 = {}
    var_3 = module_3.String(**var_2)
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
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {}
    var_5 = module_3.Integer(**var_4)
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
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_2.Definitions(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_9) == 2
    var_10 = module_0.to_json_schema(var_9)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minLength', 'contains', 'patternProperties', 'maxLength', 'properties', 'multipleOf', 'maximum', 'type', 'items', 'additionalItems', 'required', 'boolean_schema', 'uniqueItems', 'exclusiveMinimum', 'propertyNames', 'maxProperties', 'additionalProperties', 'dependencies', 'minimum', 'minProperties', 'minItems', 'maxItems', 'exclusiveMaximum', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_11 = bool(var_10 == {'components': {'schemas': {'string': {'type': 'string'}, 'integer': {'type': 'integer'}}}})

def test_case_71():
    var_0 = []
    var_1 = {}
    var_2 = module_2.Definitions(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = None
    var_4 = module_3.Field()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_3.NO_DEFAULT).__module__}.{type(module_3.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_3.FORMATS).__module__}.{type(module_3.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_3.FORMATS) == 7
    assert module_3.Field.errors == {}
    var_5 = module_5.IfThenElse(var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.if_clause).__module__}.{type(var_5.if_clause).__qualname__}' == 'typesystem.fields.Field'
    assert f'{type(var_5.then_clause).__module__}.{type(var_5.then_clause).__qualname__}' == 'typesystem.fields.Field'
    assert f'{type(var_5.else_clause).__module__}.{type(var_5.else_clause).__qualname__}' == 'typesystem.fields.Any'
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_5, var_3)

def test_case_72():
    var_0 = '$ref'
    var_1 = 'invalid_ref'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_2.Definitions(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 0
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_2, var_5)