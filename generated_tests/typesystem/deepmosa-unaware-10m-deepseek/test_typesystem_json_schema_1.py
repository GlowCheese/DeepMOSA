# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import enum as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import re as module_4
import typesystem.composites as module_5

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.from_json_schema(var_0)

def test_case_1():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_0 = module_1._EnumDict()
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
def test_case_4():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.all_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = {}
    module_0.any_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.to_json_schema(var_0)

def test_case_9():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = module_0.to_json_schema(var_3, var_2)
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
    var_6 = module_0.from_json_schema(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Union'
    assert var_6.default is None
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.any_of).__module__}.{type(var_6.any_of).__qualname__}' == 'builtins.list'
    assert len(var_6.any_of) == 5
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_5, var_5, var_5, var_5)

def test_case_10():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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

def test_case_11():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_0 = module_5.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = module_0.get_standard_properties(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    module_0.const_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    module_0.not_from_json_schema(var_0, var_0)

def test_case_15():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_0 = module_1._EnumDict()
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7

def test_case_17():
    var_0 = module_1._EnumDict()
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_3 = module_5.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_0.get_valid_types(var_0)
    var_5 = module_0.to_json_schema(var_3, var_1)
    var_6 = module_0.from_json_schema(var_5, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.Not'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.negated).__module__}.{type(var_6.negated).__qualname__}' == 'typesystem.fields.Union'
    var_7 = module_0.from_json_schema(var_5, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.Not'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.negated).__module__}.{type(var_7.negated).__qualname__}' == 'typesystem.fields.Union'

def test_case_18():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_3.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.fields).__module__}.{type(var_1.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.fields) == 0
    assert var_1.required == []
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is None
    assert var_3.property_names is None
    assert var_3.min_properties is None
    assert var_3.max_properties is None
    assert var_3.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}

def test_case_19():
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_0 = module_5.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = 1
    var_3 = '^[a-wVz]+$'
    var_4 = module_2.String(max_length=var_1, min_length=var_2, pattern=var_3, format=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length is False
    assert var_4.min_length == 1
    assert var_4.format == '^[a-wVz]+$'
    assert var_4.coerce_types is True
    assert var_4.pattern == '^[a-wVz]+$'
    assert f'{type(var_4.pattern_regex).__module__}.{type(var_4.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_0.to_json_schema(var_4)
    var_6 = module_2.Integer()
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

def test_case_21():
    var_0 = module_1._EnumDict()
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_3 = module_0.from_json_schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = {}
    var_1 = None
    var_2 = module_3.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == {}
    assert var_2.definitions is None
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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

def test_case_23():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.OneOf(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.one_of).__module__}.{type(var_2.one_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.one_of) == 0
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_3 = module_5.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.composites.OneOf'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_0.get_valid_types(var_0)
    var_5 = module_0.to_json_schema(var_2)
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.one_of == []

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4 = module_2.Const(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Const'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.const).__module__}.{type(var_4.const).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_5 = var_4.get_default_value()
    var_6 = module_0.to_json_schema(var_4, var_2)
    module_0.to_json_schema(var_2)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_1._EnumDict()
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_3 = module_2.Const(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Const'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.const).__module__}.{type(var_3.const).__qualname__}' == 'enum._EnumDict'
    assert len(var_3.const) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_4 = module_0.to_json_schema(var_3, var_1)
    var_5 = module_0.from_json_schema(var_4, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Const'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.const).__module__}.{type(var_5.const).__qualname__}' == 'enum._EnumDict'
    assert len(var_5.const) == 0
    var_6 = module_1._EnumDict()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'enum._EnumDict'
    assert len(var_6) == 0
    module_0.from_json_schema(var_1, var_1)

def test_case_26():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_2.Choice(coerce_types=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Choice'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.choices == []
    assert f'{type(var_1.coerce_types).__module__}.{type(var_1.coerce_types).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.coerce_types) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_2 = module_5.IfThenElse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.fields.Choice'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_0 = 'ku6X{/'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_3.Definitions(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 1
    module_0.to_json_schema(var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_3 = module_3.Schema(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_3.fields) == 0
    assert var_3.required == []
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_4.purge()
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
    var_5 = module_0.get_valid_types(var_0)
    var_6 = module_5.Not(var_3, **var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.Not'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.negated).__module__}.{type(var_6.negated).__qualname__}' == 'typesystem.schemas.Schema'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_7 = module_0.get_valid_types(var_0)
    var_8 = module_2.Array(exact_items=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Array'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.items is None
    assert var_8.additional_items is False
    assert var_8.min_items is None
    assert var_8.max_items is None
    assert var_8.unique_items is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = module_0.from_json_schema(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Array'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.items is None
    assert var_10.additional_items is False
    assert var_10.min_items == 0
    assert var_10.max_items is None
    assert var_10.unique_items is False
    module_0.type_from_json_schema(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_3 = module_5.IfThenElse(var_1, else_clause=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.if_clause).__module__}.{type(var_3.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_4 = module_5.Not(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.Not'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.negated).__module__}.{type(var_4.negated).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_5 = module_0.get_valid_types(var_0)
    var_6 = module_0.to_json_schema(var_4, var_2)
    var_7 = module_0.from_json_schema(var_6, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.Not'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.negated).__module__}.{type(var_7.negated).__qualname__}' == 'typesystem.composites.IfThenElse'
    module_0.type_from_json_schema(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = module_3.Definitions()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = module_0.from_json_schema_type(var_6, var_3, var_7, var_8)
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_10 = 'integer'
    var_11 = True
    var_12 = module_3.Definitions()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_12) == 0
    var_13 = module_0.from_json_schema_type(var_6, var_10, var_11, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.default is None
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is True
    assert var_13.read_only is False
    assert var_13.minimum == 0
    assert var_13.maximum == 10
    assert var_13.exclusive_minimum is None
    assert var_13.exclusive_maximum is None
    assert var_13.multiple_of is None
    assert var_13.precision is None
    assert var_13.coerce_types is False
    var_14 = 'minLength'
    var_15 = 'pattern'
    var_16 = 'string'
    var_17 = 3
    var_18 = '^[A-Z]+$'
    var_19 = {var_0: var_16, var_14: var_17, var_15: var_18}
    var_20 = False
    var_21 = module_3.Definitions()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_21) == 0
    var_22 = module_0.from_json_schema_type(var_19, var_16, var_20, var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.String'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.allow_blank is False
    assert var_22.trim_whitespace is True
    assert var_22.max_length is None
    assert var_22.min_length == 3
    assert var_22.format is None
    assert var_22.coerce_types is False
    assert var_22.pattern == '^[A-Z]+$'
    assert f'{type(var_22.pattern_regex).__module__}.{type(var_22.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_23 = 'default'
    var_24 = 'boolean'
    var_25 = {var_0: var_24, var_23: var_11}
    var_26 = module_3.Definitions()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_26) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_27 = module_0.from_json_schema_type(var_25, var_24, var_11, var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_27.default is True
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is True
    assert var_27.read_only is False
    assert var_27.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_28 = module_3.Definitions()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_28) == 0
    var_29 = 'items'
    var_30 = 'array'
    var_31 = {var_0: var_16}
    var_32 = {var_0: var_3}
    var_33 = [var_31, var_32]
    var_34 = {var_0: var_30, var_29: var_33}
    var_35 = False
    var_36 = module_0.from_json_schema_type(var_34, var_30, var_35, var_28)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Array'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert f'{type(var_36.items).__module__}.{type(var_36.items).__qualname__}' == 'builtins.list'
    assert len(var_36.items) == 2
    assert var_36.additional_items is True
    assert var_36.min_items == 0
    assert var_36.max_items is None
    assert var_36.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_37 = var_36.items
    var_38 = var_36.items
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = var_36.items[var_35]
    var_41 = {var_0: var_10}
    var_42 = {var_0: var_30, var_29: var_41}
    var_43 = False
    var_44 = module_0.from_json_schema_type(var_42, var_30, var_43, var_28)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Array'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert f'{type(var_44.items).__module__}.{type(var_44.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_44.additional_items is True
    assert var_44.min_items == 0
    assert var_44.max_items is None
    assert var_44.unique_items is False
    var_45 = var_44.items
    var_46 = 'properties'
    var_47 = 'required'
    var_48 = 'object'
    var_49 = 'name'
    var_50 = 'age'
    var_51 = {var_0: var_16}
    var_52 = {var_0: var_10}
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = [var_49]
    var_55 = {var_0: var_48, var_46: var_53, var_47: var_54}
    var_56 = False
    var_57 = module_0.from_json_schema_type(var_55, var_48, var_56, var_28)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.Object'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert f'{type(var_57.properties).__module__}.{type(var_57.properties).__qualname__}' == 'builtins.dict'
    assert len(var_57.properties) == 2
    assert var_57.pattern_properties == {}
    assert var_57.additional_properties is None
    assert var_57.property_names is None
    assert var_57.min_properties is None
    assert var_57.max_properties is None
    assert var_57.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_58 = var_57.properties[var_49]
    var_59 = var_57.properties[var_50]
    var_60 = 'patternProperties'
    var_61 = '^S_'
    var_62 = {var_0: var_16}
    var_63 = {var_61: var_62}
    var_64 = {var_0: var_48, var_60: var_63}
    var_65 = False
    var_66 = module_0.from_json_schema_type(var_64, var_48, var_65, var_28)
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
    var_67 = var_66.pattern_properties[var_61]
    var_68 = 'additionalProperties'
    var_69 = False
    var_70 = {var_0: var_48, var_68: var_69}
    var_71 = False
    var_72 = module_0.from_json_schema_type(var_70, var_48, var_71, var_28)
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
    var_73 = {var_0: var_3}
    var_74 = {var_0: var_48, var_68: var_73}
    var_75 = False
    var_76 = module_0.from_json_schema_type(var_74, var_48, var_75, var_28)
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.fields.Object'
    assert var_76.title == ''
    assert var_76.description == ''
    assert var_76.allow_null is False
    assert var_76.read_only is False
    assert var_76.properties == {}
    assert var_76.pattern_properties == {}
    assert f'{type(var_76.additional_properties).__module__}.{type(var_76.additional_properties).__qualname__}' == 'typesystem.fields.Float'
    assert var_76.property_names is None
    assert var_76.min_properties is None
    assert var_76.max_properties is None
    assert var_76.required == []
    var_77 = var_76.additional_properties
    module_0.from_json_schema_type(var_58, var_16, var_11, var_28)

def test_case_31():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'minLength'
    var_5 = 'string'
    var_6 = 5
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'pattern'
    var_9 = '^[A-Z].*'
    var_10 = {var_3: var_5, var_8: var_9}
    var_11 = {var_1: var_7, var_2: var_10}
    var_12 = module_0.if_then_else_from_json_schema(var_11, var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.if_clause).__module__}.{type(var_12.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_12.then_clause).__module__}.{type(var_12.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_12.else_clause).__module__}.{type(var_12.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_15 = 'else'
    var_16 = 'minimum'
    var_17 = 'number'
    var_18 = 0
    var_19 = {var_3: var_17, var_16: var_18}
    var_20 = 'maximum'
    var_21 = 100
    var_22 = {var_3: var_17, var_20: var_21}
    var_23 = -1
    var_24 = {var_3: var_17, var_20: var_23}
    var_25 = {var_1: var_19, var_2: var_22, var_15: var_24}
    var_26 = module_0.if_then_else_from_json_schema(var_25, var_0)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.if_clause).__module__}.{type(var_26.if_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_26.then_clause).__module__}.{type(var_26.then_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_26.else_clause).__module__}.{type(var_26.else_clause).__qualname__}' == 'typesystem.fields.Float'
    var_27 = var_26.if_clause
    var_28 = var_26.then_clause
    var_29 = var_26.else_clause
    var_30 = 'allOf'
    var_31 = 'object'
    var_32 = {var_3: var_31}
    var_33 = 'required'
    var_34 = 'active'
    var_35 = [var_34]
    var_36 = {var_33: var_35}
    var_37 = [var_32, var_36]
    var_38 = {var_30: var_37}
    var_39 = 'properties'
    var_40 = 'status'
    var_41 = {var_3: var_5}
    var_42 = {var_40: var_41}
    var_43 = {var_3: var_31, var_39: var_42}
    var_44 = 'error'
    var_45 = {var_3: var_5}
    var_46 = {var_44: var_45}
    var_47 = {var_3: var_31, var_39: var_46}
    var_48 = {var_1: var_38, var_2: var_43, var_15: var_47}
    var_49 = module_0.if_then_else_from_json_schema(var_48, var_0)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert f'{type(var_49.if_clause).__module__}.{type(var_49.if_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_49.then_clause).__module__}.{type(var_49.then_clause).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_49.else_clause).__module__}.{type(var_49.else_clause).__qualname__}' == 'typesystem.fields.Object'
    var_50 = var_49.if_clause
    var_51 = var_49.then_clause
    var_52 = var_49.else_clause
    var_53 = 'default'
    var_54 = 'boolean'
    var_55 = {var_3: var_54}
    var_56 = 'const'
    var_57 = 'yes'
    var_58 = {var_3: var_5, var_56: var_57}
    var_59 = 'no'
    var_60 = {var_3: var_5, var_56: var_59}
    var_61 = 'maybe'
    var_62 = {var_1: var_55, var_2: var_58, var_15: var_60, var_53: var_61}
    var_63 = module_0.if_then_else_from_json_schema(var_62, var_0)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_63.default == 'maybe'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert f'{type(var_63.if_clause).__module__}.{type(var_63.if_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_63.then_clause).__module__}.{type(var_63.then_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_63.else_clause).__module__}.{type(var_63.else_clause).__qualname__}' == 'typesystem.composites.AllOf'
    var_64 = 'minItems'
    var_65 = 'array'
    var_66 = 1
    var_67 = {var_3: var_65, var_64: var_66}
    var_68 = {var_1: var_67}
    var_69 = module_0.if_then_else_from_json_schema(var_68, var_0)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert f'{type(var_69.if_clause).__module__}.{type(var_69.if_clause).__qualname__}' == 'typesystem.fields.Array'
    assert f'{type(var_69.then_clause).__module__}.{type(var_69.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_69.else_clause).__module__}.{type(var_69.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_70 = var_69.if_clause
    var_71 = 'anyOf'
    var_72 = {var_3: var_5}
    var_73 = {var_3: var_17}
    var_74 = [var_72, var_73]
    var_75 = {var_71: var_74}
    var_76 = 'oneOf'
    var_77 = {var_3: var_5}
    var_78 = 'integer'
    var_79 = {var_3: var_78}
    var_80 = [var_77, var_79]
    var_81 = {var_76: var_80}
    var_82 = 'not'
    var_83 = {var_3: var_54}
    var_84 = {var_82: var_83}
    var_85 = {var_1: var_75, var_2: var_81, var_15: var_84}
    var_86 = module_0.if_then_else_from_json_schema(var_85, var_0)
    assert f'{type(var_86).__module__}.{type(var_86).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_86.title == ''
    assert var_86.description == ''
    assert var_86.allow_null is False
    assert var_86.read_only is False
    assert f'{type(var_86.if_clause).__module__}.{type(var_86.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_86.then_clause).__module__}.{type(var_86.then_clause).__qualname__}' == 'typesystem.composites.OneOf'
    assert f'{type(var_86.else_clause).__module__}.{type(var_86.else_clause).__qualname__}' == 'typesystem.composites.Not'
    var_87 = var_86.if_clause
    var_88 = var_86.then_clause
    var_89 = var_86.else_clause
    var_90 = module_2.String()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_91 = '$ref'
    var_92 = '#/components/schemas/User'
    var_93 = {var_91: var_92}
    var_94 = {var_3: var_5}
    var_95 = {var_3: var_17}
    var_96 = {var_1: var_93, var_2: var_94, var_15: var_95}
    var_97 = module_0.if_then_else_from_json_schema(var_96, var_0)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_97.title == ''
    assert var_97.description == ''
    assert var_97.allow_null is False
    assert var_97.read_only is False
    assert f'{type(var_97.if_clause).__module__}.{type(var_97.if_clause).__qualname__}' == 'typesystem.schemas.Reference'
    assert f'{type(var_97.then_clause).__module__}.{type(var_97.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_97.else_clause).__module__}.{type(var_97.else_clause).__qualname__}' == 'typesystem.fields.Float'
    var_98 = var_97.if_clause
    var_99 = var_97.then_clause
    var_100 = var_97.else_clause

def test_case_32():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = '#/definitions/Address'
    var_3 = {var_1: var_2}
    var_4 = module_0.ref_from_json_schema(var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == '#/definitions/Address'
    assert f'{type(var_4.definitions).__module__}.{type(var_4.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_5 = '$ref'
    var_6 = 'http://example.com/schema.json'
    var_7 = {var_5: var_6}
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_7, var_0)

def test_case_33():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'enum'
    var_2 = 'green'
    var_3 = 'blue'
    var_4 = [var_2, var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = module_0.enum_from_json_schema(var_5, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Choice'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.choices == [('green', 'green'), ('green', 'green'), ('blue', 'blue')]
    assert var_6.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = {var_1: var_10}
    var_12 = module_0.enum_from_json_schema(var_11, var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Choice'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.choices == [(1, 1), (2, 2), (3, 3)]
    assert var_12.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_13 = 'text'
    var_14 = 42
    var_15 = True
    var_16 = None
    var_17 = [var_13, var_14, var_15, var_16]
    var_18 = module_0.enum_from_json_schema(var_11, var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Choice'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.choices == [(1, 1), (2, 2), (3, 3)]
    assert var_18.coerce_types is True
    var_19 = {var_1: var_17}
    var_20 = module_0.enum_from_json_schema(var_19, var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Choice'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.choices == [('text', 'text'), (42, 42), (True, True), (None, None)]
    assert var_20.coerce_types is True
    var_21 = 'default'
    var_22 = 'a'
    var_23 = 'b'
    var_24 = [var_22, var_23, var_2]
    var_25 = {var_1: var_24, var_21: var_23}
    var_26 = module_0.enum_from_json_schema(var_25, var_0)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Choice'
    assert var_26.default == 'b'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.choices == [('a', 'a'), ('b', 'b'), ('green', 'green')]
    assert var_26.coerce_types is True
    var_27 = []
    var_28 = {var_1: var_27}
    var_29 = module_0.enum_from_json_schema(var_28, var_0)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Choice'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert var_29.choices == []
    assert var_29.coerce_types is True
    var_30 = 'only'
    var_31 = [var_30]
    var_32 = {var_1: var_31}
    var_33 = module_0.enum_from_json_schema(var_32, var_0)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Choice'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.choices == [('only', 'only')]
    assert var_33.coerce_types is True

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_2.Choice(coerce_types=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Choice'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.choices == []
    assert f'{type(var_1.coerce_types).__module__}.{type(var_1.coerce_types).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.coerce_types) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_4 = module_0.to_json_schema(var_1, var_2)
    var_5 = module_0.to_json_schema(var_3)
    var_6 = module_0.from_json_schema(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Choice'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.choices == []
    assert var_6.coerce_types is True
    var_7 = module_0.any_of_from_json_schema(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Union'
    assert var_7.default is None
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.any_of).__module__}.{type(var_7.any_of).__qualname__}' == 'builtins.list'
    assert len(var_7.any_of) == 5
    var_8 = module_0.to_json_schema(var_1)
    var_9 = var_7.get_default_value()
    var_10 = module_5.IfThenElse(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.if_clause is None
    assert f'{type(var_10.then_clause).__module__}.{type(var_10.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_10.else_clause).__module__}.{type(var_10.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_0.to_json_schema(var_10)

def test_case_35():
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_3.Definitions()
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_8 = module_0.type_from_json_schema(var_7, var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Union'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.any_of).__module__}.{type(var_8.any_of).__qualname__}' == 'builtins.list'
    assert len(var_8.any_of) == 2
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_9 = var_8.any_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 0
    var_12 = var_8.any_of[var_11]
    var_13 = 1
    var_14 = var_8.any_of[var_13]
    var_15 = 'null'
    var_16 = [var_1, var_15]
    var_17 = {var_0: var_16}
    var_18 = module_0.type_from_json_schema(var_17, var_3)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.String'
    assert var_18.default is None
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is True
    assert var_18.read_only is False
    assert var_18.allow_blank is True
    assert var_18.trim_whitespace is True
    assert var_18.max_length is None
    assert var_18.min_length is None
    assert var_18.format is None
    assert var_18.coerce_types is False
    assert var_18.pattern is None
    assert var_18.pattern_regex is None
    var_19 = {var_0: var_15}
    var_20 = module_0.type_from_json_schema(var_19, var_3)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Const'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.const is None
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_21 = []
    var_22 = {var_0: var_21}
    var_23 = module_0.type_from_json_schema(var_22, var_3)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Union'
    assert var_23.default is None
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is True
    assert var_23.read_only is False
    assert f'{type(var_23.any_of).__module__}.{type(var_23.any_of).__qualname__}' == 'builtins.list'
    assert len(var_23.any_of) == 5
    var_24 = [var_15]
    var_25 = {var_0: var_24}
    var_26 = module_0.type_from_json_schema(var_25, var_3)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Const'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.const is None
    var_27 = 'integer'
    var_28 = {var_0: var_27}
    var_29 = module_0.type_from_json_schema(var_28, var_3)
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
    assert var_29.coerce_types is False
    var_30 = {var_0: var_5}
    var_31 = module_0.type_from_json_schema(var_30, var_3)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Float'
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
    assert var_31.coerce_types is False
    var_32 = 'boolean'
    var_33 = {var_0: var_32}
    var_34 = module_0.type_from_json_schema(var_33, var_3)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_35 = 'array'
    var_36 = {var_0: var_35}
    var_37 = module_0.type_from_json_schema(var_36, var_3)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Array'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.items is None
    assert var_37.additional_items is True
    assert var_37.min_items == 0
    assert var_37.max_items is None
    assert var_37.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    module_0.type_from_json_schema(var_12, var_3)

def test_case_37():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
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
    var_35 = {var_4: var_5}
    var_36 = 'minLength'
    var_37 = 5
    var_38 = {var_36: var_37}
    var_39 = [var_35, var_38]
    var_40 = {var_34: var_39}
    var_41 = module_0.from_json_schema(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.all_of).__module__}.{type(var_41.all_of).__qualname__}' == 'builtins.list'
    assert len(var_41.all_of) == 2
    var_42 = {var_4: var_5}
    var_43 = {var_4: var_8}
    var_44 = [var_42, var_43]
    var_45 = {var_8: var_44}
    var_46 = module_0.from_json_schema(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Any'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    var_47 = 'oneOf'
    var_48 = {var_4: var_5}
    var_49 = {var_4: var_8}
    var_50 = [var_48, var_49]
    var_51 = {var_47: var_50}
    var_52 = module_0.from_json_schema(var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert f'{type(var_52.one_of).__module__}.{type(var_52.one_of).__qualname__}' == 'builtins.list'
    assert len(var_52.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_53 = 'not'
    var_54 = {var_4: var_5}
    var_55 = {var_53: var_54}
    var_56 = module_0.from_json_schema(var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.composites.Not'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.negated).__module__}.{type(var_56.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_57 = 'if'
    var_58 = 'then'
    var_59 = 'else'
    var_60 = {var_4: var_5}
    var_61 = {var_36: var_37}
    var_62 = {var_4: var_8}
    var_63 = {var_57: var_60, var_58: var_61, var_59: var_62}
    var_64 = module_0.from_json_schema(var_63)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert f'{type(var_64.if_clause).__module__}.{type(var_64.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_64.then_clause).__module__}.{type(var_64.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_64.else_clause).__module__}.{type(var_64.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_65 = 'maxLength'
    var_66 = 10
    var_67 = {var_4: var_5, var_36: var_37, var_65: var_66}
    var_68 = module_0.from_json_schema(var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.fields.String'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert var_68.allow_blank is False
    assert var_68.trim_whitespace is True
    assert var_68.max_length == 10
    assert var_68.min_length == 5
    assert var_68.format is None
    assert var_68.coerce_types is False
    assert var_68.pattern is None
    assert var_68.pattern_regex is None
    var_69 = {}
    var_70 = module_0.from_json_schema(var_69)
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.fields.Any'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    var_71 = '$ref'
    var_72 = 'components'
    var_73 = '#/components/schemas/User'
    var_74 = 'schemas'
    var_75 = 'User'
    var_76 = {var_4: var_5}
    var_77 = {var_75: var_76}
    var_78 = {var_74: var_77}
    var_79 = {var_71: var_73, var_72: var_78}
    var_80 = module_0.from_json_schema(var_79)
    assert f'{type(var_80).__module__}.{type(var_80).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_80.title == ''
    assert var_80.description == ''
    assert var_80.allow_null is False
    assert var_80.read_only is False
    assert var_80.to == '#/components/schemas/User'
    assert f'{type(var_80.definitions).__module__}.{type(var_80.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_80.definitions) == 1
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    var_81 = 'items'
    var_82 = {var_4: var_5}
    var_83 = {var_4: var_17, var_81: var_82}
    var_84 = module_0.from_json_schema(var_83)
    assert f'{type(var_84).__module__}.{type(var_84).__qualname__}' == 'typesystem.fields.Array'
    assert var_84.title == ''
    assert var_84.description == ''
    assert var_84.allow_null is False
    assert var_84.read_only is False
    assert f'{type(var_84.items).__module__}.{type(var_84.items).__qualname__}' == 'typesystem.fields.String'
    assert var_84.additional_items is True
    assert var_84.min_items == 0
    assert var_84.max_items is None
    assert var_84.unique_items is False
    var_85 = var_84.items
    var_86 = 'properties'
    var_87 = 'name'
    var_88 = 'age'
    var_89 = {var_4: var_5}
    var_90 = {var_4: var_8}
    var_91 = {var_87: var_89, var_88: var_90}
    var_92 = {var_4: var_20, var_86: var_91}
    var_93 = module_0.from_json_schema(var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.Object'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert f'{type(var_93.properties).__module__}.{type(var_93.properties).__qualname__}' == 'builtins.dict'
    assert len(var_93.properties) == 2
    assert var_93.pattern_properties == {}
    assert var_93.additional_properties is None
    assert var_93.property_names is None
    assert var_93.min_properties is None
    assert var_93.max_properties is None
    assert var_93.required == []
    var_94 = 'pattern'
    var_95 = '^[a-z]+$'
    var_96 = {var_4: var_5, var_94: var_95}
    var_97 = module_0.from_json_schema(var_96)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.fields.String'
    assert var_97.title == ''
    assert var_97.description == ''
    assert var_97.allow_null is False
    assert var_97.read_only is False
    assert var_97.default == ''
    assert var_97.allow_blank is True
    assert var_97.trim_whitespace is True
    assert var_97.max_length is None
    assert var_97.min_length is None
    assert var_97.format is None
    assert var_97.coerce_types is False
    assert var_97.pattern == '^[a-z]+$'
    assert f'{type(var_97.pattern_regex).__module__}.{type(var_97.pattern_regex).__qualname__}' == 're.Pattern'
    with pytest.raises(AttributeError):
        var_98 = var_97.fields

def test_case_38():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_2.Choice(coerce_types=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Choice'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.choices == []
    assert f'{type(var_1.coerce_types).__module__}.{type(var_1.coerce_types).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.coerce_types) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
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
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = False
    var_3 = module_2.Field(default=var_1, allow_null=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_4 = [var_3, var_3, var_3]
    var_5 = module_5.AllOf(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.all_of).__module__}.{type(var_5.all_of).__qualname__}' == 'builtins.list'
    assert len(var_5.all_of) == 3
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_3, var_0)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = None
    var_1 = -2172.10262
    var_2 = True
    var_3 = module_2.Float(exclusive_minimum=var_0, exclusive_maximum=var_1, coerce_types=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Float'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum is None
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum == pytest.approx(-2172.10262, abs=0.01, rel=0.01)
    assert var_3.multiple_of is None
    assert var_3.precision is None
    assert var_3.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_4 = module_0.to_json_schema(var_3, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    module_0.to_json_schema(var_0)

def test_case_41():
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
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
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_0.to_json_schema(var_9)
    var_12 = 100
    var_13 = 2
    var_14 = module_2.Integer(minimum=var_4, maximum=var_12, exclusive_minimum=var_4, exclusive_maximum=var_12, multiple_of=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum is False
    assert var_14.maximum == 100
    assert var_14.exclusive_minimum is False
    assert var_14.exclusive_maximum == 100
    assert var_14.multiple_of == 2
    assert var_14.precision is None
    assert var_14.coerce_types is True
    var_15 = module_0.to_json_schema(var_14)
    var_16 = True
    var_17 = module_2.Float(minimum=var_4, maximum=var_16)
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
    var_22 = True
    var_23 = module_2.Array(var_21, min_items=var_16, max_items=var_6, unique_items=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Array'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.items).__module__}.{type(var_23.items).__qualname__}' == 'typesystem.fields.String'
    assert var_23.additional_items is False
    assert var_23.min_items is True
    assert var_23.max_items == 10
    assert var_23.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_24 = module_0.to_json_schema(var_23)
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
    var_26 = module_2.Integer()
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
    var_27 = [var_25, var_26]
    var_28 = module_2.Array(var_27, var_4)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Array'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.items).__module__}.{type(var_28.items).__qualname__}' == 'builtins.list'
    assert len(var_28.items) == 2
    assert var_28.additional_items is False
    assert var_28.min_items == 2
    assert var_28.max_items == 2
    assert var_28.unique_items is False
    var_29 = module_0.to_json_schema(var_28)
    var_30 = 'items'
    var_31 = var_29[var_30]
    var_32 = var_29[var_30]
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = 'name'
    var_35 = 'age'
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
    var_37 = module_2.Integer()
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
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = [var_34]
    var_40 = module_2.Object(properties=var_38, additional_properties=var_4, required=var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Object'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert f'{type(var_40.properties).__module__}.{type(var_40.properties).__qualname__}' == 'builtins.dict'
    assert len(var_40.properties) == 2
    assert var_40.pattern_properties == {}
    assert var_40.additional_properties is False
    assert var_40.property_names is None
    assert var_40.min_properties is None
    assert var_40.max_properties is None
    assert var_40.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_41 = module_0.to_json_schema(var_40)
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
    var_43 = {var_7: var_42}
    var_44 = module_2.Object(pattern_properties=var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Object'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert var_44.properties == {}
    assert f'{type(var_44.pattern_properties).__module__}.{type(var_44.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_44.pattern_properties) == 1
    assert var_44.additional_properties is True
    assert var_44.property_names is None
    assert var_44.min_properties is None
    assert var_44.max_properties is None
    assert var_44.required == []
    var_45 = module_0.to_json_schema(var_44)
    var_46 = 'A'
    var_47 = (var_46, var_46)
    var_48 = 'B'
    var_49 = (var_48, var_48)
    var_50 = [var_47, var_49]
    var_51 = module_2.Choice(choices=var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Choice'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert var_51.choices == [('A', 'A'), ('B', 'B')]
    assert var_51.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_52 = module_0.to_json_schema(var_51)
    var_53 = 'fixed_value'
    var_54 = module_2.Const(var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Const'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert var_54.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_55 = module_0.to_json_schema(var_54)
    var_56 = module_2.String()
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
    var_57 = module_2.Integer()
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
    var_59 = module_2.Union(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.fields.Union'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.any_of).__module__}.{type(var_59.any_of).__qualname__}' == 'builtins.list'
    assert len(var_59.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_60 = module_0.to_json_schema(var_59)
    var_61 = 'anyOf'
    var_62 = var_60[var_61]
    var_63 = len(var_62)
    assert var_63 == 2
    var_64 = module_2.String()
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
    var_65 = module_2.Integer()
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Integer'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.minimum is None
    assert var_65.maximum is None
    assert var_65.exclusive_minimum is None
    assert var_65.exclusive_maximum is None
    assert var_65.multiple_of is None
    assert var_65.precision is None
    assert var_65.coerce_types is True
    var_66 = [var_64, var_65]
    var_67 = module_5.OneOf(var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    assert f'{type(var_67.one_of).__module__}.{type(var_67.one_of).__qualname__}' == 'builtins.list'
    assert len(var_67.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_68 = module_0.to_json_schema(var_67)
    var_69 = 'oneOf'
    var_70 = var_68[var_69]
    var_71 = len(var_70)
    assert var_71 == 2
    var_72 = module_2.String(min_length=var_22)
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.String'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert var_72.allow_blank is False
    assert var_72.trim_whitespace is True
    assert var_72.max_length is None
    assert var_72.min_length is True
    assert var_72.format is None
    assert var_72.coerce_types is True
    assert var_72.pattern is None
    assert var_72.pattern_regex is None
    var_73 = module_2.String(max_length=var_6)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.String'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert var_73.allow_blank is False
    assert var_73.trim_whitespace is True
    assert var_73.max_length == 10
    assert var_73.min_length is None
    assert var_73.format is None
    assert var_73.coerce_types is True
    assert var_73.pattern is None
    assert var_73.pattern_regex is None
    var_74 = [var_72, var_73]
    var_75 = module_5.AllOf(var_74)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert f'{type(var_75.all_of).__module__}.{type(var_75.all_of).__qualname__}' == 'builtins.list'
    assert len(var_75.all_of) == 2
    var_76 = module_0.to_json_schema(var_75)
    var_77 = 'allOf'
    var_78 = var_76[var_77]
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = module_2.String(pattern=var_7)
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
    assert var_80.pattern == '^[a-z]+$'
    assert f'{type(var_80.pattern_regex).__module__}.{type(var_80.pattern_regex).__qualname__}' == 're.Pattern'
    var_81 = module_2.Integer(minimum=var_4)
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'typesystem.fields.Integer'
    assert var_81.title == ''
    assert var_81.description == ''
    assert var_81.allow_null is False
    assert var_81.read_only is False
    assert var_81.minimum is False
    assert var_81.maximum is None
    assert var_81.exclusive_minimum is None
    assert var_81.exclusive_maximum is None
    assert var_81.multiple_of is None
    assert var_81.precision is None
    assert var_81.coerce_types is True
    var_82 = module_2.Boolean()
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert var_82.coerce_types is True
    var_83 = module_5.IfThenElse(var_80, var_81, var_82)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert f'{type(var_83.if_clause).__module__}.{type(var_83.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_83.then_clause).__module__}.{type(var_83.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_83.else_clause).__module__}.{type(var_83.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_84 = module_0.to_json_schema(var_83)
    var_85 = '^[0-9]+$'
    var_86 = module_2.String(pattern=var_85)
    assert f'{type(var_86).__module__}.{type(var_86).__qualname__}' == 'typesystem.fields.String'
    assert var_86.title == ''
    assert var_86.description == ''
    assert var_86.allow_null is False
    assert var_86.read_only is False
    assert var_86.allow_blank is False
    assert var_86.trim_whitespace is True
    assert var_86.max_length is None
    assert var_86.min_length is None
    assert var_86.format is None
    assert var_86.coerce_types is True
    assert var_86.pattern == '^[0-9]+$'
    assert f'{type(var_86.pattern_regex).__module__}.{type(var_86.pattern_regex).__qualname__}' == 're.Pattern'
    var_87 = module_5.Not(var_86)
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.composites.Not'
    assert var_87.title == ''
    assert var_87.description == ''
    assert var_87.allow_null is False
    assert var_87.read_only is False
    assert f'{type(var_87.negated).__module__}.{type(var_87.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_88 = module_0.to_json_schema(var_87)
    var_89 = module_2.String()
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.fields.String'
    assert var_89.title == ''
    assert var_89.description == ''
    assert var_89.allow_null is False
    assert var_89.read_only is False
    assert var_89.allow_blank is False
    assert var_89.trim_whitespace is True
    assert var_89.max_length is None
    assert var_89.min_length is None
    assert var_89.format is None
    assert var_89.coerce_types is True
    assert var_89.pattern is None
    assert var_89.pattern_regex is None
    var_90 = {}
    var_91 = module_0.to_json_schema(var_87, var_90)
    var_92 = module_2.String()
    assert f'{type(var_92).__module__}.{type(var_92).__qualname__}' == 'typesystem.fields.String'
    assert var_92.title == ''
    assert var_92.description == ''
    assert var_92.allow_null is False
    assert var_92.read_only is False
    assert var_92.allow_blank is False
    assert var_92.trim_whitespace is True
    assert var_92.max_length is None
    assert var_92.min_length is None
    assert var_92.format is None
    assert var_92.coerce_types is True
    assert var_92.pattern is None
    assert var_92.pattern_regex is None
    with pytest.raises(AssertionError):
        module_2.Object(properties=var_41)

@pytest.mark.xfail(strict=True)
def test_case_42():
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
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
    var_10 = module_0.to_json_schema(var_9)
    var_11 = True
    var_12 = module_2.String(allow_blank=var_4)
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
    var_14 = 100
    var_15 = 5
    var_16 = module_2.Integer(minimum=var_4, maximum=var_14, multiple_of=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Integer'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum is False
    assert var_16.maximum == 100
    assert var_16.exclusive_minimum is None
    assert var_16.exclusive_maximum is None
    assert var_16.multiple_of == 5
    assert var_16.precision is None
    assert var_16.coerce_types is True
    var_17 = module_0.to_json_schema(var_16)
    var_18 = True
    var_19 = module_2.Float(exclusive_minimum=var_4, exclusive_maximum=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Float'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.minimum is None
    assert var_19.maximum is None
    assert var_19.exclusive_minimum is False
    assert var_19.exclusive_maximum is True
    assert var_19.multiple_of is None
    assert var_19.precision is None
    assert var_19.coerce_types is True
    var_20 = module_0.to_json_schema(var_19)
    var_21 = module_2.Boolean()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_22 = module_0.to_json_schema(var_21)
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
    var_24 = True
    var_25 = module_2.Array(var_23, var_4, var_18, var_6, unique_items=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Array'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert f'{type(var_25.items).__module__}.{type(var_25.items).__qualname__}' == 'typesystem.fields.String'
    assert var_25.additional_items is False
    assert var_25.min_items is True
    assert var_25.max_items == 10
    assert var_25.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_26 = module_0.to_json_schema(var_25)
    var_27 = 'name'
    var_28 = 'age'
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
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = [var_27]
    var_33 = 2
    var_34 = module_2.Object(properties=var_31, additional_properties=var_4, min_properties=var_24, max_properties=var_33, required=var_32)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Object'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.properties).__module__}.{type(var_34.properties).__qualname__}' == 'builtins.dict'
    assert len(var_34.properties) == 2
    assert var_34.pattern_properties == {}
    assert var_34.additional_properties is False
    assert var_34.property_names is None
    assert var_34.min_properties is True
    assert var_34.max_properties == 2
    assert var_34.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_35 = module_0.to_json_schema(var_34)
    var_36 = 'a'
    var_37 = 'A'
    var_38 = (var_36, var_37)
    var_39 = 'b'
    var_40 = 'B'
    var_41 = (var_39, var_40)
    var_42 = [var_38, var_41]
    var_43 = module_2.Choice(choices=var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Choice'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.choices == [('a', 'A'), ('b', 'B')]
    assert var_43.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_44 = module_0.to_json_schema(var_43)
    var_45 = 'fixed_value'
    var_46 = module_2.Const(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Const'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert var_46.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_47 = module_0.to_json_schema(var_46)
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
    var_49 = module_2.Integer()
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
    var_51 = module_2.Union(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Union'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.any_of).__module__}.{type(var_51.any_of).__qualname__}' == 'builtins.list'
    assert len(var_51.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_52 = module_0.to_json_schema(var_51)
    var_53 = 'anyOf'
    var_54 = var_52[var_53]
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = module_2.String()
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
    var_57 = module_2.Integer()
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
    var_59 = module_5.OneOf(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.one_of).__module__}.{type(var_59.one_of).__qualname__}' == 'builtins.list'
    assert len(var_59.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_60 = module_0.to_json_schema(var_59)
    var_61 = 'oneOf'
    var_62 = var_60[var_61]
    var_63 = len(var_62)
    assert var_63 == 2
    var_64 = module_2.String(min_length=var_24)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.fields.String'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert var_64.allow_blank is False
    assert var_64.trim_whitespace is True
    assert var_64.max_length is None
    assert var_64.min_length is True
    assert var_64.format is None
    assert var_64.coerce_types is True
    assert var_64.pattern is None
    assert var_64.pattern_regex is None
    var_65 = module_2.String(max_length=var_6)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.String'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.allow_blank is False
    assert var_65.trim_whitespace is True
    assert var_65.max_length == 10
    assert var_65.min_length is None
    assert var_65.format is None
    assert var_65.coerce_types is True
    assert var_65.pattern is None
    assert var_65.pattern_regex is None
    var_66 = [var_64, var_65]
    var_67 = module_5.AllOf(var_66)
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
    var_72 = module_2.String(min_length=var_15)
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.String'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert var_72.allow_blank is False
    assert var_72.trim_whitespace is True
    assert var_72.max_length is None
    assert var_72.min_length == 5
    assert var_72.format is None
    assert var_72.coerce_types is True
    assert var_72.pattern is None
    assert var_72.pattern_regex is None
    var_73 = module_2.String(max_length=var_6)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.String'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert var_73.allow_blank is False
    assert var_73.trim_whitespace is True
    assert var_73.max_length == 10
    assert var_73.min_length is None
    assert var_73.format is None
    assert var_73.coerce_types is True
    assert var_73.pattern is None
    assert var_73.pattern_regex is None
    var_74 = module_2.Integer()
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
    var_75 = module_5.IfThenElse(var_72, var_73, var_74)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert f'{type(var_75.if_clause).__module__}.{type(var_75.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_75.then_clause).__module__}.{type(var_75.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_75.else_clause).__module__}.{type(var_75.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_76 = module_0.to_json_schema(var_75)
    var_77 = module_2.String()
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.String'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert var_77.allow_blank is False
    assert var_77.trim_whitespace is True
    assert var_77.max_length is None
    assert var_77.min_length is None
    assert var_77.format is None
    assert var_77.coerce_types is True
    assert var_77.pattern is None
    assert var_77.pattern_regex is None
    var_78 = module_5.Not(var_77)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.composites.Not'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert f'{type(var_78.negated).__module__}.{type(var_78.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_79 = module_0.to_json_schema(var_78)
    var_80 = module_3.Definitions()
    assert f'{type(var_80).__module__}.{type(var_80).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_80) == 0
    var_81 = module_2.String()
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'typesystem.fields.String'
    assert var_81.title == ''
    assert var_81.description == ''
    assert var_81.allow_null is False
    assert var_81.read_only is False
    assert var_81.allow_blank is False
    assert var_81.trim_whitespace is True
    assert var_81.max_length is None
    assert var_81.min_length is None
    assert var_81.format is None
    assert var_81.coerce_types is True
    assert var_81.pattern is None
    assert var_81.pattern_regex is None
    var_82 = 'Person'
    var_83 = module_3.Reference(var_82, var_80)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert var_83.to == 'Person'
    assert f'{type(var_83.definitions).__module__}.{type(var_83.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_83.definitions) == 0
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    module_0.to_json_schema(var_83)

@pytest.mark.xfail(strict=True)
def test_case_43():
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 10
    var_6 = True
    var_7 = module_2.String()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format is None
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_0.to_json_schema(var_7)
    var_9 = 100
    var_10 = module_2.Integer(minimum=var_4, maximum=var_9)
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
    var_12 = module_2.Float(minimum=var_4, maximum=var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Float'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum is False
    assert var_12.maximum is True
    assert var_12.exclusive_minimum is None
    assert var_12.exclusive_maximum is None
    assert var_12.multiple_of is None
    assert var_12.precision is None
    assert var_12.coerce_types is True
    var_13 = module_0.to_json_schema(var_12)
    var_14 = False
    var_15 = module_2.Boolean()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_2.String()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.String'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.allow_blank is False
    assert var_17.trim_whitespace is True
    assert var_17.max_length is None
    assert var_17.min_length is None
    assert var_17.format is None
    assert var_17.coerce_types is True
    assert var_17.pattern is None
    assert var_17.pattern_regex is None
    var_18 = 5
    var_19 = module_2.Array(var_17, min_items=var_14, max_items=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Array'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.items).__module__}.{type(var_19.items).__qualname__}' == 'typesystem.fields.String'
    assert var_19.additional_items is False
    assert var_19.min_items is False
    assert var_19.max_items == 5
    assert var_19.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
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
    var_22 = [var_21, var_10]
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
    var_24 = module_0.to_json_schema(var_23)
    var_25 = 'items'
    var_26 = var_24[var_25]
    var_27 = var_24[var_25]
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = 'name'
    var_30 = 'age'
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
    var_33 = {var_29: var_31, var_30: var_32}
    var_34 = [var_29]
    var_35 = module_2.Object(properties=var_33, required=var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Object'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.properties).__module__}.{type(var_35.properties).__qualname__}' == 'builtins.dict'
    assert len(var_35.properties) == 2
    assert var_35.pattern_properties == {}
    assert var_35.additional_properties is True
    assert var_35.property_names is None
    assert var_35.min_properties is None
    assert var_35.max_properties is None
    assert var_35.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_36 = module_0.to_json_schema(var_35)
    var_37 = '\\d+'
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
    var_39 = {var_37: var_38}
    var_40 = module_2.Object(pattern_properties=var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Object'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert var_40.properties == {}
    assert f'{type(var_40.pattern_properties).__module__}.{type(var_40.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_40.pattern_properties) == 1
    assert var_40.additional_properties is True
    assert var_40.property_names is None
    assert var_40.min_properties is None
    assert var_40.max_properties is None
    assert var_40.required == []
    var_41 = module_0.to_json_schema(var_40)
    var_42 = 'B'
    var_43 = (var_42, var_42)
    var_44 = [var_27, var_43]
    var_45 = module_2.Choice(choices=var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Choice'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.choices == [[{'type': 'string', 'minLength': 1}, {'type': 'integer', 'minimum': False, 'maximum': 100}], ('B', 'B')]
    assert var_45.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_46 = 'fixed_value'
    var_47 = module_2.Const(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.Const'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert var_47.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
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
    var_51 = [var_49, var_50]
    var_52 = module_2.Union(var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Union'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert f'{type(var_52.any_of).__module__}.{type(var_52.any_of).__qualname__}' == 'builtins.list'
    assert len(var_52.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_53 = module_0.to_json_schema(var_52)
    var_54 = 'anyOf'
    var_55 = var_53[var_54]
    var_56 = len(var_55)
    assert var_56 == 2
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
    var_58 = module_2.Integer()
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
    var_59 = [var_57, var_58]
    var_60 = module_5.OneOf(var_59)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert f'{type(var_60.one_of).__module__}.{type(var_60.one_of).__qualname__}' == 'builtins.list'
    assert len(var_60.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_61 = module_0.to_json_schema(var_60)
    var_62 = 'oneOf'
    var_63 = var_61[var_62]
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = module_2.String(min_length=var_14)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.String'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.allow_blank is False
    assert var_65.trim_whitespace is True
    assert var_65.max_length is None
    assert var_65.min_length is False
    assert var_65.format is None
    assert var_65.coerce_types is True
    assert var_65.pattern is None
    assert var_65.pattern_regex is None
    var_66 = module_2.String(max_length=var_5)
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.fields.String'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert var_66.allow_blank is False
    assert var_66.trim_whitespace is True
    assert var_66.max_length == 10
    assert var_66.min_length is None
    assert var_66.format is None
    assert var_66.coerce_types is True
    assert var_66.pattern is None
    assert var_66.pattern_regex is None
    var_67 = [var_65, var_66]
    var_68 = module_5.AllOf(var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert f'{type(var_68.all_of).__module__}.{type(var_68.all_of).__qualname__}' == 'builtins.list'
    assert len(var_68.all_of) == 2
    var_69 = module_0.to_json_schema(var_68)
    var_70 = 'allOf'
    var_71 = var_69[var_70]
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
    var_74 = module_2.Boolean()
    assert f'{type(var_74).__module__}.{type(var_74).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_74.title == ''
    assert var_74.description == ''
    assert var_74.allow_null is False
    assert var_74.read_only is False
    assert var_74.coerce_types is True
    var_75 = module_5.IfThenElse(var_72, var_73, var_74)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert f'{type(var_75.if_clause).__module__}.{type(var_75.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_75.then_clause).__module__}.{type(var_75.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_75.else_clause).__module__}.{type(var_75.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_76 = module_0.to_json_schema(var_75)
    var_77 = module_2.String()
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.String'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert var_77.allow_blank is False
    assert var_77.trim_whitespace is True
    assert var_77.max_length is None
    assert var_77.min_length is None
    assert var_77.format is None
    assert var_77.coerce_types is True
    assert var_77.pattern is None
    assert var_77.pattern_regex is None
    var_78 = module_5.Not(var_77)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.composites.Not'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert f'{type(var_78.negated).__module__}.{type(var_78.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_79 = module_0.to_json_schema(var_78)
    var_80 = module_2.String()
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
    var_81 = {var_29: var_80}
    var_82 = module_2.Object(properties=var_81)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.Object'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert f'{type(var_82.properties).__module__}.{type(var_82.properties).__qualname__}' == 'builtins.dict'
    assert len(var_82.properties) == 1
    assert var_82.pattern_properties == {}
    assert var_82.additional_properties is True
    assert var_82.property_names is None
    assert var_82.min_properties is None
    assert var_82.max_properties is None
    assert var_82.required == []
    var_83 = module_2.String()
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
    var_84 = 'default_value'
    var_85 = module_2.String()
    assert f'{type(var_85).__module__}.{type(var_85).__qualname__}' == 'typesystem.fields.String'
    assert var_85.title == ''
    assert var_85.description == ''
    assert var_85.allow_null is False
    assert var_85.read_only is False
    assert var_85.allow_blank is False
    assert var_85.trim_whitespace is True
    assert var_85.max_length is None
    assert var_85.min_length is None
    assert var_85.format is None
    assert var_85.coerce_types is True
    assert var_85.pattern is None
    assert var_85.pattern_regex is None
    var_86 = module_0.to_json_schema(var_85)
    var_87 = module_2.String()
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
    var_88 = {var_29: var_87}
    var_89 = [var_29]
    var_90 = module_3.Schema(var_88)
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_90.title == ''
    assert var_90.description == ''
    assert var_90.allow_null is False
    assert var_90.read_only is False
    assert f'{type(var_90.fields).__module__}.{type(var_90.fields).__qualname__}' == 'builtins.dict'
    assert len(var_90.fields) == 1
    assert var_90.required == ['name']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_91 = module_0.to_json_schema(var_90)
    var_92 = '^\\d+$'
    var_93 = module_4.compile(var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 're.Pattern'
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
    assert f'{type(module_4.Pattern.pattern).__module__}.{type(module_4.Pattern.pattern).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.flags).__module__}.{type(module_4.Pattern.flags).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.groups).__module__}.{type(module_4.Pattern.groups).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.groupindex).__module__}.{type(module_4.Pattern.groupindex).__qualname__}' == 'builtins.getset_descriptor'
    var_56.validate(var_26)

def test_case_44():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = module_3.Definitions()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = 'exclusiveMinimum'
    var_4 = 'multipleOf'
    var_5 = 'integer'
    var_6 = 5
    var_7 = 2
    var_8 = {var_0: var_5, var_3: var_6, var_4: var_7}
    var_9 = True
    var_10 = module_3.Definitions()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_10) == 0
    var_11 = module_0.from_json_schema_type(var_8, var_5, var_9, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Integer'
    assert var_11.default is None
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is True
    assert var_11.read_only is False
    assert var_11.minimum is None
    assert var_11.maximum is None
    assert var_11.exclusive_minimum == 5
    assert var_11.exclusive_maximum is None
    assert var_11.multiple_of == 2
    assert var_11.precision is None
    assert var_11.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_12 = 'minLength'
    var_13 = 'pattern'
    var_14 = 'string'
    var_15 = 3
    var_16 = '^[-Z]+$'
    var_17 = {var_0: var_14, var_12: var_15, var_13: var_16}
    var_18 = False
    var_19 = module_3.Definitions()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_19) == 0
    var_20 = module_0.from_json_schema_type(var_17, var_14, var_18, var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.String'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.allow_blank is False
    assert var_20.trim_whitespace is True
    assert var_20.max_length is None
    assert var_20.min_length == 3
    assert var_20.format is None
    assert var_20.coerce_types is False
    assert var_20.pattern == '^[-Z]+$'
    assert f'{type(var_20.pattern_regex).__module__}.{type(var_20.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_21 = 'default'
    var_22 = 'boolean'
    var_23 = {var_0: var_22, var_21: var_9}
    var_24 = module_3.Definitions()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_24) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_25 = module_0.from_json_schema_type(var_23, var_22, var_9, var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_25.default is True
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is True
    assert var_25.read_only is False
    assert var_25.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_26 = module_3.Definitions()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_26) == 0
    var_27 = 'items'
    var_28 = 'array'
    var_29 = {var_0: var_14}
    var_30 = {var_0: var_1}
    var_31 = [var_29, var_30]
    var_32 = {var_0: var_28, var_27: var_31}
    var_33 = False
    var_34 = module_0.from_json_schema_type(var_32, var_28, var_33, var_26)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Array'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert f'{type(var_34.items).__module__}.{type(var_34.items).__qualname__}' == 'builtins.list'
    assert len(var_34.items) == 2
    assert var_34.additional_items is True
    assert var_34.min_items == 0
    assert var_34.max_items is None
    assert var_34.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_35 = var_34.items
    var_36 = var_34.items
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = var_34.items[var_33]
    var_39 = var_34.items[var_9]
    var_40 = {var_0: var_5}
    var_41 = {var_0: var_28, var_27: var_40}
    var_42 = False
    var_43 = module_0.from_json_schema_type(var_41, var_28, var_42, var_26)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Array'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert f'{type(var_43.items).__module__}.{type(var_43.items).__qualname__}' == 'typesystem.fields.Integer'
    assert var_43.additional_items is True
    assert var_43.min_items == 0
    assert var_43.max_items is None
    assert var_43.unique_items is False
    var_44 = var_43.items
    var_45 = 'properties'
    var_46 = 'required'
    var_47 = 'object'
    var_48 = 'name'
    var_49 = 'age'
    var_50 = {var_0: var_14}
    var_51 = {var_0: var_5}
    var_52 = {var_48: var_50, var_49: var_51}
    var_53 = [var_48]
    var_54 = {var_0: var_47, var_45: var_52, var_46: var_53}
    var_55 = False
    var_56 = module_0.from_json_schema_type(var_54, var_47, var_55, var_26)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Object'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.properties).__module__}.{type(var_56.properties).__qualname__}' == 'builtins.dict'
    assert len(var_56.properties) == 2
    assert var_56.pattern_properties == {}
    assert var_56.additional_properties is None
    assert var_56.property_names is None
    assert var_56.min_properties is None
    assert var_56.max_properties is None
    assert var_56.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_57 = var_56.properties[var_48]
    var_58 = var_56.properties[var_49]
    var_59 = 'patternProperties'
    var_60 = '^S_'
    var_61 = {var_0: var_14}
    var_62 = {var_60: var_61}
    var_63 = {var_0: var_47, var_59: var_62}
    var_64 = False
    var_65 = module_0.from_json_schema_type(var_63, var_47, var_64, var_26)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Object'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
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
    var_69 = {var_0: var_47, var_67: var_68}
    var_70 = False
    var_71 = module_0.from_json_schema_type(var_69, var_47, var_70, var_26)
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
    var_72 = {var_0: var_1}
    var_73 = {var_0: var_47, var_67: var_72}
    var_74 = False
    var_75 = module_0.from_json_schema_type(var_73, var_47, var_74, var_26)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.fields.Object'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert var_75.properties == {}
    assert var_75.pattern_properties == {}
    assert f'{type(var_75.additional_properties).__module__}.{type(var_75.additional_properties).__qualname__}' == 'typesystem.fields.Float'
    assert var_75.property_names is None
    assert var_75.min_properties is None
    assert var_75.max_properties is None
    assert var_75.required == []
    var_76 = var_75.additional_properties
    var_77 = {var_0: var_14}
    var_78 = module_0.from_json_schema_type(var_77, var_14, var_9, var_26)
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.String'
    assert var_78.default is None
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is True
    assert var_78.read_only is False
    assert var_78.allow_blank is True
    assert var_78.trim_whitespace is True
    assert var_78.max_length is None
    assert var_78.min_length is None
    assert var_78.format is None
    assert var_78.coerce_types is False
    assert var_78.pattern is None
    assert var_78.pattern_regex is None
    var_79 = 42
    var_80 = {var_0: var_5, var_21: var_79}
    var_81 = False
    var_82 = module_0.from_json_schema_type(var_80, var_5, var_81, var_26)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.Integer'
    assert var_82.default == 42
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert var_82.minimum is None
    assert var_82.maximum is None
    assert var_82.exclusive_minimum is None
    assert var_82.exclusive_maximum is None
    assert var_82.multiple_of is None
    assert var_82.precision is None
    assert var_82.coerce_types is False
    var_83 = 'additionalItems'
    var_84 = False
    var_85 = {var_0: var_28, var_83: var_84}
    var_86 = False
    var_87 = module_0.from_json_schema_type(var_85, var_28, var_86, var_26)
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.fields.Array'
    assert var_87.title == ''
    assert var_87.description == ''
    assert var_87.allow_null is False
    assert var_87.read_only is False
    assert var_87.items is None
    assert var_87.additional_items is False
    assert var_87.min_items == 0
    assert var_87.max_items is None
    assert var_87.unique_items is False
    var_88 = {var_0: var_22}
    var_89 = {var_0: var_28, var_83: var_88}
    var_90 = False
    var_91 = module_0.from_json_schema_type(var_89, var_28, var_90, var_26)
    assert f'{type(var_91).__module__}.{type(var_91).__qualname__}' == 'typesystem.fields.Array'
    assert var_91.title == ''
    assert var_91.description == ''
    assert var_91.allow_null is False
    assert var_91.read_only is False
    assert var_91.items is None
    assert f'{type(var_91.additional_items).__module__}.{type(var_91.additional_items).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_91.min_items == 0
    assert var_91.max_items is None
    assert var_91.unique_items is False
    var_92 = var_91.additional_items
    var_93 = 'propertyNames'
    var_94 = '^[a-z]+$'
    var_95 = {var_13: var_94}
    var_96 = {var_0: var_47, var_93: var_95}
    var_97 = False
    var_98 = module_0.from_json_schema_type(var_96, var_47, var_97, var_26)
    assert f'{type(var_98).__module__}.{type(var_98).__qualname__}' == 'typesystem.fields.Object'
    assert var_98.title == ''
    assert var_98.description == ''
    assert var_98.allow_null is False
    assert var_98.read_only is False
    assert var_98.properties == {}
    assert var_98.pattern_properties == {}
    assert var_98.additional_properties is None
    assert f'{type(var_98.property_names).__module__}.{type(var_98.property_names).__qualname__}' == 'typesystem.fields.Union'
    assert var_98.min_properties is None
    assert var_98.max_properties is None
    assert var_98.required == []
    var_99 = var_98.property_names

@pytest.mark.xfail(strict=True)
def test_case_45():
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = '^[a-wVz]+$'
    var_7 = 'email'
    var_8 = module_2.String(max_length=var_3, min_length=var_5, pattern=var_6, format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is False
    assert var_8.min_length == 1
    assert var_8.format == 'email'
    assert var_8.coerce_types is True
    assert var_8.pattern == '^[a-wVz]+$'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
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
    var_11 = module_0.to_json_schema(var_10)
    module_2.Float(minimum=var_4, maximum=var_10)

@pytest.mark.xfail(strict=True)
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
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = '^[a-wVz]+$'
    var_7 = 'email'
    var_8 = module_2.String(max_length=var_3, min_length=var_5, pattern=var_6, format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is False
    assert var_8.min_length == 1
    assert var_8.format == 'email'
    assert var_8.coerce_types is True
    assert var_8.pattern == '^[a-wVz]+$'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
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
    var_11 = module_0.to_json_schema(var_10)
    var_12 = 2
    var_13 = module_2.Integer(minimum=var_4, maximum=var_3, exclusive_minimum=var_4, exclusive_maximum=var_3, multiple_of=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum is False
    assert var_13.maximum is False
    assert var_13.exclusive_minimum is False
    assert var_13.exclusive_maximum is False
    assert var_13.multiple_of == 2
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = module_0.to_json_schema(var_13)
    module_2.Float(minimum=var_4, maximum=var_10)

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
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = module_2.String(max_length=var_6, min_length=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length == 10
    assert var_7.min_length == 1
    assert var_7.format is None
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_0.to_json_schema(var_7)
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
    var_11 = module_0.to_json_schema(var_10)
    var_12 = 100
    var_13 = module_2.Integer(minimum=var_4, maximum=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum is False
    assert var_13.maximum == 100
    assert var_13.exclusive_minimum is None
    assert var_13.exclusive_maximum is None
    assert var_13.multiple_of is None
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = module_0.to_json_schema(var_13)
    var_15 = module_2.Float(minimum=var_4, maximum=var_9)
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
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
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
    var_21 = module_2.Array(var_20, min_items=var_17, max_items=var_6)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Array'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.items).__module__}.{type(var_21.items).__qualname__}' == 'typesystem.fields.String'
    assert var_21.additional_items is False
    assert var_21.min_items is True
    assert var_21.max_items == 10
    assert var_21.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_22 = module_0.to_json_schema(var_21)
    var_23 = 'name'
    var_24 = module_2.String()
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
    var_25 = {var_23: var_24}
    var_26 = [var_23]
    var_27 = module_2.Object(properties=var_25, required=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Object'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.properties).__module__}.{type(var_27.properties).__qualname__}' == 'builtins.dict'
    assert len(var_27.properties) == 1
    assert var_27.pattern_properties == {}
    assert var_27.additional_properties is True
    assert var_27.property_names is None
    assert var_27.min_properties is None
    assert var_27.max_properties is None
    assert var_27.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_28 = module_0.to_json_schema(var_27)
    var_29 = 'a'
    var_30 = 'A'
    var_31 = (var_29, var_30)
    var_32 = 'b'
    var_33 = (var_32, var_29)
    var_34 = [var_31, var_33]
    var_35 = module_2.Choice(choices=var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Choice'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.choices == [('a', 'A'), ('b', 'a')]
    assert var_35.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_36 = module_0.to_json_schema(var_35)
    var_37 = 'fixed_value'
    var_38 = module_2.Const(var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Const'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_39 = module_0.to_json_schema(var_38)
    var_40 = module_2.String()
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
    var_41 = module_2.Integer()
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
    var_42 = [var_40, var_41]
    var_43 = module_2.Union(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Union'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert f'{type(var_43.any_of).__module__}.{type(var_43.any_of).__qualname__}' == 'builtins.list'
    assert len(var_43.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_44 = module_0.to_json_schema(var_43)
    var_45 = 'anyOf'
    var_46 = var_44[var_45]
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = module_2.String(min_length=var_17)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.String'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert var_48.allow_blank is False
    assert var_48.trim_whitespace is True
    assert var_48.max_length is None
    assert var_48.min_length is True
    assert var_48.format is None
    assert var_48.coerce_types is True
    assert var_48.pattern is None
    assert var_48.pattern_regex is None
    var_49 = module_2.String(max_length=var_6)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.fields.String'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert var_49.allow_blank is False
    assert var_49.trim_whitespace is True
    assert var_49.max_length == 10
    assert var_49.min_length is None
    assert var_49.format is None
    assert var_49.coerce_types is True
    assert var_49.pattern is None
    assert var_49.pattern_regex is None
    var_50 = [var_48, var_49]
    var_51 = module_5.AllOf(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.all_of).__module__}.{type(var_51.all_of).__qualname__}' == 'builtins.list'
    assert len(var_51.all_of) == 2
    var_52 = module_0.to_json_schema(var_51)
    var_53 = 'allOf'
    var_54 = var_52[var_53]
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = module_2.String()
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
    var_57 = module_2.Integer()
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
    var_59 = module_5.OneOf(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.one_of).__module__}.{type(var_59.one_of).__qualname__}' == 'builtins.list'
    assert len(var_59.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_60 = module_0.to_json_schema(var_59)
    var_61 = 'oneOf'
    var_62 = var_60[var_61]
    var_63 = len(var_62)
    assert var_63 == 2
    var_64 = module_2.String()
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
    var_65 = module_2.Integer()
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Integer'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.minimum is None
    assert var_65.maximum is None
    assert var_65.exclusive_minimum is None
    assert var_65.exclusive_maximum is None
    assert var_65.multiple_of is None
    assert var_65.precision is None
    assert var_65.coerce_types is True
    var_66 = module_5.IfThenElse(var_64, var_65)
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert f'{type(var_66.if_clause).__module__}.{type(var_66.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_66.then_clause).__module__}.{type(var_66.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_66.else_clause).__module__}.{type(var_66.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_67 = module_0.to_json_schema(var_66)
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
    var_69 = module_5.Not(var_68)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.composites.Not'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert f'{type(var_69.negated).__module__}.{type(var_69.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_70 = module_0.to_json_schema(var_69)
    var_71 = 'MySchema'
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
    var_73 = {var_71: var_72}
    var_74 = module_3.Reference(var_71, var_73)
    assert f'{type(var_74).__module__}.{type(var_74).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_74.title == ''
    assert var_74.description == ''
    assert var_74.allow_null is False
    assert var_74.read_only is False
    assert var_74.to == 'MySchema'
    assert f'{type(var_74.definitions).__module__}.{type(var_74.definitions).__qualname__}' == 'builtins.dict'
    assert len(var_74.definitions) == 1
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    var_75 = module_0.to_json_schema(var_74)
    var_76 = module_2.String()
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
    var_77 = 'default_value'
    var_78 = module_2.String()
    assert f'{type(var_78).__module__}.{type(var_78).__qualname__}' == 'typesystem.fields.String'
    assert var_78.title == ''
    assert var_78.description == ''
    assert var_78.allow_null is False
    assert var_78.read_only is False
    assert var_78.allow_blank is False
    assert var_78.trim_whitespace is True
    assert var_78.max_length is None
    assert var_78.min_length is None
    assert var_78.format is None
    assert var_78.coerce_types is True
    assert var_78.pattern is None
    assert var_78.pattern_regex is None
    var_79 = module_0.to_json_schema(var_78)
    var_80 = '^\\d+$'
    var_81 = module_4.compile(var_80)
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 're.Pattern'
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
    assert f'{type(module_4.Pattern.pattern).__module__}.{type(module_4.Pattern.pattern).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.flags).__module__}.{type(module_4.Pattern.flags).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.groups).__module__}.{type(module_4.Pattern.groups).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.groupindex).__module__}.{type(module_4.Pattern.groupindex).__qualname__}' == 'builtins.getset_descriptor'
    var_82 = module_2.String()
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
    var_83 = module_0.to_json_schema(var_82)
    var_84 = module_2.Integer(exclusive_minimum=var_4, exclusive_maximum=var_12)
    assert f'{type(var_84).__module__}.{type(var_84).__qualname__}' == 'typesystem.fields.Integer'
    assert var_84.title == ''
    assert var_84.description == ''
    assert var_84.allow_null is False
    assert var_84.read_only is False
    assert var_84.minimum is None
    assert var_84.maximum is None
    assert var_84.exclusive_minimum is False
    assert var_84.exclusive_maximum == 100
    assert var_84.multiple_of is None
    assert var_84.precision is None
    assert var_84.coerce_types is True
    var_85 = module_0.to_json_schema(var_84)
    module_2.Integer(multiple_of=var_14)

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
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
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
    var_10 = module_0.to_json_schema(var_9)
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
    var_13 = module_0.to_json_schema(var_12)
    var_14 = 100
    var_15 = 2
    var_16 = module_2.Integer(minimum=var_4, maximum=var_14, exclusive_minimum=var_4, exclusive_maximum=var_14, multiple_of=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Integer'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum is False
    assert var_16.maximum == 100
    assert var_16.exclusive_minimum is False
    assert var_16.exclusive_maximum == 100
    assert var_16.multiple_of == 2
    assert var_16.precision is None
    assert var_16.coerce_types is True
    var_17 = module_0.to_json_schema(var_16)
    var_18 = True
    var_19 = module_2.Float(minimum=var_4, maximum=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Float'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.minimum is False
    assert var_19.maximum is True
    assert var_19.exclusive_minimum is None
    assert var_19.exclusive_maximum is None
    assert var_19.multiple_of is None
    assert var_19.precision is None
    assert var_19.coerce_types is True
    var_20 = module_0.to_json_schema(var_19)
    var_21 = module_2.Boolean()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_22 = module_0.to_json_schema(var_21)
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
    var_24 = True
    var_25 = module_2.Array(var_23, min_items=var_18, max_items=var_6, unique_items=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Array'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert f'{type(var_25.items).__module__}.{type(var_25.items).__qualname__}' == 'typesystem.fields.String'
    assert var_25.additional_items is False
    assert var_25.min_items is True
    assert var_25.max_items == 10
    assert var_25.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_26 = module_0.to_json_schema(var_25)
    var_27 = module_2.String()
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
    var_28 = module_2.Integer()
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
    var_30 = module_2.Array(var_29, var_4)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Array'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.items).__module__}.{type(var_30.items).__qualname__}' == 'builtins.list'
    assert len(var_30.items) == 2
    assert var_30.additional_items is False
    assert var_30.min_items == 2
    assert var_30.max_items == 2
    assert var_30.unique_items is False
    var_31 = 'items'
    with pytest.raises(TypeError):
        var_32 = var_25[var_31]

def test_case_49():
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
    assert module_0.TYPE_CONSTRAINTS == {'minimum', 'exclusiveMinimum', 'dependencies', 'multipleOf', 'contains', 'patternProperties', 'required', 'minItems', 'maximum', 'maxLength', 'minLength', 'boolean_schema', 'maxProperties', 'properties', 'maxItems', 'propertyNames', 'minProperties', 'additionalItems', 'items', 'exclusiveMaximum', 'type', 'uniqueItems', 'additionalProperties', 'pattern'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_5.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = module_2.String(max_length=var_6, min_length=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length == 10
    assert var_7.min_length == 1
    assert var_7.format is None
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_0.to_json_schema(var_7)
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
    var_11 = module_0.to_json_schema(var_10)
    var_12 = 100
    var_13 = module_2.Integer(minimum=var_4, maximum=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum is False
    assert var_13.maximum == 100
    assert var_13.exclusive_minimum is None
    assert var_13.exclusive_maximum is None
    assert var_13.multiple_of is None
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = module_0.to_json_schema(var_13)
    var_15 = module_2.Float(minimum=var_4, maximum=var_9)
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
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
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
    var_22 = module_2.Integer()
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
    var_23 = 'name'
    var_24 = module_2.String()
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
    var_25 = '\\d+'
    var_26 = module_2.Integer()
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
    var_27 = {var_25: var_26}
    var_28 = module_2.Object(pattern_properties=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Object'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.properties == {}
    assert f'{type(var_28.pattern_properties).__module__}.{type(var_28.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_28.pattern_properties) == 1
    assert var_28.additional_properties is True
    assert var_28.property_names is None
    assert var_28.min_properties is None
    assert var_28.max_properties is None
    assert var_28.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = 'A'
    var_31 = (var_30, var_30)
    var_32 = 'B'
    var_33 = (var_32, var_32)
    var_34 = [var_31, var_33]
    var_35 = module_2.Choice(choices=var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Choice'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.choices == [('A', 'A'), ('B', 'B')]
    assert var_35.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_36 = module_0.to_json_schema(var_35)
    var_37 = 'fixed_value'
    var_38 = module_2.Const(var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Const'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_39 = module_0.to_json_schema(var_38)
    var_40 = module_2.String()
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
    var_41 = module_2.Integer()
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
    var_42 = [var_40, var_41]
    var_43 = module_2.Union(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Union'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert f'{type(var_43.any_of).__module__}.{type(var_43.any_of).__qualname__}' == 'builtins.list'
    assert len(var_43.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_44 = module_0.to_json_schema(var_43)
    var_45 = 'anyOf'
    var_46 = var_44[var_45]
    var_47 = len(var_46)
    assert var_47 == 2
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
    var_49 = module_2.Integer()
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
    var_51 = module_5.OneOf(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.one_of).__module__}.{type(var_51.one_of).__qualname__}' == 'builtins.list'
    assert len(var_51.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_52 = module_0.to_json_schema(var_51)
    var_53 = 'oneOf'
    var_54 = var_52[var_53]
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = module_2.String(min_length=var_17)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.String'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert var_56.allow_blank is False
    assert var_56.trim_whitespace is True
    assert var_56.max_length is None
    assert var_56.min_length is True
    assert var_56.format is None
    assert var_56.coerce_types is True
    assert var_56.pattern is None
    assert var_56.pattern_regex is None
    var_57 = module_2.String(max_length=var_6)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.String'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert var_57.allow_blank is False
    assert var_57.trim_whitespace is True
    assert var_57.max_length == 10
    assert var_57.min_length is None
    assert var_57.format is None
    assert var_57.coerce_types is True
    assert var_57.pattern is None
    assert var_57.pattern_regex is None
    var_58 = [var_56, var_57]
    var_59 = module_5.AllOf(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.all_of).__module__}.{type(var_59.all_of).__qualname__}' == 'builtins.list'
    assert len(var_59.all_of) == 2
    var_60 = module_0.to_json_schema(var_59)
    var_61 = 'allOf'
    var_62 = var_60[var_61]
    var_63 = len(var_62)
    assert var_63 == 2
    var_64 = module_2.String()
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
    var_65 = module_2.Integer()
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Integer'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.minimum is None
    assert var_65.maximum is None
    assert var_65.exclusive_minimum is None
    assert var_65.exclusive_maximum is None
    assert var_65.multiple_of is None
    assert var_65.precision is None
    assert var_65.coerce_types is True
    var_66 = module_2.Boolean()
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert var_66.coerce_types is True
    var_67 = module_5.IfThenElse(var_64, var_65, var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    assert f'{type(var_67.if_clause).__module__}.{type(var_67.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_67.then_clause).__module__}.{type(var_67.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_67.else_clause).__module__}.{type(var_67.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_68 = module_0.to_json_schema(var_67)
    var_69 = module_2.String()
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
    assert var_69.pattern is None
    assert var_69.pattern_regex is None
    var_70 = module_5.Not(var_69)
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.composites.Not'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert f'{type(var_70.negated).__module__}.{type(var_70.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_71 = module_0.to_json_schema(var_70)
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
    var_73 = module_2.String()
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
    var_74 = {var_23: var_73}
    var_75 = module_2.Object(properties=var_74)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.fields.Object'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert f'{type(var_75.properties).__module__}.{type(var_75.properties).__qualname__}' == 'builtins.dict'
    assert len(var_75.properties) == 1
    assert var_75.pattern_properties == {}
    assert var_75.additional_properties is True
    assert var_75.property_names is None
    assert var_75.min_properties is None
    assert var_75.max_properties is None
    assert var_75.required == []
    var_76 = 'street'
    var_77 = module_2.String()
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.String'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert var_77.allow_blank is False
    assert var_77.trim_whitespace is True
    assert var_77.max_length is None
    assert var_77.min_length is None
    assert var_77.format is None
    assert var_77.coerce_types is True
    assert var_77.pattern is None
    assert var_77.pattern_regex is None
    var_78 = {var_76: var_77}
    var_79 = module_2.Object(properties=var_78)
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.fields.Object'
    assert var_79.title == ''
    assert var_79.description == ''
    assert var_79.allow_null is False
    assert var_79.read_only is False
    assert f'{type(var_79.properties).__module__}.{type(var_79.properties).__qualname__}' == 'builtins.dict'
    assert len(var_79.properties) == 1
    assert var_79.pattern_properties == {}
    assert var_79.additional_properties is True
    assert var_79.property_names is None
    assert var_79.min_properties is None
    assert var_79.max_properties is None
    assert var_79.required == []
    var_80 = 'default_value'
    var_81 = module_2.String()
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'typesystem.fields.String'
    assert var_81.title == ''
    assert var_81.description == ''
    assert var_81.allow_null is False
    assert var_81.read_only is False
    assert var_81.allow_blank is False
    assert var_81.trim_whitespace is True
    assert var_81.max_length is None
    assert var_81.min_length is None
    assert var_81.format is None
    assert var_81.coerce_types is True
    assert var_81.pattern is None
    assert var_81.pattern_regex is None
    var_82 = module_0.to_json_schema(var_81)
    var_83 = module_2.String()
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
    var_84 = {var_23: var_83}
    var_85 = [var_23]
    var_86 = module_3.Schema(var_84)
    assert f'{type(var_86).__module__}.{type(var_86).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_86.title == ''
    assert var_86.description == ''
    assert var_86.allow_null is False
    assert var_86.read_only is False
    assert f'{type(var_86.fields).__module__}.{type(var_86.fields).__qualname__}' == 'builtins.dict'
    assert len(var_86.fields) == 1
    assert var_86.required == ['name']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_87 = module_0.to_json_schema(var_86)
    var_88 = '^\\d+$'
    var_89 = module_4.compile(var_88)
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 're.Pattern'
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
    assert f'{type(module_4.Pattern.pattern).__module__}.{type(module_4.Pattern.pattern).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.flags).__module__}.{type(module_4.Pattern.flags).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.groups).__module__}.{type(module_4.Pattern.groups).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.groupindex).__module__}.{type(module_4.Pattern.groupindex).__qualname__}' == 'builtins.getset_descriptor'
    var_90 = module_2.String()
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
    var_91 = module_0.to_json_schema(var_90)
    var_92 = True
    var_93 = module_2.Array(unique_items=var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.Array'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert var_93.items is None
    assert var_93.additional_items is False
    assert var_93.min_items is None
    assert var_93.max_items is None
    assert var_93.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_94 = module_0.to_json_schema(var_93)
    var_95 = module_2.Object(additional_properties=var_4)
    assert f'{type(var_95).__module__}.{type(var_95).__qualname__}' == 'typesystem.fields.Object'
    assert var_95.title == ''
    assert var_95.description == ''
    assert var_95.allow_null is False
    assert var_95.read_only is False
    assert var_95.properties == {}
    assert var_95.pattern_properties == {}
    assert var_95.additional_properties is False
    assert var_95.property_names is None
    assert var_95.min_properties is None
    assert var_95.max_properties is None
    assert var_95.required == []
    var_96 = module_0.to_json_schema(var_95)
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
    var_98 = module_2.Object(additional_properties=var_97)
    assert f'{type(var_98).__module__}.{type(var_98).__qualname__}' == 'typesystem.fields.Object'
    assert var_98.title == ''
    assert var_98.description == ''
    assert var_98.allow_null is False
    assert var_98.read_only is False
    assert var_98.properties == {}
    assert var_98.pattern_properties == {}
    assert f'{type(var_98.additional_properties).__module__}.{type(var_98.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_98.property_names is None
    assert var_98.min_properties is None
    assert var_98.max_properties is None
    assert var_98.required == []
    var_99 = module_0.to_json_schema(var_98)
    var_100 = 'additionalProperties'
    var_101 = var_99[var_100]
    var_102 = module_2.Object(property_names=var_83)
    assert f'{type(var_102).__module__}.{type(var_102).__qualname__}' == 'typesystem.fields.Object'
    assert var_102.title == ''
    assert var_102.description == ''
    assert var_102.allow_null is False
    assert var_102.read_only is False
    assert var_102.properties == {}
    assert var_102.pattern_properties == {}
    assert var_102.additional_properties is True
    assert f'{type(var_102.property_names).__module__}.{type(var_102.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_102.min_properties is None
    assert var_102.max_properties is None
    assert var_102.required == []
    var_103 = module_0.to_json_schema(var_102)
    var_104 = module_2.Decimal()
    assert f'{type(var_104).__module__}.{type(var_104).__qualname__}' == 'typesystem.fields.Decimal'
    assert var_104.title == ''
    assert var_104.description == ''
    assert var_104.allow_null is False
    assert var_104.read_only is False
    assert var_104.minimum is None
    assert var_104.maximum is None
    assert var_104.exclusive_minimum is None
    assert var_104.exclusive_maximum is None
    assert var_104.multiple_of is None
    assert var_104.precision is None
    assert var_104.coerce_types is True
    var_105 = module_0.to_json_schema(var_104)