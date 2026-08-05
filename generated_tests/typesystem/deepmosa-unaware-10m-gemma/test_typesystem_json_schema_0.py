# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import enum as module_1
import typesystem.fields as module_2
import re as module_3
import typesystem.composites as module_4
import typesystem.schemas as module_5

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.from_json_schema(var_0)

def test_case_1():
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
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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

def test_case_2():
    var_0 = None
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_3.purge()
    assert module_3.ASCII == module_3.RegexFlag.ASCII
    assert module_3.A == module_3.RegexFlag.ASCII
    assert module_3.IGNORECASE == module_3.RegexFlag.IGNORECASE
    assert module_3.I == module_3.RegexFlag.IGNORECASE
    assert module_3.LOCALE == module_3.RegexFlag.LOCALE
    assert module_3.L == module_3.RegexFlag.LOCALE
    assert module_3.UNICODE == module_3.RegexFlag.UNICODE
    assert module_3.U == module_3.RegexFlag.UNICODE
    assert module_3.MULTILINE == module_3.RegexFlag.MULTILINE
    assert module_3.M == module_3.RegexFlag.MULTILINE
    assert module_3.DOTALL == module_3.RegexFlag.DOTALL
    assert module_3.S == module_3.RegexFlag.DOTALL
    assert module_3.VERBOSE == module_3.RegexFlag.VERBOSE
    assert module_3.X == module_3.RegexFlag.VERBOSE
    assert module_3.TEMPLATE == module_3.RegexFlag.TEMPLATE
    assert module_3.T == module_3.RegexFlag.TEMPLATE
    assert module_3.DEBUG == module_3.RegexFlag.DEBUG
    module_0.to_json_schema(var_0)

def test_case_4():
    pass

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
    module_0.any_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

def test_case_10():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_4.NeverMatch.errors == {'never': 'This never validates.'}

def test_case_11():
    var_0 = {}
    var_1 = module_4.OneOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.one_of == {}
    assert module_4.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_2 = module_4.Not(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.composites.OneOf'
    assert module_4.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    assert f'{type(var_4.negated).__module__}.{type(var_4.negated).__qualname__}' == 'typesystem.composites.OneOf'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    module_0.to_json_schema(var_0, var_0)

def test_case_13():
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
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    var_2 = module_0.to_json_schema(var_1, var_0)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7

def test_case_14():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
def test_case_15():
    var_0 = {}
    var_1 = None
    var_2 = module_4.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_4.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.from_json_schema(var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_0.to_json_schema(var_2, var_0)
    assert var_4 is False
    module_0.to_json_schema(var_1, var_1)

def test_case_16():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    var_0 = '_vh25 6g<pv0.\nxE'
    var_1 = module_2.Any(description=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == '_vh25 6g<pv0.\nxE'
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_2 = module_4.IfThenElse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    var_0 = {}
    var_1 = None
    var_2 = None
    var_3 = module_2.Const(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Const'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.const is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_4 = module_0.from_json_schema(var_0, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = None
    var_6 = module_0.to_json_schema(var_3, var_5)
    var_7 = module_0.from_json_schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Const'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.const is None
    module_0.to_json_schema(var_1, var_1)

def test_case_20():
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
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    var_2 = module_0.to_json_schema(var_1, var_0)
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

def test_case_21():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_5.Schema(var_0, **var_0)
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
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    var_0 = {}
    var_1 = None
    var_2 = module_4.AllOf(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.all_of == {}
    var_3 = module_3.purge()
    assert module_3.ASCII == module_3.RegexFlag.ASCII
    assert module_3.A == module_3.RegexFlag.ASCII
    assert module_3.IGNORECASE == module_3.RegexFlag.IGNORECASE
    assert module_3.I == module_3.RegexFlag.IGNORECASE
    assert module_3.LOCALE == module_3.RegexFlag.LOCALE
    assert module_3.L == module_3.RegexFlag.LOCALE
    assert module_3.UNICODE == module_3.RegexFlag.UNICODE
    assert module_3.U == module_3.RegexFlag.UNICODE
    assert module_3.MULTILINE == module_3.RegexFlag.MULTILINE
    assert module_3.M == module_3.RegexFlag.MULTILINE
    assert module_3.DOTALL == module_3.RegexFlag.DOTALL
    assert module_3.S == module_3.RegexFlag.DOTALL
    assert module_3.VERBOSE == module_3.RegexFlag.VERBOSE
    assert module_3.X == module_3.RegexFlag.VERBOSE
    assert module_3.TEMPLATE == module_3.RegexFlag.TEMPLATE
    assert module_3.T == module_3.RegexFlag.TEMPLATE
    assert module_3.DEBUG == module_3.RegexFlag.DEBUG
    var_4 = module_0.to_json_schema(var_2, var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.all_of == []
    module_0.to_json_schema(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = {}
    var_1 = None
    var_2 = module_4.AllOf(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.all_of == {}
    var_3 = module_3.purge()
    assert module_3.ASCII == module_3.RegexFlag.ASCII
    assert module_3.A == module_3.RegexFlag.ASCII
    assert module_3.IGNORECASE == module_3.RegexFlag.IGNORECASE
    assert module_3.I == module_3.RegexFlag.IGNORECASE
    assert module_3.LOCALE == module_3.RegexFlag.LOCALE
    assert module_3.L == module_3.RegexFlag.LOCALE
    assert module_3.UNICODE == module_3.RegexFlag.UNICODE
    assert module_3.U == module_3.RegexFlag.UNICODE
    assert module_3.MULTILINE == module_3.RegexFlag.MULTILINE
    assert module_3.M == module_3.RegexFlag.MULTILINE
    assert module_3.DOTALL == module_3.RegexFlag.DOTALL
    assert module_3.S == module_3.RegexFlag.DOTALL
    assert module_3.VERBOSE == module_3.RegexFlag.VERBOSE
    assert module_3.X == module_3.RegexFlag.VERBOSE
    assert module_3.TEMPLATE == module_3.RegexFlag.TEMPLATE
    assert module_3.T == module_3.RegexFlag.TEMPLATE
    assert module_3.DEBUG == module_3.RegexFlag.DEBUG
    var_4 = module_0.to_json_schema(var_2, var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    module_0.to_json_schema(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = True
    var_1 = module_2.Integer(minimum=var_0, maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is True
    assert var_1.maximum is True
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
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_2.String(min_length=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is True
    assert var_3.format is None
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_2.Array(var_3, min_items=var_0, unique_items=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.items).__module__}.{type(var_4.items).__qualname__}' == 'typesystem.fields.String'
    assert var_4.additional_items is False
    assert var_4.min_items is True
    assert var_4.max_items is None
    assert var_4.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_5 = module_0.to_json_schema(var_4)
    var_6 = module_2.String()
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
    var_8 = module_2.Integer()
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
    assert var_8.coerce_types is True
    var_9 = [var_7, var_8]
    var_10 = module_2.Union(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Union'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.any_of).__module__}.{type(var_10.any_of).__qualname__}' == 'builtins.list'
    assert len(var_10.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_11 = module_0.to_json_schema(var_10)
    var_12 = 'anyOf'
    var_13 = var_11[var_12]
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = module_2.String()
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
    var_16 = module_2.Integer()
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
    var_17 = module_4.AllOf(var_13)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.all_of == [{'type': 'string', 'minLength': 1}, {'type': 'integer'}]
    module_0.to_json_schema(var_17)

@pytest.mark.xfail(strict=True)
def test_case_25():
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
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    var_3 = module_0.to_json_schema(var_2, var_1)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_4 = module_0.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5
    var_5 = var_4.has_default()
    assert var_5 is True
    var_6 = module_4.IfThenElse(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.if_clause).__module__}.{type(var_6.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_6.then_clause).__module__}.{type(var_6.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_6.else_clause).__module__}.{type(var_6.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_7 = module_2.Object(properties=var_1, property_names=var_2, **var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Object'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.properties == {}
    assert var_7.pattern_properties == {}
    assert var_7.additional_properties is True
    assert f'{type(var_7.property_names).__module__}.{type(var_7.property_names).__qualname__}' == 'typesystem.fields.Union'
    assert var_7.min_properties is None
    assert var_7.max_properties is None
    assert var_7.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_8 = module_5.Reference(var_1, var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.to is None
    assert var_8.definitions is None
    assert module_5.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_5.Reference.target).__module__}.{type(module_5.Reference.target).__qualname__}' == 'builtins.property'
    var_9 = var_2.__or__(var_6)
    assert len(var_2.any_of) == 6
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Union'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.any_of).__module__}.{type(var_9.any_of).__qualname__}' == 'builtins.list'
    assert len(var_9.any_of) == 6
    module_0.to_json_schema(var_8)

def test_case_26():
    var_0 = {}
    var_1 = True
    var_2 = None
    var_3 = module_2.Object(min_properties=var_1, max_properties=var_2, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Object'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.properties == {}
    assert var_3.pattern_properties == {}
    assert var_3.additional_properties is True
    assert var_3.property_names is None
    assert var_3.min_properties is True
    assert var_3.max_properties is None
    assert var_3.required == []
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_27():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_2.Object(properties=var_0, required=var_0)
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
    assert f'{type(var_1.required).__module__}.{type(var_1.required).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.required) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
def test_case_28():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = '#/definitions/MySchema'
    var_3 = {var_1: var_2}
    var_4 = module_0.ref_from_json_schema(var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == '#/definitions/MySchema'
    assert f'{type(var_4.definitions).__module__}.{type(var_4.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    var_5 = '#/components/schemas/User'
    var_6 = {var_1: var_5}
    var_7 = module_0.ref_from_json_schema(var_6, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.to == '#/components/schemas/User'
    assert f'{type(var_7.definitions).__module__}.{type(var_7.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_7.definitions) == 0
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    module_0.ref_from_json_schema(var_10, var_0)

def test_case_29():
    var_0 = '$ref'
    var_1 = 'https://example.com/schema.json'
    var_2 = {var_0: var_1}
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_2, var_2)

def test_case_30():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'enum'
    var_2 = 'default'
    var_3 = 'apple'
    var_4 = 'banana'
    var_5 = 'cherry'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_1: var_6, var_2: var_3}
    var_8 = module_0.enum_from_json_schema(var_7, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Choice'
    assert var_8.default == 'apple'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.choices == [('apple', 'apple'), ('banana', 'banana'), ('cherry', 'cherry')]
    assert var_8.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    var_9 = 1
    var_10 = 2
    var_11 = 'a'
    var_12 = True
    var_13 = [var_9, var_11, var_12]
    var_14 = {var_1: var_13}
    var_15 = module_0.enum_from_json_schema(var_14, var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Choice'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.choices == [(1, 1), ('a', 'a'), (True, True)]
    assert var_15.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_16 = 'id'
    var_17 = {var_16: var_10}
    var_18 = [var_7, var_17]
    var_19 = {var_1: var_18}
    var_20 = module_0.enum_from_json_schema(var_19, var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Choice'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.choices == [({'enum': ['apple', 'banana', 'cherry'], 'default': 'apple'}, {'enum': ['apple', 'banana', 'cherry'], 'default': 'apple'}), ({'id': 2}, {'id': 2})]
    assert var_20.coerce_types is True

def test_case_31():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'allOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = module_0.all_of_from_json_schema(var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.all_of).__module__}.{type(var_9.all_of).__qualname__}' == 'builtins.list'
    assert len(var_9.all_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.all_of[var_12]
    var_14 = 1
    var_15 = var_9.all_of[var_14]
    var_16 = 'default'
    var_17 = {var_2: var_3}
    var_18 = [var_17]
    var_19 = 'hello'
    var_20 = {var_1: var_18, var_16: var_19}
    var_21 = module_0.all_of_from_json_schema(var_20, var_0)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_21.default == 'hello'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.all_of).__module__}.{type(var_21.all_of).__qualname__}' == 'builtins.list'
    assert len(var_21.all_of) == 1
    var_22 = {var_2: var_5}
    var_23 = 'minimum'
    var_24 = 10
    var_25 = {var_23: var_24}
    var_26 = [var_22, var_25]
    var_27 = {var_1: var_26}
    var_28 = 'number'
    var_29 = {var_2: var_28}
    var_30 = [var_27, var_29]
    var_31 = {var_1: var_30}
    var_32 = module_0.all_of_from_json_schema(var_31, var_0)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.all_of).__module__}.{type(var_32.all_of).__qualname__}' == 'builtins.list'
    assert len(var_32.all_of) == 2
    var_33 = var_32.all_of
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = var_32.all_of[var_12]
    var_36 = var_32.all_of[var_14]
    var_37 = []
    var_38 = {var_1: var_37}
    var_39 = module_0.all_of_from_json_schema(var_38, var_0)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.all_of == []
    var_40 = var_39.all_of
    var_41 = len(var_40)
    assert var_41 == 0

def test_case_32():
    var_0 = False
    var_1 = 10
    var_2 = '\x0bhvH^o:v?jm:HsNoZ'
    var_3 = module_2.String(max_length=var_1, min_length=var_1, format=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length == 10
    assert var_3.min_length == 10
    assert var_3.format == '\x0bhvH^o:v?jm:HsNoZ'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_0.from_json_schema(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_4.NeverMatch.errors == {'never': 'This never validates.'}
    var_5 = module_0.get_standard_properties(var_4)
    var_6 = module_2.Boolean()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_7 = module_2.String(min_length=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is False
    assert var_7.format is None
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    var_8 = module_2.Array(var_7, min_items=var_0, unique_items=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Array'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.items).__module__}.{type(var_8.items).__qualname__}' == 'typesystem.fields.String'
    assert var_8.additional_items is False
    assert var_8.min_items is False
    assert var_8.max_items is None
    assert var_8.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_9 = 'name'
    var_10 = {var_9: var_3}
    var_11 = [var_9]
    var_12 = module_2.Object(properties=var_10, required=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Object'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.properties).__module__}.{type(var_12.properties).__qualname__}' == 'builtins.dict'
    assert len(var_12.properties) == 1
    assert var_12.pattern_properties == {}
    assert var_12.additional_properties is True
    assert var_12.property_names is None
    assert var_12.min_properties is None
    assert var_12.max_properties is None
    assert var_12.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_13 = module_0.to_json_schema(var_12)
    var_14 = var_8.has_default()
    assert var_14 is False
    var_15 = module_0.type_from_json_schema(var_13, var_9)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Object'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.properties).__module__}.{type(var_15.properties).__qualname__}' == 'builtins.dict'
    assert len(var_15.properties) == 1
    assert var_15.pattern_properties == {}
    assert var_15.additional_properties is True
    assert var_15.property_names is None
    assert var_15.min_properties is None
    assert var_15.max_properties is None
    assert var_15.required == ['name']
    var_16 = module_2.Integer()
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

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = True
    var_1 = 5
    var_2 = 10
    var_3 = 'email'
    var_4 = module_2.String(max_length=var_2, min_length=var_1, format=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length == 10
    assert var_4.min_length == 5
    assert var_4.format == 'email'
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_6 = False
    var_7 = 100
    var_8 = module_2.Integer(minimum=var_6, maximum=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Integer'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.minimum is False
    assert var_8.maximum == 100
    assert var_8.exclusive_minimum is None
    assert var_8.exclusive_maximum is None
    assert var_8.multiple_of is None
    assert var_8.precision is None
    assert var_8.coerce_types is True
    var_9 = module_2.Boolean()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_2.String(min_length=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.String'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.allow_blank is False
    assert var_11.trim_whitespace is True
    assert var_11.max_length is None
    assert var_11.min_length is True
    assert var_11.format is None
    assert var_11.coerce_types is True
    assert var_11.pattern is None
    assert var_11.pattern_regex is None
    var_12 = module_2.Array(var_11, min_items=var_0, unique_items=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Array'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.items).__module__}.{type(var_12.items).__qualname__}' == 'typesystem.fields.String'
    assert var_12.additional_items is False
    assert var_12.min_items is True
    assert var_12.max_items is None
    assert var_12.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_13 = module_0.to_json_schema(var_12)
    var_14 = 'name'
    var_15 = {var_14: var_4}
    var_16 = [var_14]
    var_17 = module_2.Object(properties=var_15, required=var_16)
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
    assert var_17.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_18 = module_0.to_json_schema(var_17)
    var_19 = 'red'
    var_20 = (var_19, var_19)
    var_21 = (var_19, var_19)
    var_22 = [var_20, var_21]
    var_23 = module_2.Choice(choices=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Choice'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.choices == [('red', 'red'), ('red', 'red')]
    assert var_23.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
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
    var_28 = module_2.Union(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Union'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.any_of).__module__}.{type(var_28.any_of).__qualname__}' == 'builtins.list'
    assert len(var_28.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = 'anyOf'
    var_31 = var_29[var_30]
    var_32 = len(var_31)
    assert var_32 == 2
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
    var_34 = module_2.Integer()
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Integer'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.minimum is None
    assert var_34.maximum is None
    assert var_34.exclusive_minimum is None
    assert var_34.exclusive_maximum is None
    assert var_34.multiple_of is None
    assert var_34.precision is None
    assert var_34.coerce_types is True
    module_0.to_json_schema(var_31)

def test_case_34():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'integer'
    var_6 = {var_4: var_5}
    var_7 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = {var_1: var_7, var_2: var_6, var_3: var_3}
    var_9 = None
    var_10 = module_0.type_from_json_schema(var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Union'
    assert var_10.default is None
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is True
    assert var_10.read_only is False
    assert f'{type(var_10.any_of).__module__}.{type(var_10.any_of).__qualname__}' == 'builtins.list'
    assert len(var_10.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_11 = module_2.Boolean()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_12 = {var_4: var_5}
    var_13 = {var_1: var_6, var_2: var_12}
    var_14 = module_0.if_then_else_from_json_schema(var_13, var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.if_clause).__module__}.{type(var_14.if_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_14.then_clause).__module__}.{type(var_14.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_14.else_clause).__module__}.{type(var_14.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_15 = module_2.String()
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_16 = module_0.to_json_schema(var_10, var_9)
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
    var_18 = module_0.from_json_schema(var_16)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Union'
    assert var_18.default is None
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert f'{type(var_18.any_of).__module__}.{type(var_18.any_of).__qualname__}' == 'builtins.list'
    assert len(var_18.any_of) == 5
    var_19 = module_0.to_json_schema(var_14)

def test_case_35():
    var_0 = True
    var_1 = module_2.String(min_length=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is True
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_2.Array(var_1, min_items=var_0, unique_items=var_0)
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
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    var_6 = module_0.from_json_schema(var_3, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.items).__module__}.{type(var_6.items).__qualname__}' == 'typesystem.fields.String'
    assert var_6.additional_items is False
    assert var_6.min_items is True
    assert var_6.max_items is None
    assert var_6.unique_items is True

def test_case_36():
    var_0 = '.*'
    var_1 = False
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = {var_0: var_2}
    var_4 = module_2.Object(pattern_properties=var_3, additional_properties=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Object'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.properties == {}
    assert f'{type(var_4.pattern_properties).__module__}.{type(var_4.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_4.pattern_properties) == 1
    assert var_4.additional_properties is False
    assert var_4.property_names is None
    assert var_4.min_properties is None
    assert var_4.max_properties is None
    assert var_4.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_37():
    var_0 = False
    var_1 = module_2.String()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_2.Integer()
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
    var_3 = [var_1, var_2]
    var_4 = True
    var_5 = module_2.String()
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
    var_6 = module_2.Array(var_3, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.items).__module__}.{type(var_6.items).__qualname__}' == 'builtins.list'
    assert len(var_6.items) == 2
    assert f'{type(var_6.additional_items).__module__}.{type(var_6.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_6.min_items == 2
    assert var_6.max_items is None
    assert var_6.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_38():
    var_0 = True
    var_1 = module_2.Integer(minimum=var_0, maximum=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is True
    assert var_1.maximum is True
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
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_2.Boolean()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_4 = module_2.String(min_length=var_0)
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
    assert var_4.coerce_types is True
    assert var_4.pattern is None
    assert var_4.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_5 = module_2.Array(var_4, min_items=var_0, unique_items=var_0)
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
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_6 = module_0.to_json_schema(var_5)
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
    var_8 = [var_4, var_7]
    var_9 = module_2.Union(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Union'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.any_of).__module__}.{type(var_9.any_of).__qualname__}' == 'builtins.list'
    assert len(var_9.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_10 = module_0.to_json_schema(var_9)
    with pytest.raises(KeyError):
        var_11 = var_10[var_4]

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = {}
    var_1 = 2122
    var_2 = "'itWZ3s+e\nNm3y(lR"
    var_3 = module_2.Float(maximum=var_1, precision=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Float'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.minimum is None
    assert var_3.maximum == 2122
    assert var_3.exclusive_minimum is None
    assert var_3.exclusive_maximum is None
    assert var_3.multiple_of is None
    assert var_3.precision == "'itWZ3s+e\nNm3y(lR"
    assert var_3.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_4 = [var_3, var_3]
    var_5 = module_4.OneOf(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.one_of).__module__}.{type(var_5.one_of).__qualname__}' == 'builtins.list'
    assert len(var_5.one_of) == 2
    assert module_4.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_6 = module_0.from_json_schema(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Any'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = module_4.Not(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.Not'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.negated).__module__}.{type(var_7.negated).__qualname__}' == 'typesystem.composites.OneOf'
    assert module_4.Not.errors == {'negated': 'Must not match.'}
    var_8 = None
    var_9 = module_0.to_json_schema(var_5)
    module_3.match(var_8, var_8, var_8)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = True
    var_1 = -3379
    var_2 = 'ZvOg=qWC'
    var_3 = module_2.String(max_length=var_1, min_length=var_1, format=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length == -3379
    assert var_3.min_length == -3379
    assert var_3.format == 'ZvOg=qWC'
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_0.get_standard_properties(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_2.String(min_length=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length is None
    assert var_5.min_length is True
    assert var_5.format is None
    assert var_5.coerce_types is True
    assert var_5.pattern is None
    assert var_5.pattern_regex is None
    var_6 = 'name'
    var_7 = [var_6]
    var_8 = module_2.Object(properties=var_4, required=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Object'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.properties == {}
    assert var_8.pattern_properties == {}
    assert var_8.additional_properties is True
    assert var_8.property_names is None
    assert var_8.min_properties is None
    assert var_8.max_properties is None
    assert var_8.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = module_0.type_from_json_schema(var_9, var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Object'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.properties == {}
    assert var_10.pattern_properties == {}
    assert var_10.additional_properties is True
    assert var_10.property_names is None
    assert var_10.min_properties is None
    assert var_10.max_properties is None
    assert var_10.required == ['name']
    var_5.validate(var_4)

def test_case_41():
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
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    var_3 = module_0.from_json_schema(var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_4 = module_0.to_json_schema(var_2, var_1)
    var_5 = module_0.from_json_schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.any_of).__module__}.{type(var_5.any_of).__qualname__}' == 'builtins.list'
    assert len(var_5.any_of) == 5
    var_6 = var_5.has_default()
    assert var_6 is True
    var_7 = module_4.IfThenElse(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.if_clause).__module__}.{type(var_7.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_7.then_clause).__module__}.{type(var_7.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_7.else_clause).__module__}.{type(var_7.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_8 = module_2.Object(properties=var_1, property_names=var_2, **var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Object'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.properties == {}
    assert var_8.pattern_properties == {}
    assert var_8.additional_properties is True
    assert f'{type(var_8.property_names).__module__}.{type(var_8.property_names).__qualname__}' == 'typesystem.fields.Union'
    assert var_8.min_properties is None
    assert var_8.max_properties is None
    assert var_8.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_9 = module_4.IfThenElse(var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.if_clause is None
    assert f'{type(var_9.then_clause).__module__}.{type(var_9.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_9.else_clause).__module__}.{type(var_9.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_10 = module_0.to_json_schema(var_8)

def test_case_42():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'oneOf'
    var_2 = 'default'
    var_3 = 'string'
    var_4 = {var_1: var_3}
    var_5 = [var_4, var_4]
    var_6 = 'some_default'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = module_0.one_of_from_json_schema(var_7, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_8.default == 'some_default'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.one_of).__module__}.{type(var_8.one_of).__qualname__}' == 'builtins.list'
    assert len(var_8.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_4.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_9 = 0
    var_10 = var_8.one_of[var_9]
    var_11 = 'items'
    var_12 = {var_3: var_3}
    var_13 = {var_10: var_1, var_11: var_12}
    var_14 = 'object'
    var_15 = 'name'
    var_16 = {var_2: var_3}
    var_17 = {var_15: var_16}
    var_18 = {var_11: var_14, var_14: var_17}
    var_19 = [var_13, var_18]
    var_20 = {var_1: var_19}
    var_21 = module_0.one_of_from_json_schema(var_20, var_0)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.one_of).__module__}.{type(var_21.one_of).__qualname__}' == 'builtins.list'
    assert len(var_21.one_of) == 2
    var_22 = var_21.one_of[var_9]
    var_23 = 'boolean'
    var_24 = {var_3: var_23}
    var_25 = [var_24]
    var_26 = {var_1: var_25}
    var_27 = module_0.one_of_from_json_schema(var_26, var_0)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.one_of).__module__}.{type(var_27.one_of).__qualname__}' == 'builtins.list'
    assert len(var_27.one_of) == 1
    var_28 = 'number'
    var_29 = {var_1: var_28}
    var_30 = [var_29]
    var_31 = {var_1: var_30}
    var_32 = module_0.one_of_from_json_schema(var_31, var_0)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.one_of).__module__}.{type(var_32.one_of).__qualname__}' == 'builtins.list'
    assert len(var_32.one_of) == 1
    var_33 = var_32.one_of[var_9]

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'k'
    var_5 = 'boolean'
    var_6 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_8 = {var_4: var_3}
    var_9 = {var_1: var_8, var_2: var_8}
    var_10 = module_0.if_then_else_from_json_schema(var_9, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.if_clause).__module__}.{type(var_10.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_10.then_clause).__module__}.{type(var_10.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_10.else_clause).__module__}.{type(var_10.else_clause).__qualname__}' == 'typesystem.fields.Any'
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
    var_12 = {var_4: var_2}
    var_13 = {var_4: var_5}
    var_14 = {var_1: var_12, var_3: var_13}
    var_15 = module_0.if_then_else_from_json_schema(var_14, var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.if_clause).__module__}.{type(var_15.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_15.then_clause).__module__}.{type(var_15.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_15.else_clause).__module__}.{type(var_15.else_clause).__qualname__}' == 'typesystem.fields.Any'
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    module_0.if_then_else_from_json_schema(var_6, var_0)

def test_case_44():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
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
    assert module_4.NeverMatch.errors == {'never': 'This never validates.'}
    var_4 = 'type'
    var_5 = 'minLength'
    var_6 = 'string'
    var_7 = 5
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.from_json_schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length is None
    assert var_9.min_length == 5
    assert var_9.format is None
    assert var_9.coerce_types is False
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = 'minimum'
    var_11 = 'integer'
    var_12 = 10
    var_13 = {var_4: var_11, var_10: var_12}
    var_14 = module_0.from_json_schema(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum == 10
    assert var_14.maximum is None
    assert var_14.exclusive_minimum is None
    assert var_14.exclusive_maximum is None
    assert var_14.multiple_of is None
    assert var_14.precision is None
    assert var_14.coerce_types is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_15 = 'enum'
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 'c'
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_15: var_19}
    var_21 = module_0.from_json_schema(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Choice'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_21.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_22 = 'const'
    var_23 = 42
    var_24 = {var_22: var_23}
    var_25 = module_0.from_json_schema(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Const'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.const == 42
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_26 = 'items'
    var_27 = 'array'
    var_28 = {var_4: var_6}
    var_29 = {var_4: var_27, var_26: var_28}
    var_30 = module_0.from_json_schema(var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Array'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.items).__module__}.{type(var_30.items).__qualname__}' == 'typesystem.fields.String'
    assert var_30.additional_items is True
    assert var_30.min_items == 0
    assert var_30.max_items is None
    assert var_30.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    with pytest.raises(AttributeError):
        var_31 = var_30.constraints[var_26]

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'else'
    var_2 = None
    var_3 = var_0.__setitem__(var_1, var_2)
    assert len(var_0) == 1
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = 'integer'
    var_7 = {var_4: var_6}
    var_8 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_9 = None
    var_10 = module_0.type_from_json_schema(var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Union'
    assert var_10.default is None
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is True
    assert var_10.read_only is False
    assert f'{type(var_10.any_of).__module__}.{type(var_10.any_of).__qualname__}' == 'builtins.list'
    assert len(var_10.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_11 = module_2.String()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_12 = var_11.validate(var_5)
    assert var_12 == 'string'
    var_13 = module_2.Boolean()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_14 = {var_5: var_4, var_1: var_3, var_4: var_6}
    var_15 = {var_1: var_7, var_2: var_14}
    var_16 = module_0.if_then_else_from_json_schema(var_15, var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert f'{type(var_16.if_clause).__module__}.{type(var_16.if_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_16.then_clause).__module__}.{type(var_16.then_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_16.else_clause).__module__}.{type(var_16.else_clause).__qualname__}' == 'typesystem.fields.Any'
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
    var_18 = module_0.to_json_schema(var_10, var_9)
    module_0.any_of_from_json_schema(var_9, var_0)

def test_case_47():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'type'
    var_2 = 'minimum'
    var_3 = 'maximum'
    var_4 = 'multipleOf'
    var_5 = 'number'
    var_6 = 0
    var_7 = 10
    var_8 = 2
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = False
    var_11 = module_0.from_json_schema_type(var_9, var_5, var_10, var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Float'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.minimum == 0
    assert var_11.maximum == 10
    assert var_11.exclusive_minimum is None
    assert var_11.exclusive_maximum is None
    assert var_11.multiple_of == 2
    assert var_11.precision is None
    assert var_11.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'uniqueItems', 'additionalProperties', 'minItems', 'type', 'pattern', 'multipleOf', 'properties', 'minLength', 'contains', 'minimum', 'dependencies', 'maxProperties', 'maximum', 'minProperties', 'exclusiveMinimum', 'maxItems', 'items', 'additionalItems', 'boolean_schema', 'patternProperties', 'exclusiveMaximum', 'required', 'maxLength', 'propertyNames'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_12 = 'integer'
    var_13 = False
    var_14 = module_0.from_json_schema_type(var_9, var_12, var_13, var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum == 0
    assert var_14.maximum == 10
    assert var_14.exclusive_minimum is None
    assert var_14.exclusive_maximum is None
    assert var_14.multiple_of == 2
    assert var_14.precision is None
    assert var_14.coerce_types is False
    var_15 = 'string'
    var_16 = 5
    var_17 = 'default'
    var_18 = 'boolean'
    var_19 = True
    var_20 = {var_1: var_18, var_17: var_19}
    var_21 = False
    var_22 = module_0.from_json_schema_type(var_20, var_18, var_21, var_0)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_22.default is True
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_23 = 'items'
    var_24 = 'minItems'
    var_25 = 'maxItems'
    var_26 = 'uniqueItems'
    var_27 = 'array'
    var_28 = {var_1: var_15}
    var_29 = True
    var_30 = {var_1: var_27, var_23: var_28, var_24: var_19, var_25: var_16, var_26: var_29}
    var_31 = False
    var_32 = module_0.from_json_schema_type(var_30, var_27, var_31, var_0)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Array'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.items).__module__}.{type(var_32.items).__qualname__}' == 'typesystem.fields.String'
    assert var_32.additional_items is True
    assert var_32.min_items is True
    assert var_32.max_items == 5
    assert var_32.unique_items is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_33 = var_32.items
    var_34 = 'additionalItems'
    var_35 = False
    var_36 = {var_1: var_27, var_34: var_35}
    var_37 = False
    var_38 = module_0.from_json_schema_type(var_36, var_27, var_37, var_0)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Array'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.items is None
    assert var_38.additional_items is False
    assert var_38.min_items == 0
    assert var_38.max_items is None
    assert var_38.unique_items is False
    var_39 = 'properties'
    var_40 = 'required'
    var_41 = 'minProperties'
    var_42 = 'additionalProperties'
    var_43 = 'object'
    var_44 = 'name'
    var_45 = 'age'
    var_46 = {var_1: var_12}
    var_47 = {var_44: var_9, var_45: var_46}
    var_48 = [var_44]
    var_49 = True
    var_50 = {var_1: var_43, var_39: var_47, var_40: var_48, var_41: var_29, var_42: var_49}
    var_51 = False
    var_52 = module_0.from_json_schema_type(var_50, var_43, var_51, var_0)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Object'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert f'{type(var_52.properties).__module__}.{type(var_52.properties).__qualname__}' == 'builtins.dict'
    assert len(var_52.properties) == 2
    assert var_52.pattern_properties == {}
    assert var_52.additional_properties is True
    assert var_52.property_names is None
    assert var_52.min_properties is True
    assert var_52.max_properties is None
    assert var_52.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_53 = var_52.properties[var_44]
    var_54 = var_52.properties[var_45]
    var_55 = 'patternProperties'
    var_56 = '^prop_'
    var_57 = {var_1: var_15}
    var_58 = {var_56: var_57}
    var_59 = {var_1: var_43, var_55: var_58}
    var_60 = False
    var_61 = module_0.from_json_schema_type(var_59, var_43, var_60, var_0)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.fields.Object'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert var_61.properties == {}
    assert f'{type(var_61.pattern_properties).__module__}.{type(var_61.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_61.pattern_properties) == 1
    assert var_61.additional_properties is None
    assert var_61.property_names is None
    assert var_61.min_properties is None
    assert var_61.max_properties is None
    assert var_61.required == []
    var_62 = var_61.pattern_properties[var_56]
    var_63 = 'null'
    var_64 = True
    var_65 = {var_1: var_5, var_63: var_64}
    var_66 = True
    var_67 = module_0.from_json_schema_type(var_65, var_5, var_66, var_0)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.fields.Float'
    assert var_67.default is None
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is True
    assert var_67.read_only is False
    assert var_67.minimum is None
    assert var_67.maximum is None
    assert var_67.exclusive_minimum is None
    assert var_67.exclusive_maximum is None
    assert var_67.multiple_of is None
    assert var_67.precision is None
    assert var_67.coerce_types is False

def test_case_48():
    var_0 = 'else'
    var_1 = 'string'
    var_2 = module_2.Field(title=var_1, description=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.title == 'string'
    assert var_2.description == 'else'
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_2)