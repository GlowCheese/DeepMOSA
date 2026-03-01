# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import enum as module_1
import typesystem.fields as module_2
import re as module_3
import typesystem.schemas as module_4
import typesystem.composites as module_5

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_0 = {}
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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

def test_case_4():
    var_0 = {}
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.type_from_json_schema(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is True
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_4 = module_3.purge()
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
    var_5 = module_0.from_json_schema(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.any_of).__module__}.{type(var_5.any_of).__qualname__}' == 'builtins.list'
    assert len(var_5.any_of) == 5
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_3, var_4, var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = {}
    module_0.to_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
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

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.ref_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.all_of_from_json_schema(var_0, var_0)

def test_case_9():
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
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
def test_case_10():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = {}
    module_0.not_from_json_schema(var_0, var_0)

def test_case_14():
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
    var_1 = module_2.Const(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Const'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.const is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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

def test_case_16():
    var_0 = '.O_.j9?4g?+FO%'
    var_1 = module_2.Field(description=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.title == ''
    assert var_1.description == '.O_.j9?4g?+FO%'
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_1)

def test_case_17():
    var_0 = {}
    var_1 = module_4.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = module_5.IfThenElse(var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = module_0.to_json_schema(var_2, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = {}
    var_1 = module_5.Not(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.Not'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.negated == {}
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    module_0.to_json_schema(var_1)

def test_case_19():
    var_0 = {}
    var_1 = module_4.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_5.OneOf(var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.one_of).__module__}.{type(var_1.one_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_1.one_of) == 0
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.one_of == []

def test_case_21():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_0.to_json_schema(var_2)
    assert var_5 is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7

def test_case_22():
    var_0 = None
    var_1 = {}
    var_2 = module_0.from_json_schema(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is True
    var_4 = {var_0: var_0}
    var_5 = module_0.get_valid_types(var_4)
    var_6 = module_0.type_from_json_schema(var_4, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Union'
    assert var_6.default is None
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is True
    assert var_6.read_only is False
    assert f'{type(var_6.any_of).__module__}.{type(var_6.any_of).__qualname__}' == 'builtins.list'
    assert len(var_6.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_7 = module_2.Integer(exclusive_maximum=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Integer'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.minimum is None
    assert var_7.maximum is None
    assert var_7.exclusive_minimum is None
    assert var_7.exclusive_maximum is True
    assert var_7.multiple_of is None
    assert var_7.precision is None
    assert var_7.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_8 = module_0.to_json_schema(var_7)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = {}
    var_2 = module_5.AllOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.all_of == {}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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

def test_case_24():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_2 = 'type'
    var_3 = 'string'
    var_4 = 'number'
    var_5 = {var_2: var_4}
    var_6 = module_0.from_json_schema(var_5)
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
    assert var_6.coerce_types is False
    var_7 = 'integer'
    var_8 = {var_2: var_7}
    var_9 = module_0.from_json_schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Integer'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.minimum is None
    assert var_9.maximum is None
    assert var_9.exclusive_minimum is None
    assert var_9.exclusive_maximum is None
    assert var_9.multiple_of is None
    assert var_9.precision is None
    assert var_9.coerce_types is False
    var_10 = 'array'
    var_11 = {var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Array'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.items is None
    assert var_12.additional_items is True
    assert var_12.min_items == 0
    assert var_12.max_items is None
    assert var_12.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_13 = 'object'
    var_14 = {var_2: var_13}
    var_15 = module_0.from_json_schema(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Object'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.properties == {}
    assert var_15.pattern_properties == {}
    assert var_15.additional_properties is None
    assert var_15.property_names is None
    assert var_15.min_properties is None
    assert var_15.max_properties is None
    assert var_15.required == []
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 'c'
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_3: var_19}
    var_21 = module_0.from_json_schema(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Any'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    var_22 = 'const'
    var_23 = 'fixed_value'
    var_24 = {var_22: var_23}
    var_25 = module_0.from_json_schema(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Const'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_26 = 'allOf'
    var_27 = {var_2: var_3}
    var_28 = 'minLength'
    var_29 = 5
    var_30 = {var_28: var_29}
    var_31 = [var_27, var_30]
    var_32 = {var_26: var_31}
    var_33 = module_0.from_json_schema(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.all_of).__module__}.{type(var_33.all_of).__qualname__}' == 'builtins.list'
    assert len(var_33.all_of) == 2
    with pytest.raises(AttributeError):
        var_34 = var_33.schemas

def test_case_25():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_8 = 'number'
    var_9 = {var_4: var_8}
    var_10 = module_0.from_json_schema(var_9)
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
    var_13 = module_0.from_json_schema(var_12)
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
    var_14 = 'array'
    var_15 = {var_4: var_14}
    var_16 = module_0.from_json_schema(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Array'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.items is None
    assert var_16.additional_items is True
    assert var_16.min_items == 0
    assert var_16.max_items is None
    assert var_16.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_17 = 'object'
    var_18 = {var_4: var_17}
    var_19 = module_0.from_json_schema(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Object'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.properties == {}
    assert var_19.pattern_properties == {}
    assert var_19.additional_properties is None
    assert var_19.property_names is None
    assert var_19.min_properties is None
    assert var_19.max_properties is None
    assert var_19.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_20 = 'enum'
    var_21 = 'a'
    var_22 = 'b'
    var_23 = 'c'
    var_24 = [var_21, var_22, var_23]
    var_25 = {var_20: var_24}
    var_26 = module_0.from_json_schema(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Choice'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_26.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_27 = 'const'
    var_28 = 'fixed_value'
    var_29 = {var_27: var_28}
    var_30 = module_0.from_json_schema(var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Const'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_31 = 'allOf'
    var_32 = {var_4: var_5}
    var_33 = 'minLength'
    var_34 = 5
    var_35 = {var_33: var_34}
    var_36 = [var_32, var_35]
    var_37 = {var_31: var_36}
    var_38 = module_0.from_json_schema(var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert f'{type(var_38.all_of).__module__}.{type(var_38.all_of).__qualname__}' == 'builtins.list'
    assert len(var_38.all_of) == 2
    with pytest.raises(AttributeError):
        var_39 = var_38.schemas

def test_case_26():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'allOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = 'test'
    var_6 = {var_1: var_5, var_2: var_5}
    var_7 = module_0.all_of_from_json_schema(var_6, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_7.default == 'test'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.all_of).__module__}.{type(var_7.all_of).__qualname__}' == 'builtins.list'
    assert len(var_7.all_of) == 4
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = var_7.all_of
    var_9 = len(var_8)
    var_10 = 0
    var_11 = var_7.all_of[var_10]
    var_12 = 1
    var_13 = var_7.all_of[var_12]
    var_14 = 'properties'
    var_15 = 'object'
    var_16 = 'name'
    var_17 = {var_3: var_4}
    var_18 = {var_16: var_17}
    var_19 = {var_3: var_15, var_14: var_18}
    var_20 = 'age'
    var_21 = 'integer'
    var_22 = {var_3: var_21}
    var_23 = {var_20: var_22}
    var_24 = {var_3: var_15, var_14: var_23}
    var_25 = [var_24]
    var_26 = {var_1: var_25}
    var_27 = [var_19, var_26]
    var_28 = {var_1: var_27}
    var_29 = module_0.all_of_from_json_schema(var_28, var_0)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.all_of).__module__}.{type(var_29.all_of).__qualname__}' == 'builtins.list'
    assert len(var_29.all_of) == 2
    var_30 = var_29.all_of
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = var_29.all_of[var_10]
    var_33 = var_29.all_of[var_12]

def test_case_27():
    var_0 = 10
    var_1 = '\ty`e\tiJf.c'
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == 10
    assert var_2.min_length == 10
    assert var_2.format == '\ty`e\tiJf.c'
    assert var_2.coerce_types is True
    assert var_2.pattern == '\ty`e\tiJf.c'
    assert f'{type(var_2.pattern_regex).__module__}.{type(var_2.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = 0
    var_4 = 2112
    var_5 = module_2.Integer(minimum=var_3, maximum=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Integer'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum == 0
    assert var_5.maximum == 2112
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of is None
    assert var_5.precision is None
    assert var_5.coerce_types is True
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = None
    var_8 = var_2.serialize(var_7)
    var_9 = module_1._EnumDict()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'enum._EnumDict'
    assert len(var_9) == 0
    var_10 = module_2.Boolean()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_11 = module_2.Array(var_2, min_items=var_0, max_items=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Array'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.items).__module__}.{type(var_11.items).__qualname__}' == 'typesystem.fields.String'
    assert var_11.additional_items is False
    assert var_11.min_items == 10
    assert var_11.max_items == 10
    assert var_11.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_12 = module_0.to_json_schema(var_11)
    var_13 = module_0.to_json_schema(var_10, var_8)
    var_14 = module_0.from_json_schema(var_6, var_8)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum == 0
    assert var_14.maximum == 2112
    assert var_14.exclusive_minimum is None
    assert var_14.exclusive_maximum is None
    assert var_14.multiple_of is None
    assert var_14.precision is None
    assert var_14.coerce_types is False
    var_15 = module_0.from_json_schema(var_9)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Any'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False

def test_case_28():
    var_0 = 10
    var_1 = '\tye\tiJ~f.c'
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == 10
    assert var_2.min_length == 10
    assert var_2.format == '\tye\tiJ~f.c'
    assert var_2.coerce_types is True
    assert var_2.pattern == '\tye\tiJ~f.c'
    assert f'{type(var_2.pattern_regex).__module__}.{type(var_2.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_2.Array(var_2, min_items=var_0, max_items=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.String'
    assert var_3.additional_items is False
    assert var_3.min_items == 10
    assert var_3.max_items == 10
    assert var_3.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    assert var_5.min_items == 10
    assert var_5.max_items == 10
    assert var_5.unique_items is False

def test_case_29():
    var_0 = '$ref'
    var_1 = '#/components/schemas/User'
    var_2 = {var_0: var_1}
    var_3 = module_4.Definitions()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 0
    var_4 = module_0.ref_from_json_schema(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == '#/components/schemas/User'
    assert f'{type(var_4.definitions).__module__}.{type(var_4.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_5 = module_4.Definitions()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 0
    var_6 = module_0.ref_from_json_schema(var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.to == '#/components/schemas/User'
    assert f'{type(var_6.definitions).__module__}.{type(var_6.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6.definitions) == 0

def test_case_30():
    var_0 = '$ref'
    var_1 = '#/components/schemas/User'
    var_2 = {var_0: var_1}
    var_3 = module_4.Definitions()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 0
    var_4 = module_0.ref_from_json_schema(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == '#/components/schemas/User'
    assert f'{type(var_4.definitions).__module__}.{type(var_4.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_5 = 'components/schemas/User'
    var_6 = {var_0: var_5}
    var_7 = module_4.Definitions()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_7) == 0
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_6, var_7)

def test_case_31():
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
    var_9 = 'default_value'
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = module_0.one_of_from_json_schema(var_10, var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_11.default == 'default_value'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.one_of).__module__}.{type(var_11.one_of).__qualname__}' == 'builtins.list'
    assert len(var_11.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_12 = var_11.one_of
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 0
    var_15 = var_11.one_of[var_14]
    var_16 = 1
    var_17 = var_11.one_of[var_16]

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'number'
    var_6 = 'boolean'
    var_7 = {var_4: var_6}
    var_8 = {var_1: var_7, var_2: var_7, var_3: var_7}
    var_9 = module_0.if_then_else_from_json_schema(var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.if_clause).__module__}.{type(var_9.if_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_9.then_clause).__module__}.{type(var_9.then_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(var_9.else_clause).__module__}.{type(var_9.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_13 = {var_4: var_5}
    var_14 = {var_1: var_13, var_2: var_13}
    var_15 = module_0.if_then_else_from_json_schema(var_14, var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.if_clause).__module__}.{type(var_15.if_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_15.then_clause).__module__}.{type(var_15.then_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_15.else_clause).__module__}.{type(var_15.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_16 = var_15.if_clause
    var_17 = var_15.then_clause
    var_18 = {var_4: var_3}
    var_19 = {var_1: var_18}
    module_0.if_then_else_from_json_schema(var_19, var_0)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_4.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'string'
    var_5 = {var_2: var_4}
    var_6 = 'number'
    var_7 = {var_2: var_6}
    var_8 = 'boolean'
    var_9 = {var_6: var_8}
    var_10 = {var_1: var_5, var_2: var_7, var_3: var_9}
    var_11 = module_0.if_then_else_from_json_schema(var_10, var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.if_clause).__module__}.{type(var_11.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_11.then_clause).__module__}.{type(var_11.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_11.else_clause).__module__}.{type(var_11.else_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_12 = var_11.if_clause
    var_13 = var_11.then_clause
    var_14 = var_11.else_clause
    var_15 = {var_4: var_4}
    var_16 = {var_2: var_6}
    var_17 = {var_1: var_15, var_2: var_16}
    var_18 = module_0.if_then_else_from_json_schema(var_17, var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert f'{type(var_18.if_clause).__module__}.{type(var_18.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_18.then_clause).__module__}.{type(var_18.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_18.else_clause).__module__}.{type(var_18.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_19 = var_18.if_clause
    var_20 = var_18.then_clause
    var_21 = {var_6: var_4}
    var_22 = {var_1: var_21}
    var_23 = module_0.if_then_else_from_json_schema(var_22, var_0)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.if_clause).__module__}.{type(var_23.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_23.then_clause).__module__}.{type(var_23.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_23.else_clause).__module__}.{type(var_23.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_24 = var_23.if_clause
    module_3.match(var_14, var_14)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_5.AllOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.all_of == {None: None}
    module_0.to_json_schema(var_2)

@pytest.mark.xfail(strict=True)
def test_case_35():
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
    var_1 = module_3.purge()
    var_2 = module_5.IfThenElse(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.if_clause is None
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_0.to_json_schema(var_2)

def test_case_36():
    var_0 = "jxY?p]'J+x|~m>"
    var_1 = module_2.String(pattern=var_0)
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
    assert var_1.pattern == "jxY?p]'J+x|~m>"
    assert f'{type(var_1.pattern_regex).__module__}.{type(var_1.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_0 = -2
    var_1 = '\tye\tiJ~f.c'
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == -2
    assert var_2.min_length == -2
    assert var_2.format == '\tye\tiJ~f.c'
    assert var_2.coerce_types is True
    assert var_2.pattern == '\tye\tiJ~f.c'
    assert f'{type(var_2.pattern_regex).__module__}.{type(var_2.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_2.Array(var_2, min_items=var_0, max_items=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.items).__module__}.{type(var_3.items).__qualname__}' == 'typesystem.fields.String'
    assert var_3.additional_items is False
    assert var_3.min_items == -2
    assert var_3.max_items == -2
    assert var_3.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
def test_case_38():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = var_1.validate(var_0)
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
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_5 = module_3.purge()
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
    var_6 = module_0.from_json_schema(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Union'
    assert var_6.default is None
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.any_of).__module__}.{type(var_6.any_of).__qualname__}' == 'builtins.list'
    assert len(var_6.any_of) == 5
    var_7 = None
    var_8 = module_4.Reference(var_2, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.to == {}
    assert var_8.definitions is None
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_9 = module_2.String(format=var_5, coerce_types=var_0)
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
    assert var_9.coerce_types == {}
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    module_0.to_json_schema(var_8)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = {}
    var_1 = module_4.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = module_5.Not(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.negated).__module__}.{type(var_2.negated).__qualname__}' == 'typesystem.schemas.Schema'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_3.purge()
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
    var_5 = module_0.from_json_schema(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.Not'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.negated).__module__}.{type(var_5.negated).__qualname__}' == 'typesystem.fields.Object'
    var_6 = module_0.to_json_schema(var_1, var_1)
    module_0.to_json_schema(var_4)

def test_case_40():
    var_0 = 1
    var_1 = 10
    var_2 = 'emil37'
    var_3 = module_2.String(max_length=var_1, min_length=var_0, pattern=var_2, format=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length == 10
    assert var_3.min_length == 1
    assert var_3.format == 'emil37'
    assert var_3.coerce_types is True
    assert var_3.pattern == 'emil37'
    assert f'{type(var_3.pattern_regex).__module__}.{type(var_3.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = 0
    var_5 = module_2.Integer(minimum=var_4, maximum=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Integer'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum == 0
    assert var_5.maximum == 10
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of is None
    assert var_5.precision is None
    assert var_5.coerce_types is True
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = module_2.Boolean()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_8 = module_2.Array(var_3, min_items=var_0, max_items=var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Array'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.items).__module__}.{type(var_8.items).__qualname__}' == 'typesystem.fields.String'
    assert var_8.additional_items is False
    assert var_8.min_items == 1
    assert var_8.max_items == 10
    assert var_8.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
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
    var_11 = var_3.serialize(var_2)
    assert var_11 == 'emil37'
    var_12 = module_0.from_json_schema(var_9, var_4)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Array'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.items).__module__}.{type(var_12.items).__qualname__}' == 'typesystem.fields.String'
    assert var_12.additional_items is False
    assert var_12.min_items == 1
    assert var_12.max_items == 10
    assert var_12.unique_items is False

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = {}
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_5.OneOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.one_of == ({'boolean', 'object', 'string', 'number', 'array'}, True)
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    module_0.to_json_schema(var_2)

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
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_4 = True
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
    assert var_8.min_length is True
    assert var_8.format == 'email'
    assert var_8.coerce_types is True
    assert var_8.pattern == '[a-z]+'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = False
    var_11 = 100
    var_12 = module_2.Integer(minimum=var_10, maximum=var_11, exclusive_minimum=var_4)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Integer'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum is False
    assert var_12.maximum == 100
    assert var_12.exclusive_minimum is True
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
    var_21 = module_2.Array(var_19, var_10, var_4, var_20, unique_items=var_4)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Array'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.items).__module__}.{type(var_21.items).__qualname__}' == 'typesystem.fields.String'
    assert var_21.additional_items is False
    assert var_21.min_items is True
    assert var_21.max_items == 5
    assert var_21.unique_items is True
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
    var_27 = module_2.Object(properties=var_25, additional_properties=var_10, min_properties=var_4, max_properties=var_5, required=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Object'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.properties).__module__}.{type(var_27.properties).__qualname__}' == 'builtins.dict'
    assert len(var_27.properties) == 1
    assert var_27.pattern_properties == {}
    assert var_27.additional_properties is False
    assert var_27.property_names is None
    assert var_27.min_properties is True
    assert var_27.max_properties == 10
    assert var_27.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_28 = module_0.to_json_schema(var_27)
    var_29 = 'a'
    var_30 = (var_29, var_29)
    var_31 = 'b'
    var_32 = (var_31, var_31)
    var_33 = [var_30, var_32]
    var_34 = module_2.Choice(choices=var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Choice'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.choices == [('a', 'a'), ('b', 'b')]
    assert var_34.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_35 = module_0.to_json_schema(var_34)
    var_36 = 42
    var_37 = module_2.Const(var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Const'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.const == 42
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_38 = module_0.to_json_schema(var_37)
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
    var_41 = [var_39, var_40]
    var_42 = module_2.Union(var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.Union'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert f'{type(var_42.any_of).__module__}.{type(var_42.any_of).__qualname__}' == 'builtins.list'
    assert len(var_42.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_43 = module_0.to_json_schema(var_42)
    var_44 = module_2.String()
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
    var_45 = 'test'
    var_46 = module_2.Const(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Const'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert var_46.const == 'test'
    var_47 = [var_44, var_46]
    var_48 = module_5.AllOf(var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert f'{type(var_48.all_of).__module__}.{type(var_48.all_of).__qualname__}' == 'builtins.list'
    assert len(var_48.all_of) == 2
    var_49 = module_0.to_json_schema(var_48)
    var_50 = module_2.String()
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
    var_51 = module_2.Const(var_4)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Const'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert var_51.const is True
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
    var_53 = module_2.Integer()
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
    var_54 = module_5.IfThenElse(var_51, var_52, var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert f'{type(var_54.if_clause).__module__}.{type(var_54.if_clause).__qualname__}' == 'typesystem.fields.Const'
    assert f'{type(var_54.then_clause).__module__}.{type(var_54.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_54.else_clause).__module__}.{type(var_54.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_55 = module_2.String()
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.String'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.allow_blank is False
    assert var_55.trim_whitespace is True
    assert var_55.max_length is None
    assert var_55.min_length is None
    assert var_55.format is None
    assert var_55.coerce_types is True
    assert var_55.pattern is None
    assert var_55.pattern_regex is None
    var_56 = module_5.Not(var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.composites.Not'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.negated).__module__}.{type(var_56.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
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
    var_59 = {var_23: var_58}
    var_60 = [var_23]
    var_61 = module_4.Schema(var_59)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert f'{type(var_61.fields).__module__}.{type(var_61.fields).__qualname__}' == 'builtins.dict'
    assert len(var_61.fields) == 1
    assert var_61.required == ['name']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_62 = module_0.to_json_schema(var_61)

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
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_3 = 10
    var_4 = '[a-z]+'
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
    assert var_6.min_length == 1
    assert var_6.format == 'email'
    assert var_6.coerce_types is True
    assert var_6.pattern == '[a-z]+'
    assert f'{type(var_6.pattern_regex).__module__}.{type(var_6.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = module_0.to_json_schema(var_6)
    var_8 = 0
    var_9 = module_2.Float(minimum=var_8, maximum=var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Float'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.minimum == 0
    assert var_9.maximum == 1
    assert var_9.exclusive_minimum is None
    assert var_9.exclusive_maximum is None
    assert var_9.multiple_of is None
    assert var_9.precision is None
    assert var_9.coerce_types is True
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_2.Boolean()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_12 = module_0.to_json_schema(var_11)
    var_13 = module_2.String()
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
    var_14 = 'P'
    var_15 = {var_4: var_6}
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
    assert var_17.required == ['P']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_18 = module_0.to_json_schema(var_17)
    var_19 = (var_4, var_4)
    var_20 = 'b'
    var_21 = (var_20, var_20)
    var_22 = [var_19, var_21]
    var_23 = module_2.Choice(choices=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Choice'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.choices == [('[a-z]+', '[a-z]+'), ('b', 'b')]
    assert var_23.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_24 = 'fixed'
    var_25 = module_2.Const(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Const'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.const == 'fixed'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
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
    var_30 = module_2.Union(var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Union'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.any_of).__module__}.{type(var_30.any_of).__qualname__}' == 'builtins.list'
    assert len(var_30.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_31 = module_0.to_json_schema(var_30)
    var_32 = module_2.String()
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
    var_33 = module_2.Integer()
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
    var_35 = module_5.OneOf(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.one_of).__module__}.{type(var_35.one_of).__qualname__}' == 'builtins.list'
    assert len(var_35.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_36 = module_0.to_json_schema(var_35)
    var_37 = module_3.purge()
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
    var_38 = [var_27, var_37]
    var_39 = module_5.AllOf(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.all_of).__module__}.{type(var_39.all_of).__qualname__}' == 'builtins.list'
    assert len(var_39.all_of) == 2
    module_0.to_json_schema(var_39)

def test_case_44():
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
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_9 = 'format'
    var_10 = module_0.to_json_schema(var_8)
    var_11 = 0
    var_12 = True
    var_13 = module_2.Integer(minimum=var_11, maximum=var_1, exclusive_maximum=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum == 0
    assert var_13.maximum is True
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
    var_26 = 'name'
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
    var_28 = {var_26: var_27}
    var_29 = [var_26]
    var_30 = module_2.Object(properties=var_28, min_properties=var_25, max_properties=var_22, required=var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Object'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.properties).__module__}.{type(var_30.properties).__qualname__}' == 'builtins.dict'
    assert len(var_30.properties) == 1
    assert var_30.pattern_properties == {}
    assert var_30.additional_properties is True
    assert var_30.property_names is None
    assert var_30.min_properties is True
    assert var_30.max_properties == 5
    assert var_30.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_31 = module_0.to_json_schema(var_30)
    var_32 = (var_9, var_9)
    var_33 = 'b'
    var_34 = (var_33, var_33)
    var_35 = [var_32, var_34]
    var_36 = module_2.Choice(choices=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Choice'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.choices == [('format', 'format'), ('b', 'b')]
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
    var_47 = module_2.Integer()
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
    var_48 = [var_46, var_47]
    var_49 = module_5.OneOf(var_48)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert f'{type(var_49.one_of).__module__}.{type(var_49.one_of).__qualname__}' == 'builtins.list'
    assert len(var_49.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_50 = module_0.to_json_schema(var_49)
    var_51 = module_2.String()
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
    assert var_51.pattern is None
    assert var_51.pattern_regex is None
    var_52 = module_2.Integer()
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Integer'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert var_52.minimum is None
    assert var_52.maximum is None
    assert var_52.exclusive_minimum is None
    assert var_52.exclusive_maximum is None
    assert var_52.multiple_of is None
    assert var_52.precision is None
    assert var_52.coerce_types is True
    var_53 = [var_51, var_52]
    var_54 = module_5.AllOf(var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert f'{type(var_54.all_of).__module__}.{type(var_54.all_of).__qualname__}' == 'builtins.list'
    assert len(var_54.all_of) == 2
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
    var_58 = module_2.Boolean()
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.coerce_types is True
    var_59 = module_5.IfThenElse(var_56, var_57, var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.if_clause).__module__}.{type(var_59.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_59.then_clause).__module__}.{type(var_59.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_59.else_clause).__module__}.{type(var_59.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_60 = module_0.to_json_schema(var_59)
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
    var_62 = module_5.Not(var_61)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.composites.Not'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert f'{type(var_62.negated).__module__}.{type(var_62.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_63 = module_0.to_json_schema(var_62)
    var_64 = 'test'
    var_65 = module_2.String()
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.String'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.allow_blank is False
    assert var_65.trim_whitespace is True
    assert var_65.max_length is None
    assert var_65.min_length is None
    assert var_65.format is None
    assert var_65.coerce_types is True
    assert var_65.pattern is None
    assert var_65.pattern_regex is None
    var_66 = {var_64: var_65}
    var_67 = module_4.Reference(var_64, var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    assert var_67.to == 'test'
    assert f'{type(var_67.definitions).__module__}.{type(var_67.definitions).__qualname__}' == 'builtins.dict'
    assert len(var_67.definitions) == 1
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
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
    var_70 = {var_26: var_69}
    var_71 = [var_26]
    var_72 = module_4.Schema(var_70)
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert f'{type(var_72.fields).__module__}.{type(var_72.fields).__qualname__}' == 'builtins.dict'
    assert len(var_72.fields) == 1
    assert var_72.required == ['name']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_73 = module_0.to_json_schema(var_72)
    var_74 = module_2.String()
    assert f'{type(var_74).__module__}.{type(var_74).__qualname__}' == 'typesystem.fields.String'
    assert var_74.title == ''
    assert var_74.description == ''
    assert var_74.allow_null is False
    assert var_74.read_only is False
    assert var_74.allow_blank is False
    assert var_74.trim_whitespace is True
    assert var_74.max_length is None
    assert var_74.min_length is None
    assert var_74.format is None
    assert var_74.coerce_types is True
    assert var_74.pattern is None
    assert var_74.pattern_regex is None
    var_75 = module_2.Integer()
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.fields.Integer'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert var_75.minimum is None
    assert var_75.maximum is None
    assert var_75.exclusive_minimum is None
    assert var_75.exclusive_maximum is None
    assert var_75.multiple_of is None
    assert var_75.precision is None
    assert var_75.coerce_types is True
    var_76 = True
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
    var_78 = module_0.to_json_schema(var_77)
    var_79 = 'default_value'
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
    var_81 = module_0.to_json_schema(var_80)

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
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_4 = 1
    var_5 = 10
    var_6 = 'email'
    var_7 = module_2.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_6)
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
    assert var_7.pattern == 'email'
    assert f'{type(var_7.pattern_regex).__module__}.{type(var_7.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_0.to_json_schema(var_7)
    var_9 = 0
    var_10 = 100
    var_11 = True
    var_12 = module_2.Integer(minimum=var_9, maximum=var_10, exclusive_maximum=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Integer'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum == 0
    assert var_12.maximum == 100
    assert var_12.exclusive_minimum is None
    assert var_12.exclusive_maximum is True
    assert var_12.multiple_of is None
    assert var_12.precision is None
    assert var_12.coerce_types is True
    var_13 = True
    var_14 = module_0.to_json_schema(var_12)
    var_15 = 0.5
    var_16 = module_2.Float(multiple_of=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Float'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum is None
    assert var_16.maximum is None
    assert var_16.exclusive_minimum is None
    assert var_16.exclusive_maximum is None
    assert var_16.multiple_of == pytest.approx(0.5, abs=0.01, rel=0.01)
    assert var_16.precision is None
    assert var_16.coerce_types is True
    var_17 = module_0.to_json_schema(var_16)
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
    var_21 = 5
    var_22 = True
    var_23 = module_2.Array(var_20, min_items=var_13, max_items=var_21, unique_items=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Array'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.items).__module__}.{type(var_23.items).__qualname__}' == 'typesystem.fields.String'
    assert var_23.additional_items is False
    assert var_23.min_items is True
    assert var_23.max_items == 5
    assert var_23.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_24 = module_0.to_json_schema(var_23)
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
    var_29 = False
    var_30 = module_2.Object(properties=var_27, additional_properties=var_29, required=var_28)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Object'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.properties).__module__}.{type(var_30.properties).__qualname__}' == 'builtins.dict'
    assert len(var_30.properties) == 1
    assert var_30.pattern_properties == {}
    assert var_30.additional_properties is False
    assert var_30.property_names is None
    assert var_30.min_properties is None
    assert var_30.max_properties is None
    assert var_30.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_31 = module_0.to_json_schema(var_30)
    var_32 = 'a'
    var_33 = (var_32, var_32)
    var_34 = (var_17, var_17)
    var_35 = [var_33, var_34]
    var_36 = module_2.Choice(choices=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Choice'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert f'{type(var_36.choices).__module__}.{type(var_36.choices).__qualname__}' == 'builtins.list'
    assert len(var_36.choices) == 2
    assert var_36.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_37 = module_0.to_json_schema(var_36)
    var_38 = module_2.String()
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
    var_39 = module_2.Integer()
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
    var_41 = module_2.Union(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Union'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.any_of).__module__}.{type(var_41.any_of).__qualname__}' == 'builtins.list'
    assert len(var_41.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_42 = module_0.to_json_schema(var_41)
    var_43 = module_2.String()
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
    var_44 = 'test'
    var_45 = module_2.Const(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Const'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.const == 'test'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_46 = [var_43, var_45]
    var_47 = module_5.AllOf(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert f'{type(var_47.all_of).__module__}.{type(var_47.all_of).__qualname__}' == 'builtins.list'
    assert len(var_47.all_of) == 2
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
    var_50 = module_2.String()
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
    var_51 = {var_25: var_50}
    var_52 = [var_25]
    var_53 = module_4.Schema(var_51)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert f'{type(var_53.fields).__module__}.{type(var_53.fields).__qualname__}' == 'builtins.dict'
    assert len(var_53.fields) == 1
    assert var_53.required == ['name']
    assert module_4.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_54 = module_0.to_json_schema(var_53)
    var_55 = module_2.String()
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.String'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.allow_blank is False
    assert var_55.trim_whitespace is True
    assert var_55.max_length is None
    assert var_55.min_length is None
    assert var_55.format is None
    assert var_55.coerce_types is True
    assert var_55.pattern is None
    assert var_55.pattern_regex is None
    var_56 = module_2.Integer()
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Integer'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert var_56.minimum is None
    assert var_56.maximum is None
    assert var_56.exclusive_minimum is None
    assert var_56.exclusive_maximum is None
    assert var_56.multiple_of is None
    assert var_56.precision is None
    assert var_56.coerce_types is True
    var_57 = module_2.Boolean()
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert var_57.coerce_types is True
    var_58 = module_5.IfThenElse(var_55, var_56, var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert f'{type(var_58.if_clause).__module__}.{type(var_58.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_58.then_clause).__module__}.{type(var_58.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_58.else_clause).__module__}.{type(var_58.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_59 = module_0.to_json_schema(var_58)
    var_60 = module_5.Not(var_55)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.composites.Not'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert f'{type(var_60.negated).__module__}.{type(var_60.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_61 = module_0.to_json_schema(var_60)

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
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_3 = 10
    var_4 = '[a-z]+'
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
    assert var_6.min_length == 1
    assert var_6.format == 'email'
    assert var_6.coerce_types is True
    assert var_6.pattern == '[a-z]+'
    assert f'{type(var_6.pattern_regex).__module__}.{type(var_6.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = module_0.to_json_schema(var_6)
    var_8 = 0
    var_9 = module_2.Float(minimum=var_8, maximum=var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Float'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.minimum == 0
    assert var_9.maximum == 1
    assert var_9.exclusive_minimum is None
    assert var_9.exclusive_maximum is None
    assert var_9.multiple_of is None
    assert var_9.precision is None
    assert var_9.coerce_types is True
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_2.Boolean()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_12 = module_0.to_json_schema(var_11)
    var_13 = module_2.String()
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
    var_14 = module_2.Array(var_13, min_items=var_2, max_items=var_3)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Array'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.items).__module__}.{type(var_14.items).__qualname__}' == 'typesystem.fields.String'
    assert var_14.additional_items is False
    assert var_14.min_items == 1
    assert var_14.max_items == 10
    assert var_14.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_15 = module_0.to_json_schema(var_14)
    var_16 = 'name'
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
    var_18 = {var_16: var_17}
    var_19 = [var_16]
    var_20 = module_2.Object(properties=var_18, required=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Object'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.properties).__module__}.{type(var_20.properties).__qualname__}' == 'builtins.dict'
    assert len(var_20.properties) == 1
    assert var_20.pattern_properties == {}
    assert var_20.additional_properties is True
    assert var_20.property_names is None
    assert var_20.min_properties is None
    assert var_20.max_properties is None
    assert var_20.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_21 = module_0.to_json_schema(var_20)
    var_22 = 'a'
    var_23 = (var_22, var_22)
    var_24 = 'b'
    var_25 = (var_24, var_24)
    var_26 = [var_23, var_25]
    var_27 = module_2.Choice(choices=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Choice'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.choices == [('a', 'a'), ('b', 'b')]
    assert var_27.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_28 = module_0.to_json_schema(var_27)
    var_29 = 'fixed'
    var_30 = module_2.Const(var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Const'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.const == 'fixed'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_31 = module_0.to_json_schema(var_30)
    var_32 = module_2.String()
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
    var_33 = module_2.Integer()
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
    var_35 = module_2.Union(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Union'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.any_of).__module__}.{type(var_35.any_of).__qualname__}' == 'builtins.list'
    assert len(var_35.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
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
    var_40 = module_5.OneOf(var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert f'{type(var_40.one_of).__module__}.{type(var_40.one_of).__qualname__}' == 'builtins.list'
    assert len(var_40.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_41 = module_0.to_json_schema(var_40)
    var_42 = module_2.String()
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
    var_43 = module_2.Integer()
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Integer'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.minimum is None
    assert var_43.maximum is None
    assert var_43.exclusive_minimum is None
    assert var_43.exclusive_maximum is None
    assert var_43.multiple_of is None
    assert var_43.precision is None
    assert var_43.coerce_types is True
    var_44 = [var_42, var_43]
    var_45 = module_5.AllOf(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert f'{type(var_45.all_of).__module__}.{type(var_45.all_of).__qualname__}' == 'builtins.list'
    assert len(var_45.all_of) == 2
    var_46 = module_0.to_json_schema(var_45)
    var_47 = module_2.String()
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
    assert var_47.coerce_types is True
    assert var_47.pattern is None
    assert var_47.pattern_regex is None
    var_48 = module_2.Integer()
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.Integer'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert var_48.minimum is None
    assert var_48.maximum is None
    assert var_48.exclusive_minimum is None
    assert var_48.exclusive_maximum is None
    assert var_48.multiple_of is None
    assert var_48.precision is None
    assert var_48.coerce_types is True
    var_49 = module_2.Boolean()
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert var_49.coerce_types is True
    var_50 = module_5.IfThenElse(var_47, var_48, var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert f'{type(var_50.if_clause).__module__}.{type(var_50.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_50.then_clause).__module__}.{type(var_50.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_50.else_clause).__module__}.{type(var_50.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_51 = module_0.to_json_schema(var_50)
    var_52 = module_5.Not(var_42)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.composites.Not'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert f'{type(var_52.negated).__module__}.{type(var_52.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_53 = module_0.to_json_schema(var_52)
    var_54 = module_4.Definitions()
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_54) == 0

def test_case_47():
    var_0 = {}
    var_1 = module_4.Definitions(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_0 = 'type'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 'default'
    var_6 = 'number'
    var_7 = 0
    var_8 = 100
    var_9 = 2
    var_10 = 50
    var_11 = {var_0: var_6, var_5: var_7, var_1: var_8, var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10}
    var_12 = False
    var_13 = module_4.Definitions()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13) == 0
    var_14 = module_0.from_json_schema_type(var_11, var_6, var_12, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Float'
    assert var_14.default == 50
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum is None
    assert var_14.maximum == 100
    assert var_14.exclusive_minimum == 0
    assert var_14.exclusive_maximum == 100
    assert var_14.multiple_of == 2
    assert var_14.precision is None
    assert var_14.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_16 = {var_0: var_15, var_3: var_12, var_1: var_8, var_2: var_12, var_3: var_8, var_4: var_9, var_5: var_10}
    var_17 = False
    var_18 = module_4.Definitions()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_18) == 0
    var_19 = module_0.from_json_schema_type(var_16, var_15, var_17, var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Integer'
    assert var_19.default == 50
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.minimum is None
    assert var_19.maximum == 100
    assert var_19.exclusive_minimum is False
    assert var_19.exclusive_maximum == 100
    assert var_19.multiple_of == 2
    assert var_19.precision is None
    assert var_19.coerce_types is False
    var_20 = 'minLength'
    var_21 = 'maxLength'
    var_22 = 'format'
    var_23 = 'pattern'
    var_24 = 'string'
    var_25 = 5
    var_26 = 10
    var_27 = 'email'
    var_28 = '^[a-zA-Z0-9]+$'
    var_29 = 'test'
    var_30 = {var_0: var_24, var_20: var_25, var_21: var_26, var_22: var_27, var_23: var_28, var_5: var_29}
    var_31 = False
    var_32 = module_4.Definitions()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_32) == 0
    var_33 = module_0.from_json_schema_type(var_30, var_24, var_31, var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.String'
    assert var_33.default == 'test'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.allow_blank is False
    assert var_33.trim_whitespace is True
    assert var_33.max_length == 10
    assert var_33.min_length == 5
    assert var_33.format == 'email'
    assert var_33.coerce_types is False
    assert var_33.pattern == '^[a-zA-Z0-9]+$'
    assert f'{type(var_33.pattern_regex).__module__}.{type(var_33.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_34 = 'boolean'
    var_35 = True
    var_36 = {var_0: var_34, var_5: var_35}
    var_37 = False
    var_38 = module_4.Definitions()
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_38) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_39 = module_0.from_json_schema_type(var_36, var_34, var_37, var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_39.default is True
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_40 = 'items'
    var_41 = 'additionalItems'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_24}
    var_47 = False
    var_48 = [var_29]
    var_49 = {var_0: var_45, var_40: var_46, var_41: var_47, var_42: var_35, var_43: var_26, var_44: var_35, var_5: var_48}
    var_50 = False
    var_51 = module_4.Definitions()
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_51) == 0
    var_52 = module_0.from_json_schema_type(var_49, var_45, var_50, var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Array'
    assert var_52.default == ['test']
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert f'{type(var_52.items).__module__}.{type(var_52.items).__qualname__}' == 'typesystem.fields.String'
    assert var_52.additional_items is False
    assert var_52.min_items is True
    assert var_52.max_items == 10
    assert var_52.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_53 = var_52.items
    var_54 = 'properties'
    var_55 = 'patternProperties'
    var_56 = 'additionalProperties'
    var_57 = 'propertyNames'
    var_58 = 'minProperties'
    var_59 = 'maxProperties'
    var_60 = 'required'
    var_61 = 'object'
    var_62 = 'name'
    var_63 = 'age'
    var_64 = {var_0: var_24}
    var_65 = {var_0: var_15}
    var_66 = {var_62: var_64, var_63: var_65}
    var_67 = '^S_'
    var_68 = '^I_'
    var_69 = {var_0: var_24}
    var_70 = {var_0: var_15}
    var_71 = {var_67: var_69, var_68: var_70}
    var_72 = False
    var_73 = {var_0: var_24}
    var_74 = [var_62]
    var_75 = 25
    var_76 = {var_62: var_29, var_63: var_75}
    var_77 = {var_0: var_61, var_54: var_66, var_55: var_71, var_56: var_72, var_57: var_73, var_58: var_35, var_59: var_26, var_60: var_74, var_5: var_76}
    var_78 = False
    var_79 = module_4.Definitions()
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_79) == 0
    var_80 = module_0.from_json_schema_type(var_77, var_61, var_78, var_79)
    assert f'{type(var_80).__module__}.{type(var_80).__qualname__}' == 'typesystem.fields.Object'
    assert var_80.default == {'name': 'test', 'age': 25}
    assert var_80.title == ''
    assert var_80.description == ''
    assert var_80.allow_null is False
    assert var_80.read_only is False
    assert f'{type(var_80.properties).__module__}.{type(var_80.properties).__qualname__}' == 'builtins.dict'
    assert len(var_80.properties) == 2
    assert f'{type(var_80.pattern_properties).__module__}.{type(var_80.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_80.pattern_properties) == 2
    assert var_80.additional_properties is False
    assert f'{type(var_80.property_names).__module__}.{type(var_80.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_80.min_properties is True
    assert var_80.max_properties == 10
    assert var_80.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_81 = var_80.properties[var_62]
    var_82 = var_80.properties[var_63]
    var_83 = var_80.pattern_properties[var_67]
    var_84 = var_80.pattern_properties[var_68]
    var_85 = var_80.property_names

def test_case_49():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_4.Definitions()
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
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_5 = 'integer'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_4.Definitions()
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
    var_12 = 0
    var_13 = var_9.any_of[var_12]
    var_14 = 1
    var_15 = var_9.any_of[var_14]
    var_16 = 'null'
    var_17 = [var_1, var_16]
    var_18 = {var_0: var_17}
    var_19 = module_4.Definitions()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_19) == 0
    var_20 = module_0.type_from_json_schema(var_18, var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.String'
    assert var_20.default is None
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is True
    assert var_20.read_only is False
    assert var_20.allow_blank is True
    assert var_20.trim_whitespace is True
    assert var_20.max_length is None
    assert var_20.min_length is None
    assert var_20.format is None
    assert var_20.coerce_types is False
    assert var_20.pattern is None
    assert var_20.pattern_regex is None
    var_21 = []
    var_22 = {var_0: var_21}
    var_23 = module_4.Definitions()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_23) == 0
    var_24 = module_0.type_from_json_schema(var_22, var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Union'
    assert var_24.default is None
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is True
    assert var_24.read_only is False
    assert f'{type(var_24.any_of).__module__}.{type(var_24.any_of).__qualname__}' == 'builtins.list'
    assert len(var_24.any_of) == 5
    var_25 = [var_16]
    var_26 = {var_0: var_25}
    var_27 = module_0.type_from_json_schema(var_26, var_23)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Const'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.const is None
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_28 = 'minLength'
    var_29 = 'maxLength'
    var_30 = 5
    var_31 = 10
    var_32 = {var_0: var_1, var_28: var_30, var_29: var_31}
    var_33 = module_4.Definitions()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_33) == 0
    var_34 = module_0.type_from_json_schema(var_32, var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.String'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.allow_blank is False
    assert var_34.trim_whitespace is True
    assert var_34.max_length == 10
    assert var_34.min_length == 5
    assert var_34.format is None
    assert var_34.coerce_types is False
    assert var_34.pattern is None
    assert var_34.pattern_regex is None
    var_35 = 'minimum'
    var_36 = 'maximum'
    var_37 = 'number'
    var_38 = 100
    var_39 = {var_0: var_37, var_35: var_12, var_36: var_38}
    var_40 = module_4.Definitions()
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_40) == 0
    var_41 = module_0.type_from_json_schema(var_39, var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Float'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.minimum == 0
    assert var_41.maximum == 100
    assert var_41.exclusive_minimum is None
    assert var_41.exclusive_maximum is None
    assert var_41.multiple_of is None
    assert var_41.precision is None
    assert var_41.coerce_types is False
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'array'
    var_45 = {var_0: var_44, var_42: var_14, var_43: var_31}
    var_46 = module_4.Definitions()
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_46) == 0
    var_47 = module_0.type_from_json_schema(var_45, var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.Array'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert var_47.items is None
    assert var_47.additional_items is True
    assert var_47.min_items == 1
    assert var_47.max_items == 10
    assert var_47.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_48 = 'minProperties'
    var_49 = 'maxProperties'
    var_50 = 'object'
    var_51 = {var_0: var_50, var_48: var_14, var_49: var_31}
    var_52 = module_4.Definitions()
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_52) == 0
    var_53 = module_0.type_from_json_schema(var_51, var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.Object'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert var_53.properties == {}
    assert var_53.pattern_properties == {}
    assert var_53.additional_properties is None
    assert var_53.property_names is None
    assert var_53.min_properties == 1
    assert var_53.max_properties == 10
    assert var_53.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}

def test_case_50():
    var_0 = 1
    var_1 = 10
    var_2 = 'a-z]+'
    var_3 = module_2.String(max_length=var_1, min_length=var_0, pattern=var_2, format=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length == 10
    assert var_3.min_length == 1
    assert var_3.format == 'a-z]+'
    assert var_3.coerce_types is True
    assert var_3.pattern == 'a-z]+'
    assert f'{type(var_3.pattern_regex).__module__}.{type(var_3.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = 0
    var_5 = 2112
    var_6 = module_2.Integer(minimum=var_4, maximum=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Integer'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.minimum == 0
    assert var_6.maximum == 2112
    assert var_6.exclusive_minimum is None
    assert var_6.exclusive_maximum is None
    assert var_6.multiple_of is None
    assert var_6.precision is None
    assert var_6.coerce_types is True
    var_7 = module_3.purge()
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
    var_8 = module_2.Float(minimum=var_4, maximum=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Float'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.minimum == 0
    assert var_8.maximum == 1
    assert var_8.exclusive_minimum is None
    assert var_8.exclusive_maximum is None
    assert var_8.multiple_of is None
    assert var_8.precision is None
    assert var_8.coerce_types is True
    var_9 = module_0.to_json_schema(var_8)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_10 = module_2.Boolean()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_11 = module_2.Array(additional_items=var_8, min_items=var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Array'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.items is None
    assert f'{type(var_11.additional_items).__module__}.{type(var_11.additional_items).__qualname__}' == 'typesystem.fields.Float'
    assert var_11.min_items is None
    assert var_11.max_items is None
    assert var_11.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_12 = module_0.to_json_schema(var_6)
    var_13 = module_0.to_json_schema(var_11, var_7)
    var_14 = module_0.from_json_schema(var_9, var_7)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Float'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum == 0
    assert var_14.maximum == 1
    assert var_14.exclusive_minimum is None
    assert var_14.exclusive_maximum is None
    assert var_14.multiple_of is None
    assert var_14.precision is None
    assert var_14.coerce_types is False

def test_case_51():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'patternProperties', 'additionalProperties', 'type', 'multipleOf', 'required', 'maximum', 'pattern', 'minimum', 'minItems', 'exclusiveMaximum', 'dependencies', 'maxProperties', 'properties', 'maxLength', 'boolean_schema', 'propertyNames', 'minProperties', 'exclusiveMinimum', 'items', 'maxItems', 'minLength', 'uniqueItems', 'additionalItems', 'contains'}
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
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'test_schema'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = module_4.Definitions()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13) == 0
    var_14 = module_0.from_json_schema(var_12, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Any'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    var_15 = '$ref'
    var_16 = '#/components/schemas/test_schema'
    var_17 = {var_15: var_16}
    var_18 = module_0.from_json_schema(var_17, var_13)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.to == '#/components/schemas/test_schema'
    assert f'{type(var_18.definitions).__module__}.{type(var_18.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_18.definitions) == 0
    assert module_4.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_4.Reference.target).__module__}.{type(module_4.Reference.target).__qualname__}' == 'builtins.property'
    var_19 = 'minLength'
    var_20 = {var_7: var_8, var_19: var_0}
    var_21 = module_0.from_json_schema(var_20)
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
    assert var_21.coerce_types is False
    assert var_21.pattern is None
    assert var_21.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_22 = 'enum'
    var_23 = 'a'
    var_24 = 'b'
    var_25 = 'c'
    var_26 = [var_23, var_24, var_25]
    var_27 = {var_22: var_26}
    var_28 = module_0.from_json_schema(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Choice'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_28.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_29 = 'const'
    var_30 = 'test'
    var_31 = {var_29: var_30}
    var_32 = module_0.from_json_schema(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Const'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.const == 'test'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_33 = {var_7: var_8}
    var_34 = {var_19: var_0}
    var_35 = [var_33, var_34]
    var_36 = {var_30: var_35}
    var_37 = module_0.from_json_schema(var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Any'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    var_38 = 'anyOf'
    var_39 = {var_7: var_8}
    var_40 = 'number'
    var_41 = {var_7: var_40}
    var_42 = [var_39, var_41]
    var_43 = {var_38: var_42}
    var_44 = module_0.from_json_schema(var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Union'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert f'{type(var_44.any_of).__module__}.{type(var_44.any_of).__qualname__}' == 'builtins.list'
    assert len(var_44.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_45 = 'oneOf'
    var_46 = {var_7: var_8}
    var_47 = {var_7: var_40}
    var_48 = [var_46, var_47]
    var_49 = {var_45: var_48}
    var_50 = module_0.from_json_schema(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert f'{type(var_50.one_of).__module__}.{type(var_50.one_of).__qualname__}' == 'builtins.list'
    assert len(var_50.one_of) == 2
    assert module_5.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_51 = 'not'
    var_52 = {var_7: var_8}
    var_53 = {var_51: var_52}
    var_54 = module_0.from_json_schema(var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.composites.Not'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert f'{type(var_54.negated).__module__}.{type(var_54.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_5.Not.errors == {'negated': 'Must not match.'}
    var_55 = 'if'
    var_56 = 'then'
    var_57 = 'else'
    var_58 = {var_7: var_8}
    var_59 = {var_19: var_0}
    var_60 = {var_19: var_2}
    var_61 = {var_55: var_58, var_56: var_59, var_57: var_60}
    var_62 = module_0.from_json_schema(var_61)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert f'{type(var_62.if_clause).__module__}.{type(var_62.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_62.then_clause).__module__}.{type(var_62.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_62.else_clause).__module__}.{type(var_62.else_clause).__qualname__}' == 'typesystem.fields.Union'
    var_63 = [var_23, var_24, var_25]
    var_64 = {var_7: var_8, var_19: var_0, var_22: var_63}
    var_65 = module_0.from_json_schema(var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert f'{type(var_65.all_of).__module__}.{type(var_65.all_of).__qualname__}' == 'builtins.list'
    assert len(var_65.all_of) == 2
    var_66 = {}
    var_67 = module_0.from_json_schema(var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.fields.Any'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False