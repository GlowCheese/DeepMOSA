# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import typesystem.fields as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3
import re as module_4
import enum as module_5

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.from_json_schema(var_0, var_0)

def test_case_1():
    var_0 = b't\xf9$\xe8Z!\xbb@\xc0\xec#'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.get_valid_types(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    var_0 = b'X\xbd2'
    var_1 = {var_0: var_0, var_0: var_0}
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    var_1 = {}
    var_2 = 'b[r?>/26Gvg+q8'
    var_3 = True
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_1, var_2, var_3, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.ref_from_json_schema(var_0, var_0)

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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
def test_case_10():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = {}
    module_0.not_from_json_schema(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

def test_case_13():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    var_2 = module_0.to_json_schema(var_1)
    assert var_2 is False

def test_case_14():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = var_1.__or__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Union'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.any_of).__module__}.{type(var_2.any_of).__qualname__}' == 'builtins.list'
    assert len(var_2.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = False
    module_0.to_json_schema(var_0, var_0)

def test_case_16():
    var_0 = False
    var_1 = b'2\xa1'
    var_2 = {var_0: var_1, var_0: var_0}
    var_3 = None
    var_4 = module_0.type_from_json_schema(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.default is None
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is True
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    var_5 = module_0.get_standard_properties(var_4)
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = False
    var_1 = b'\x91\xd2\x94'
    var_2 = {var_0: var_1, var_0: var_0}
    var_3 = module_0.from_json_schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    module_0.to_json_schema(var_0, var_0)

def test_case_18():
    var_0 = None
    var_1 = module_1.Const(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Const'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.const is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    var_0 = 'oneOf'
    var_1 = 1
    var_2 = module_3.Definitions()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = '$ref'
    var_4 = '#/components/schemas/StringSchema'
    var_5 = {var_3: var_4}
    var_6 = '#/components/schemas/IntSchema'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = {var_0: var_8}
    var_10 = module_0.one_of_from_json_schema(var_9, var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.one_of).__module__}.{type(var_10.one_of).__qualname__}' == 'builtins.list'
    assert len(var_10.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    var_11 = var_10.one_of
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = var_10.one_of[var_1]
    var_14 = var_10.one_of[var_1]

def test_case_20():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.from_json_schema(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}

def test_case_21():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_3.Definitions()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = module_0.one_of_from_json_schema(var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.one_of).__module__}.{type(var_9.one_of).__qualname__}' == 'builtins.list'
    assert len(var_9.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.one_of[var_12]
    var_14 = 1
    var_15 = var_9.one_of[var_14]
    var_16 = 'default'
    var_17 = {var_1: var_2, var_15: var_10, var_2: var_4}
    var_18 = 'number'
    var_19 = {var_1: var_18}
    var_20 = [var_17, var_19]
    var_21 = 'test'
    var_22 = {var_0: var_20, var_16: var_21}
    var_23 = module_0.one_of_from_json_schema(var_22, var_13)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_23.default == 'test'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.one_of).__module__}.{type(var_23.one_of).__qualname__}' == 'builtins.list'
    assert len(var_23.one_of) == 2
    var_24 = 'properties'
    var_25 = 'object'
    var_26 = 'name'
    var_27 = {var_1: var_2}
    var_28 = {var_26: var_27}
    var_29 = {var_1: var_25, var_24: var_28}
    var_30 = 'items'
    var_31 = 'array'
    var_32 = {var_1: var_4}
    var_33 = {var_1: var_31, var_30: var_32}
    var_34 = [var_29, var_33]
    var_35 = {var_0: var_34}
    var_36 = module_3.Definitions()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_36) == 0
    var_37 = module_0.one_of_from_json_schema(var_35, var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert f'{type(var_37.one_of).__module__}.{type(var_37.one_of).__qualname__}' == 'builtins.list'
    assert len(var_37.one_of) == 2
    var_38 = var_37.one_of
    with pytest.raises(TypeError):
        var_39 = len(var_23)
    assert var_39 == 2

def test_case_22():
    var_0 = '$ref'
    var_1 = '#/components/schemas/User'
    var_2 = {var_0: var_1}
    var_3 = module_3.Definitions()
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    var_5 = module_3.Definitions()
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
    var_7 = '#/components/schemas/nested/Item'
    var_8 = {var_0: var_7}
    var_9 = module_3.Definitions()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_9) == 0
    var_10 = module_0.ref_from_json_schema(var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.to == '#/components/schemas/nested/Item'
    assert f'{type(var_10.definitions).__module__}.{type(var_10.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_10.definitions) == 0
    var_11 = 'components/schemas/User'
    var_12 = {var_0: var_11}
    var_13 = module_3.Definitions()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13) == 0
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_12, var_13)

def test_case_23():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_0.from_json_schema(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.String'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.default == ''
    assert var_6.allow_blank is True
    assert var_6.trim_whitespace is True
    assert var_6.max_length is None
    assert var_6.min_length is None
    assert var_6.format is None
    assert var_6.coerce_types is False
    assert var_6.pattern is None
    assert var_6.pattern_regex is None
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = 'integer'
    var_8 = {var_3: var_7}
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_10 = 'number'
    var_11 = {var_3: var_10}
    var_12 = module_0.from_json_schema(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Float'
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
    assert var_12.coerce_types is False
    var_13 = 'boolean'
    var_14 = {var_3: var_13}
    var_15 = module_0.from_json_schema(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.coerce_types is False
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_16 = 'array'
    var_17 = {var_3: var_16}
    var_18 = module_0.from_json_schema(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Array'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.items is None
    assert var_18.additional_items is True
    assert var_18.min_items == 0
    assert var_18.max_items is None
    assert var_18.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_19 = {var_3: var_13}
    var_20 = module_0.from_json_schema(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.coerce_types is False
    var_21 = 'enum'
    var_22 = 2
    var_23 = 3
    var_24 = [var_0, var_22, var_23]
    var_25 = {var_21: var_24}
    var_26 = module_0.from_json_schema(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Choice'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.choices == [(True, True), (2, 2), (3, 3)]
    assert var_26.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_27 = 'const'
    var_28 = 'value'
    var_29 = {var_27: var_28}
    var_30 = module_0.from_json_schema(var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Const'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.const == 'value'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_31 = 'minLength'
    var_32 = 'maxLength'
    var_33 = 5
    var_34 = 10
    var_35 = {var_3: var_4, var_31: var_33, var_32: var_34}
    var_36 = module_0.from_json_schema(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.String'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.allow_blank is False
    assert var_36.trim_whitespace is True
    assert var_36.max_length == 10
    assert var_36.min_length == 5
    assert var_36.format is None
    assert var_36.coerce_types is False
    assert var_36.pattern is None
    assert var_36.pattern_regex is None
    var_37 = 'minimum'
    var_38 = 'maximum'
    var_39 = 100
    var_40 = {var_3: var_10, var_37: var_0, var_38: var_39}
    var_41 = module_0.from_json_schema(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Float'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.minimum is True
    assert var_41.maximum == 100
    assert var_41.exclusive_minimum is None
    assert var_41.exclusive_maximum is None
    assert var_41.multiple_of is None
    assert var_41.precision is None
    assert var_41.coerce_types is False
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = {var_3: var_16, var_42: var_0, var_43: var_33}
    var_45 = module_0.from_json_schema(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Array'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.items is None
    assert var_45.additional_items is True
    assert var_45.min_items is True
    assert var_45.max_items == 5
    assert var_45.unique_items is False
    var_46 = 'properties'
    var_47 = 'name'
    var_48 = 'a#e'
    var_49 = {var_3: var_4}
    var_50 = {}
    var_51 = {var_47: var_49, var_48: var_50}
    var_52 = {var_3: var_16, var_46: var_51}
    var_53 = module_0.from_json_schema(var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.Array'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert var_53.items is None
    assert var_53.additional_items is True
    assert var_53.min_items == 0
    assert var_53.max_items is None
    assert var_53.unique_items is False
    var_54 = 'allOf'
    var_55 = {var_3: var_4}
    var_56 = {var_31: var_33}
    var_57 = [var_55, var_56]
    var_58 = {var_54: var_57}
    var_59 = module_0.from_json_schema(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.all_of).__module__}.{type(var_59.all_of).__qualname__}' == 'builtins.list'
    assert len(var_59.all_of) == 2
    var_60 = 'anyOf'
    var_61 = {var_3: var_4}
    var_62 = {var_3: var_7}
    var_63 = [var_61, var_62]
    var_64 = {var_60: var_63}
    var_65 = module_0.from_json_schema(var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Union'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert f'{type(var_65.any_of).__module__}.{type(var_65.any_of).__qualname__}' == 'builtins.list'
    assert len(var_65.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_66 = 'oneOf'
    var_67 = {var_3: var_4}
    var_68 = {var_3: var_7}
    var_69 = [var_67, var_68]
    var_70 = {var_66: var_69}
    var_71 = module_0.from_json_schema(var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert f'{type(var_71.one_of).__module__}.{type(var_71.one_of).__qualname__}' == 'builtins.list'
    assert len(var_71.one_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_72 = 'not'
    var_73 = {var_3: var_4}
    var_74 = {var_72: var_73}
    var_75 = module_0.from_json_schema(var_74)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.composites.Not'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert f'{type(var_75.negated).__module__}.{type(var_75.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_76 = 'if'
    var_77 = 'then'
    var_78 = 'else'
    var_79 = {var_3: var_4}
    var_80 = {var_31: var_33}
    var_81 = {var_3: var_7}
    var_82 = {var_76: var_79, var_77: var_80, var_78: var_81}
    var_83 = module_0.from_json_schema(var_82)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert f'{type(var_83.if_clause).__module__}.{type(var_83.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_83.then_clause).__module__}.{type(var_83.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_83.else_clause).__module__}.{type(var_83.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_84 = module_3.Definitions()
    assert f'{type(var_84).__module__}.{type(var_84).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_84) == 0
    var_85 = {var_3: var_4, var_21: var_57, var_31: var_0}
    var_86 = module_0.from_json_schema(var_85)
    assert f'{type(var_86).__module__}.{type(var_86).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_86.title == ''
    assert var_86.description == ''
    assert var_86.allow_null is False
    assert var_86.read_only is False
    assert f'{type(var_86.all_of).__module__}.{type(var_86.all_of).__qualname__}' == 'builtins.list'
    assert len(var_86.all_of) == 2
    var_87 = 'pat(tern'
    var_88 = {var_3: var_4, var_87: var_4}
    var_89 = module_0.from_json_schema(var_88)
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.fields.String'
    assert var_89.title == ''
    assert var_89.description == ''
    assert var_89.allow_null is False
    assert var_89.read_only is False
    assert var_89.default == ''
    assert var_89.allow_blank is True
    assert var_89.trim_whitespace is True
    assert var_89.max_length is None
    assert var_89.min_length is None
    assert var_89.format is None
    assert var_89.coerce_types is False
    assert var_89.pattern is None
    assert var_89.pattern_regex is None
    var_90 = 'schemas'
    var_91 = 'User'
    var_92 = {var_3: var_4}
    var_93 = {var_91: var_92}
    var_94 = {var_90: var_93}
    var_95 = {var_3: var_16, var_38: var_94}
    var_96 = module_0.from_json_schema(var_95)
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.fields.Array'
    assert var_96.title == ''
    assert var_96.description == ''
    assert var_96.allow_null is False
    assert var_96.read_only is False
    assert var_96.items is None
    assert var_96.additional_items is True
    assert var_96.min_items == 0
    assert var_96.max_items is None
    assert var_96.unique_items is False

def test_case_24():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'integer'
    var_8 = {var_4: var_7}
    var_9 = 'boolean'
    var_10 = {var_4: var_9}
    var_11 = {var_1: var_6, var_2: var_8, var_3: var_10}
    var_12 = module_0.if_then_else_from_json_schema(var_11, var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.if_clause).__module__}.{type(var_12.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_12.then_clause).__module__}.{type(var_12.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_12.else_clause).__module__}.{type(var_12.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_13 = module_0.if_then_else_from_json_schema(var_11, var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.if_clause).__module__}.{type(var_13.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_13.then_clause).__module__}.{type(var_13.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_13.else_clause).__module__}.{type(var_13.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_14 = var_0.__iter__()
    var_15 = {var_4: var_5}
    var_16 = {var_1: var_15}
    var_17 = module_0.if_then_else_from_json_schema(var_16, var_0)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.if_clause).__module__}.{type(var_17.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_17.then_clause).__module__}.{type(var_17.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_17.else_clause).__module__}.{type(var_17.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_18 = {var_4: var_5}
    var_19 = {var_1: var_18}
    var_20 = module_0.if_then_else_from_json_schema(var_19, var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.if_clause).__module__}.{type(var_20.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_20.then_clause).__module__}.{type(var_20.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_20.else_clause).__module__}.{type(var_20.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_21 = var_0.clear()
    var_22 = var_0.__iter__()
    var_23 = 'items'
    var_24 = 'array'
    var_25 = {var_4: var_7}
    var_26 = {var_4: var_24, var_23: var_25}
    var_27 = 'enum'
    var_28 = 1
    var_29 = 2
    var_30 = [var_28, var_29, var_28]
    var_31 = {var_27: var_30}
    var_32 = {var_1: var_22, var_2: var_26, var_3: var_31}
    var_33 = module_0.if_then_else_from_json_schema(var_32, var_0)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.if_clause).__module__}.{type(var_33.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_33.then_clause).__module__}.{type(var_33.then_clause).__qualname__}' == 'typesystem.fields.Array'
    assert f'{type(var_33.else_clause).__module__}.{type(var_33.else_clause).__qualname__}' == 'typesystem.fields.Choice'

def test_case_25():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
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
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_17 = 'array'
    var_18 = module_0.from_json_schema(var_12)
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
    var_19 = 'object'
    var_20 = {var_4: var_19}
    var_21 = module_0.from_json_schema(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Object'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.properties == {}
    assert var_21.pattern_properties == {}
    assert var_21.additional_properties is None
    assert var_21.property_names is None
    assert var_21.min_properties is None
    assert var_21.max_properties is None
    assert var_21.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_22 = 'enum'
    var_23 = 2
    var_24 = 3
    var_25 = [var_0, var_23, var_24]
    var_26 = {var_22: var_25}
    var_27 = module_0.from_json_schema(var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Choice'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.choices == [(True, True), (2, 2), (3, 3)]
    assert var_27.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_28 = 'const'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = module_0.from_json_schema(var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Const'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.const == 'value'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_32 = 'minLength'
    var_33 = 'maxLength'
    var_34 = 5
    var_35 = 10
    var_36 = {var_4: var_5, var_32: var_34, var_33: var_35}
    var_37 = module_0.from_json_schema(var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.String'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.allow_blank is False
    assert var_37.trim_whitespace is True
    assert var_37.max_length == 10
    assert var_37.min_length == 5
    assert var_37.format is None
    assert var_37.coerce_types is False
    assert var_37.pattern is None
    assert var_37.pattern_regex is None
    var_38 = 'maximum'
    var_39 = 100
    var_40 = {var_4: var_11, var_14: var_2, var_38: var_39}
    var_41 = module_0.from_json_schema(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Float'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.minimum is None
    assert var_41.maximum == 100
    assert var_41.exclusive_minimum is None
    assert var_41.exclusive_maximum is None
    assert var_41.multiple_of is None
    assert var_41.precision is None
    assert var_41.coerce_types is False
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = {var_4: var_17, var_42: var_0, var_43: var_34}
    var_45 = module_0.from_json_schema(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Array'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.items is None
    assert var_45.additional_items is True
    assert var_45.min_items is True
    assert var_45.max_items == 5
    assert var_45.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_46 = 'properties'
    var_47 = 'name'
    var_48 = 'age'
    var_49 = {var_4: var_5}
    var_50 = {var_4: var_8}
    var_51 = {var_47: var_49, var_48: var_50}
    var_52 = {var_4: var_19, var_46: var_51}
    var_53 = module_0.from_json_schema(var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.Object'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert f'{type(var_53.properties).__module__}.{type(var_53.properties).__qualname__}' == 'builtins.dict'
    assert len(var_53.properties) == 2
    assert var_53.pattern_properties == {}
    assert var_53.additional_properties is None
    assert var_53.property_names is None
    assert var_53.min_properties is None
    assert var_53.max_properties is None
    assert var_53.required == []
    var_54 = 'allOf'
    var_55 = {var_4: var_5}
    var_56 = {var_32: var_34}
    var_57 = [var_55, var_56]
    var_58 = {var_54: var_57}
    var_59 = module_0.from_json_schema(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.all_of).__module__}.{type(var_59.all_of).__qualname__}' == 'builtins.list'
    assert len(var_59.all_of) == 2
    var_60 = 'anyOf'
    var_61 = {var_4: var_5}
    var_62 = {var_4: var_8}
    var_63 = [var_61, var_62]
    var_64 = {var_60: var_63}
    var_65 = module_0.from_json_schema(var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Union'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert f'{type(var_65.any_of).__module__}.{type(var_65.any_of).__qualname__}' == 'builtins.list'
    assert len(var_65.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_66 = 'oneOf'
    var_67 = {var_4: var_5}
    var_68 = {var_4: var_8}
    var_69 = [var_67, var_68]
    var_70 = {var_66: var_69}
    var_71 = module_0.from_json_schema(var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert f'{type(var_71.one_of).__module__}.{type(var_71.one_of).__qualname__}' == 'builtins.list'
    assert len(var_71.one_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_72 = 'not'
    var_73 = {var_4: var_5}
    var_74 = {var_72: var_73}
    var_75 = module_0.from_json_schema(var_74)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.composites.Not'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert f'{type(var_75.negated).__module__}.{type(var_75.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_76 = 'if'
    var_77 = 'then'
    var_78 = 'else'
    var_79 = {var_4: var_5}
    var_80 = {var_32: var_34}
    var_81 = {var_4: var_8}
    var_82 = {var_76: var_79, var_77: var_80, var_78: var_81}
    var_83 = module_0.from_json_schema(var_82)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert f'{type(var_83.if_clause).__module__}.{type(var_83.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_83.then_clause).__module__}.{type(var_83.then_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_83.else_clause).__module__}.{type(var_83.else_clause).__qualname__}' == 'typesystem.fields.Integer'
    var_84 = module_3.Definitions()
    assert f'{type(var_84).__module__}.{type(var_84).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_84) == 0
    var_85 = '$ref'
    var_86 = '#/components/schemas/User'
    var_87 = {var_85: var_86}
    var_88 = module_0.from_json_schema(var_87, var_84)
    assert f'{type(var_88).__module__}.{type(var_88).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_88.title == ''
    assert var_88.description == ''
    assert var_88.allow_null is False
    assert var_88.read_only is False
    assert var_88.to == '#/components/schemas/User'
    assert f'{type(var_88.definitions).__module__}.{type(var_88.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_88.definitions) == 0
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    var_89 = {}
    var_90 = module_0.from_json_schema(var_89)
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'typesystem.fields.Any'
    assert var_90.title == ''
    assert var_90.description == ''
    assert var_90.allow_null is False
    assert var_90.read_only is False
    var_91 = 'pattern'
    var_92 = '^[a-z]+$'
    var_93 = {var_4: var_5, var_91: var_92}
    var_94 = module_0.from_json_schema(var_93)
    assert f'{type(var_94).__module__}.{type(var_94).__qualname__}' == 'typesystem.fields.String'
    assert var_94.title == ''
    assert var_94.description == ''
    assert var_94.allow_null is False
    assert var_94.read_only is False
    assert var_94.default == ''
    assert var_94.allow_blank is True
    assert var_94.trim_whitespace is True
    assert var_94.max_length is None
    assert var_94.min_length is None
    assert var_94.format is None
    assert var_94.coerce_types is False
    assert var_94.pattern == '^[a-z]+$'
    assert f'{type(var_94.pattern_regex).__module__}.{type(var_94.pattern_regex).__qualname__}' == 're.Pattern'
    var_95 = 'components'
    var_96 = 'schemas'
    var_97 = 'User'
    var_98 = {var_97: var_36}
    var_99 = {var_96: var_98}
    var_100 = {var_4: var_19, var_95: var_99}
    var_101 = module_0.from_json_schema(var_100)
    assert f'{type(var_101).__module__}.{type(var_101).__qualname__}' == 'typesystem.fields.Object'
    assert var_101.title == ''
    assert var_101.description == ''
    assert var_101.allow_null is False
    assert var_101.read_only is False
    assert var_101.properties == {}
    assert var_101.pattern_properties == {}
    assert var_101.additional_properties is None
    assert var_101.property_names is None
    assert var_101.min_properties is None
    assert var_101.max_properties is None
    assert var_101.required == []

def test_case_26():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    var_5 = 'integer'
    var_6 = {var_0: var_5}
    var_7 = module_3.Definitions()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_7) == 0
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_8 = module_0.type_from_json_schema(var_6, var_7)
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
    var_9 = 'number'
    var_10 = {var_0: var_9}
    var_11 = module_3.Definitions()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_11) == 0
    var_12 = module_0.type_from_json_schema(var_10, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Float'
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
    assert var_12.coerce_types is False
    var_13 = 'boolean'
    var_14 = {var_0: var_13}
    var_15 = module_3.Definitions()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_15) == 0
    var_16 = module_0.type_from_json_schema(var_14, var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.coerce_types is False
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_17 = 'array'
    var_18 = {var_0: var_17}
    var_19 = module_3.Definitions()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_19) == 0
    var_20 = module_0.type_from_json_schema(var_18, var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Array'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.items is None
    assert var_20.additional_items is True
    assert var_20.min_items == 0
    assert var_20.max_items is None
    assert var_20.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_21 = 'object'
    var_22 = {var_0: var_21}
    var_23 = module_3.Definitions()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_23) == 0
    var_24 = module_0.type_from_json_schema(var_22, var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Object'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.properties == {}
    assert var_24.pattern_properties == {}
    assert var_24.additional_properties is None
    assert var_24.property_names is None
    assert var_24.min_properties is None
    assert var_24.max_properties is None
    assert var_24.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_25 = [var_1, var_5]
    var_26 = {var_0: var_25}
    var_27 = module_3.Definitions()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_27) == 0
    var_28 = module_0.type_from_json_schema(var_26, var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Union'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.any_of).__module__}.{type(var_28.any_of).__qualname__}' == 'builtins.list'
    assert len(var_28.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_29 = 'null'
    var_30 = [var_1, var_29]
    var_31 = {var_0: var_30}
    var_32 = module_3.Definitions()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_32) == 0
    var_33 = module_0.type_from_json_schema(var_31, var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.String'
    assert var_33.default is None
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is True
    assert var_33.read_only is False
    assert var_33.allow_blank is True
    assert var_33.trim_whitespace is True
    assert var_33.max_length is None
    assert var_33.min_length is None
    assert var_33.format is None
    assert var_33.coerce_types is False
    assert var_33.pattern is None
    assert var_33.pattern_regex is None
    var_34 = {var_0: var_29}
    var_35 = module_3.Definitions()
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_35) == 0
    var_36 = module_0.type_from_json_schema(var_34, var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Const'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.const is None
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_37 = {}
    var_38 = module_3.Definitions()
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_38) == 0
    var_39 = module_0.type_from_json_schema(var_37, var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Union'
    assert var_39.default is None
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is True
    assert var_39.read_only is False
    assert f'{type(var_39.any_of).__module__}.{type(var_39.any_of).__qualname__}' == 'builtins.list'
    assert len(var_39.any_of) == 5
    var_40 = 'minLength'
    var_41 = 5
    var_42 = {var_0: var_1, var_40: var_41}
    var_43 = module_3.Definitions()
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_43) == 0
    var_44 = module_0.type_from_json_schema(var_42, var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.String'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert var_44.allow_blank is False
    assert var_44.trim_whitespace is True
    assert var_44.max_length is None
    assert var_44.min_length == 5
    assert var_44.format is None
    assert var_44.coerce_types is False
    assert var_44.pattern is None
    assert var_44.pattern_regex is None
    var_45 = 'minimum'
    var_46 = 10
    var_47 = {var_0: var_5, var_45: var_46}
    var_48 = module_3.Definitions()
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_48) == 0
    var_49 = module_0.type_from_json_schema(var_47, var_48)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.fields.Integer'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert var_49.minimum == 10
    assert var_49.maximum is None
    assert var_49.exclusive_minimum is None
    assert var_49.exclusive_maximum is None
    assert var_49.multiple_of is None
    assert var_49.precision is None
    assert var_49.coerce_types is False
    var_50 = 'items'
    var_51 = {var_0: var_1}
    var_52 = {var_0: var_17, var_50: var_51}
    var_53 = module_3.Definitions()
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_53) == 0
    var_54 = module_0.type_from_json_schema(var_52, var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Array'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert f'{type(var_54.items).__module__}.{type(var_54.items).__qualname__}' == 'typesystem.fields.String'
    assert var_54.additional_items is True
    assert var_54.min_items == 0
    assert var_54.max_items is None
    assert var_54.unique_items is False
    var_55 = 'properties'
    var_56 = 'name'
    var_57 = {var_0: var_1}
    var_58 = {var_56: var_57}
    var_59 = {var_0: var_21, var_55: var_58}
    var_60 = module_3.Definitions()
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_60) == 0
    var_61 = module_0.type_from_json_schema(var_59, var_60)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.fields.Object'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert f'{type(var_61.properties).__module__}.{type(var_61.properties).__qualname__}' == 'builtins.dict'
    assert len(var_61.properties) == 1
    assert var_61.pattern_properties == {}
    assert var_61.additional_properties is None
    assert var_61.property_names is None
    assert var_61.min_properties is None
    assert var_61.max_properties is None
    assert var_61.required == []
    var_62 = [var_1, var_5]
    var_63 = 1
    var_64 = {var_0: var_62, var_40: var_63}
    var_65 = module_3.Definitions()
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_65) == 0
    var_66 = module_0.type_from_json_schema(var_64, var_65)
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.fields.Union'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert f'{type(var_66.any_of).__module__}.{type(var_66.any_of).__qualname__}' == 'builtins.list'
    assert len(var_66.any_of) == 2
    var_67 = 'pattern'
    var_68 = '^[a-z]+$'
    var_69 = {var_0: var_1, var_67: var_68}
    var_70 = module_3.Definitions()
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_70) == 0
    var_71 = module_0.type_from_json_schema(var_69, var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.String'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.default == ''
    assert var_71.allow_blank is True
    assert var_71.trim_whitespace is True
    assert var_71.max_length is None
    assert var_71.min_length is None
    assert var_71.format is None
    assert var_71.coerce_types is False
    assert var_71.pattern == '^[a-z]+$'
    assert f'{type(var_71.pattern_regex).__module__}.{type(var_71.pattern_regex).__qualname__}' == 're.Pattern'
    var_72 = 'multipleOf'
    var_73 = 0.5
    var_74 = {var_0: var_9, var_72: var_73}
    var_75 = module_3.Definitions()
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_75) == 0
    var_76 = module_0.type_from_json_schema(var_74, var_75)
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.fields.Float'
    assert var_76.title == ''
    assert var_76.description == ''
    assert var_76.allow_null is False
    assert var_76.read_only is False
    assert var_76.minimum is None
    assert var_76.maximum is None
    assert var_76.exclusive_minimum is None
    assert var_76.exclusive_maximum is None
    assert var_76.multiple_of == pytest.approx(0.5, abs=0.01, rel=0.01)
    assert var_76.precision is None
    assert var_76.coerce_types is False
    var_77 = 'exclusiveMinimum'
    var_78 = 'exclusiveMaximum'
    var_79 = 0
    var_80 = 100
    var_81 = {var_0: var_5, var_77: var_79, var_78: var_80}
    var_82 = module_3.Definitions()
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_82) == 0
    var_83 = module_0.type_from_json_schema(var_81, var_82)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.fields.Integer'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert var_83.minimum is None
    assert var_83.maximum is None
    assert var_83.exclusive_minimum == 0
    assert var_83.exclusive_maximum == 100
    assert var_83.multiple_of is None
    assert var_83.precision is None
    assert var_83.coerce_types is False

@pytest.mark.xfail(strict=True)
def test_case_27():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_2.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
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
    var_5 = True
    var_6 = 5
    var_7 = 10
    var_8 = module_1.String(max_length=var_7, min_length=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length == 10
    assert var_8.min_length == 5
    assert var_8.format is None
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    var_9 = module_0.to_json_schema(var_8)
    var_10 = '^[a-z]+$'
    var_11 = module_1.String(pattern=var_10)
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
    assert var_11.pattern == '^[a-z]+$'
    assert f'{type(var_11.pattern_regex).__module__}.{type(var_11.pattern_regex).__qualname__}' == 're.Pattern'
    var_12 = 'email'
    var_13 = module_1.String(format=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.String'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.allow_blank is False
    assert var_13.trim_whitespace is True
    assert var_13.max_length is None
    assert var_13.min_length is None
    assert var_13.format == 'email'
    assert var_13.coerce_types is True
    assert var_13.pattern is None
    assert var_13.pattern_regex is None
    var_14 = module_0.to_json_schema(var_13)
    var_15 = module_1.Integer()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Integer'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is None
    assert var_15.maximum is None
    assert var_15.exclusive_minimum is None
    assert var_15.exclusive_maximum is None
    assert var_15.multiple_of is None
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_0.to_json_schema(var_15)
    var_18 = 0
    var_19 = 100
    var_20 = module_0.to_json_schema(var_15)
    var_21 = module_1.Integer(exclusive_minimum=var_18, exclusive_maximum=var_19)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Integer'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.minimum is None
    assert var_21.maximum is None
    assert var_21.exclusive_minimum == 0
    assert var_21.exclusive_maximum == 100
    assert var_21.multiple_of is None
    assert var_21.precision is None
    assert var_21.coerce_types is True
    var_22 = module_0.to_json_schema(var_21)
    var_23 = module_1.Integer(multiple_of=var_6)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Integer'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.minimum is None
    assert var_23.maximum is None
    assert var_23.exclusive_minimum is None
    assert var_23.exclusive_maximum is None
    assert var_23.multiple_of == 5
    assert var_23.precision is None
    assert var_23.coerce_types is True
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_1.Float()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Float'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.minimum is None
    assert var_25.maximum is None
    assert var_25.exclusive_minimum is None
    assert var_25.exclusive_maximum is None
    assert var_25.multiple_of is None
    assert var_25.precision is None
    assert var_25.coerce_types is True
    var_26 = module_0.to_json_schema(var_25)
    var_27 = module_1.Boolean()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_28 = module_0.to_json_schema(var_27)
    var_29 = module_1.Boolean()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert var_29.coerce_types is True
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
    var_32 = module_1.Array(min_items=var_5, max_items=var_7)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Array'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.items is None
    assert var_32.additional_items is False
    assert var_32.min_items is True
    assert var_32.max_items == 10
    assert var_32.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_33 = module_0.to_json_schema(var_32)
    var_34 = module_1.Array(unique_items=var_5)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Array'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.items is None
    assert var_34.additional_items is False
    assert var_34.min_items is None
    assert var_34.max_items is None
    assert var_34.unique_items is True
    var_35 = module_0.to_json_schema(var_34)
    var_36 = module_0.to_json_schema(var_34)
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
    var_38 = module_1.Array(additional_items=var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Array'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.items is None
    assert f'{type(var_38.additional_items).__module__}.{type(var_38.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_38.min_items is None
    assert var_38.max_items is None
    assert var_38.unique_items is False
    var_39 = module_0.to_json_schema(var_38)
    var_40 = module_0.to_json_schema(var_25)
    var_41 = 'namDe'
    var_42 = 'age'
    var_43 = module_1.String()
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
    var_44 = module_1.Integer()
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Integer'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert var_44.minimum is None
    assert var_44.maximum is None
    assert var_44.exclusive_minimum is None
    assert var_44.exclusive_maximum is None
    assert var_44.multiple_of is None
    assert var_44.precision is None
    assert var_44.coerce_types is True
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = module_1.Object(properties=var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Object'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.properties).__module__}.{type(var_46.properties).__qualname__}' == 'builtins.dict'
    assert len(var_46.properties) == 2
    assert var_46.pattern_properties == {}
    assert var_46.additional_properties is True
    assert var_46.property_names is None
    assert var_46.min_properties is None
    assert var_46.max_properties is None
    assert var_46.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
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
    var_49 = False
    var_50 = module_1.Object(additional_properties=var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.fields.Object'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert var_50.properties == {}
    assert var_50.pattern_properties == {}
    assert var_50.additional_properties is False
    assert var_50.property_names is None
    assert var_50.min_properties is None
    assert var_50.max_properties is None
    assert var_50.required == []
    var_51 = module_0.to_json_schema(var_50)
    var_52 = module_1.String()
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
    var_53 = module_1.Object(additional_properties=var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.Object'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert var_53.properties == {}
    assert var_53.pattern_properties == {}
    assert f'{type(var_53.additional_properties).__module__}.{type(var_53.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_53.property_names is None
    assert var_53.min_properties is None
    assert var_53.max_properties is None
    assert var_53.required == []
    var_54 = module_1.Object(property_names=var_13)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Object'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert var_54.properties == {}
    assert var_54.pattern_properties == {}
    assert var_54.additional_properties is True
    assert f'{type(var_54.property_names).__module__}.{type(var_54.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_54.min_properties is None
    assert var_54.max_properties is None
    assert var_54.required == []
    var_55 = module_0.to_json_schema(var_54)
    var_56 = module_1.Object(min_properties=var_5, max_properties=var_6)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Object'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert var_56.properties == {}
    assert var_56.pattern_properties == {}
    assert var_56.additional_properties is True
    assert var_56.property_names is None
    assert var_56.min_properties is True
    assert var_56.max_properties == 5
    assert var_56.required == []
    var_57 = module_0.to_json_schema(var_56)
    var_58 = 'red'
    var_59 = (var_58, var_58)
    var_60 = (var_20, var_20)
    var_61 = (var_12, var_12)
    var_62 = [var_59, var_60, var_61]
    var_63 = module_1.Choice(choices=var_62)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.fields.Choice'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert var_63.choices == [('red', 'red'), ({'type': 'integer'}, {'type': 'integer'}), ('email', 'email')]
    assert var_63.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_64 = module_0.to_json_schema(var_63)
    var_65 = 'constant_value'
    var_66 = module_1.Const(var_65)
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.fields.Const'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert var_66.const == 'constant_value'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_67 = module_0.to_json_schema(var_66)
    var_68 = module_1.String()
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
    var_69 = [var_68, var_44]
    var_70 = module_1.Union(var_69)
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.fields.Union'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert f'{type(var_70.any_of).__module__}.{type(var_70.any_of).__qualname__}' == 'builtins.list'
    assert len(var_70.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_71 = module_0.to_json_schema(var_70)
    var_72 = 'anyOf'
    var_73 = var_71[var_72]
    var_74 = len(var_73)
    assert var_74 == 2
    var_75 = module_1.String()
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.fields.String'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert var_75.allow_blank is False
    assert var_75.trim_whitespace is True
    assert var_75.max_length is None
    assert var_75.min_length is None
    assert var_75.format is None
    assert var_75.coerce_types is True
    assert var_75.pattern is None
    assert var_75.pattern_regex is None
    var_76 = module_1.Integer()
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.fields.Integer'
    assert var_76.title == ''
    assert var_76.description == ''
    assert var_76.allow_null is False
    assert var_76.read_only is False
    assert var_76.minimum is None
    assert var_76.maximum is None
    assert var_76.exclusive_minimum is None
    assert var_76.exclusive_maximum is None
    assert var_76.multiple_of is None
    assert var_76.precision is None
    assert var_76.coerce_types is True
    var_77 = module_2.OneOf(var_69)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert f'{type(var_77.one_of).__module__}.{type(var_77.one_of).__qualname__}' == 'builtins.list'
    assert len(var_77.one_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_78 = module_0.to_json_schema(var_77)
    var_79 = 'oneOf'
    var_80 = var_78[var_79]
    var_81 = len(var_80)
    assert var_81 == 2
    var_82 = [var_68, var_11]
    var_83 = module_2.AllOf(var_82)
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert f'{type(var_83.all_of).__module__}.{type(var_83.all_of).__qualname__}' == 'builtins.list'
    assert len(var_83.all_of) == 2
    var_84 = module_0.to_json_schema(var_83)
    var_85 = 'allOf'
    var_86 = var_84[var_85]
    var_87 = len(var_86)
    assert var_87 == 2
    var_88 = module_1.String()
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
    assert var_88.pattern is None
    assert var_88.pattern_regex is None
    var_89 = module_2.Not(var_88)
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.composites.Not'
    assert var_89.title == ''
    assert var_89.description == ''
    assert var_89.allow_null is False
    assert var_89.read_only is False
    assert f'{type(var_89.negated).__module__}.{type(var_89.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_90 = module_0.to_json_schema(var_89)
    var_91 = module_1.String()
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
    var_92 = module_1.Integer()
    assert f'{type(var_92).__module__}.{type(var_92).__qualname__}' == 'typesystem.fields.Integer'
    assert var_92.title == ''
    assert var_92.description == ''
    assert var_92.allow_null is False
    assert var_92.read_only is False
    assert var_92.minimum is None
    assert var_92.maximum is None
    assert var_92.exclusive_minimum is None
    assert var_92.exclusive_maximum is None
    assert var_92.multiple_of is None
    assert var_92.precision is None
    assert var_92.coerce_types is True
    var_93 = module_1.Boolean()
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert var_93.coerce_types is True
    var_94 = module_2.IfThenElse(var_91, var_92, var_93)
    assert f'{type(var_94).__module__}.{type(var_94).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_94.title == ''
    assert var_94.description == ''
    assert var_94.allow_null is False
    assert var_94.read_only is False
    assert f'{type(var_94.if_clause).__module__}.{type(var_94.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_94.then_clause).__module__}.{type(var_94.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_94.else_clause).__module__}.{type(var_94.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_95 = module_0.to_json_schema(var_94)
    var_96 = module_1.String()
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.fields.String'
    assert var_96.title == ''
    assert var_96.description == ''
    assert var_96.allow_null is False
    assert var_96.read_only is False
    assert var_96.allow_blank is False
    assert var_96.trim_whitespace is True
    assert var_96.max_length is None
    assert var_96.min_length is None
    assert var_96.format is None
    assert var_96.coerce_types is True
    assert var_96.pattern is None
    assert var_96.pattern_regex is None
    var_97 = module_2.IfThenElse(var_96, var_73)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_97.title == ''
    assert var_97.description == ''
    assert var_97.allow_null is False
    assert var_97.read_only is False
    assert f'{type(var_97.if_clause).__module__}.{type(var_97.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert var_97.then_clause == [{'type': 'string', 'minLength': 1}, {'type': 'integer'}]
    assert f'{type(var_97.else_clause).__module__}.{type(var_97.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_0.to_json_schema(var_97)

def test_case_28():
    var_0 = module_2.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    var_4 = 'minLength'
    var_5 = True
    var_6 = module_0.to_json_schema(var_2)
    var_7 = 5
    var_8 = 10
    var_9 = module_1.String(max_length=var_8, min_length=var_7)
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
    assert var_9.coerce_types is True
    assert var_9.pattern is None
    assert var_9.pattern_regex is None
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_1.String(pattern=var_4)
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
    assert var_11.pattern == 'minLength'
    assert f'{type(var_11.pattern_regex).__module__}.{type(var_11.pattern_regex).__qualname__}' == 're.Pattern'
    var_12 = module_0.to_json_schema(var_11)
    var_13 = 'email'
    var_14 = module_1.String(format=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.String'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.allow_blank is False
    assert var_14.trim_whitespace is True
    assert var_14.max_length is None
    assert var_14.min_length is None
    assert var_14.format == 'email'
    assert var_14.coerce_types is True
    assert var_14.pattern is None
    assert var_14.pattern_regex is None
    var_15 = module_1.Integer()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Integer'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is None
    assert var_15.maximum is None
    assert var_15.exclusive_minimum is None
    assert var_15.exclusive_maximum is None
    assert var_15.multiple_of is None
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
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
    var_18 = module_0.to_json_schema(var_17)
    var_19 = 0
    var_20 = 100
    var_21 = module_1.Integer(minimum=var_19, maximum=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Integer'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.minimum == 0
    assert var_21.maximum == 100
    assert var_21.exclusive_minimum is None
    assert var_21.exclusive_maximum is None
    assert var_21.multiple_of is None
    assert var_21.precision is None
    assert var_21.coerce_types is True
    var_22 = module_0.to_json_schema(var_21)
    var_23 = module_1.Integer(exclusive_minimum=var_19, exclusive_maximum=var_20)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Integer'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.minimum is None
    assert var_23.maximum is None
    assert var_23.exclusive_minimum == 0
    assert var_23.exclusive_maximum == 100
    assert var_23.multiple_of is None
    assert var_23.precision is None
    assert var_23.coerce_types is True
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_1.Integer(multiple_of=var_7)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Integer'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.minimum is None
    assert var_25.maximum is None
    assert var_25.exclusive_minimum is None
    assert var_25.exclusive_maximum is None
    assert var_25.multiple_of == 5
    assert var_25.precision is None
    assert var_25.coerce_types is True
    var_26 = module_0.to_json_schema(var_25)
    var_27 = module_1.Float()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Float'
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
    var_28 = module_0.to_json_schema(var_27)
    var_29 = module_1.Float()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Float'
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
    var_30 = module_1.Boolean()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_31 = module_0.to_json_schema(var_30)
    var_32 = module_1.Boolean()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.coerce_types is True
    var_33 = module_0.to_json_schema(var_32)
    var_34 = module_1.String()
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.String'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.allow_blank is False
    assert var_34.trim_whitespace is True
    assert var_34.max_length is None
    assert var_34.min_length is None
    assert var_34.format is None
    assert var_34.coerce_types is True
    assert var_34.pattern is None
    assert var_34.pattern_regex is None
    var_35 = module_1.Array(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Array'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.items).__module__}.{type(var_35.items).__qualname__}' == 'typesystem.fields.String'
    assert var_35.additional_items is False
    assert var_35.min_items is None
    assert var_35.max_items is None
    assert var_35.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_36 = module_0.to_json_schema(var_35)
    var_37 = module_1.Array(min_items=var_5, max_items=var_8)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Array'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.items is None
    assert var_37.additional_items is False
    assert var_37.min_items is True
    assert var_37.max_items == 10
    assert var_37.unique_items is False
    var_38 = module_0.to_json_schema(var_37)
    var_39 = module_1.Array(unique_items=var_5)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Array'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.items is None
    assert var_39.additional_items is False
    assert var_39.min_items is None
    assert var_39.max_items is None
    assert var_39.unique_items is True
    var_40 = module_0.to_json_schema(var_39)
    var_41 = False
    var_42 = module_1.Array(additional_items=var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.Array'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert var_42.items is None
    assert var_42.additional_items is False
    assert var_42.min_items is None
    assert var_42.max_items is None
    assert var_42.unique_items is False
    var_43 = var_37.serialize(var_40)
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
    var_45 = module_1.Array(additional_items=var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Array'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.items is None
    assert f'{type(var_45.additional_items).__module__}.{type(var_45.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_45.min_items is None
    assert var_45.max_items is None
    assert var_45.unique_items is False
    var_46 = module_0.to_json_schema(var_45)
    var_47 = module_1.Object()
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.Object'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert var_47.properties == {}
    assert var_47.pattern_properties == {}
    assert var_47.additional_properties is True
    assert var_47.property_names is None
    assert var_47.min_properties is None
    assert var_47.max_properties is None
    assert var_47.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_48 = module_0.to_json_schema(var_47)
    var_49 = 'namDe'
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
    var_51 = var_14.get_default_value()
    var_52 = {var_49: var_50, var_4: var_51}
    with pytest.raises(AssertionError):
        module_1.Object(properties=var_52)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_1.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_2.NeverMatch()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = module_0.to_json_schema(var_1)
    assert var_2 is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
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
    var_5 = 'maxLength'
    var_6 = True
    var_7 = module_1.String()
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
    var_8 = module_0.to_json_schema(var_7)
    var_9 = 5
    var_10 = 10
    var_11 = module_1.String(max_length=var_10, min_length=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.String'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.allow_blank is False
    assert var_11.trim_whitespace is True
    assert var_11.max_length == 10
    assert var_11.min_length == 5
    assert var_11.format is None
    assert var_11.coerce_types is True
    assert var_11.pattern is None
    assert var_11.pattern_regex is None
    var_12 = module_0.to_json_schema(var_11)
    var_13 = module_1.String(pattern=var_5)
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
    assert var_13.pattern == 'maxLength'
    assert f'{type(var_13.pattern_regex).__module__}.{type(var_13.pattern_regex).__qualname__}' == 're.Pattern'
    var_14 = 'email'
    var_15 = module_1.String(format=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.String'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.allow_blank is False
    assert var_15.trim_whitespace is True
    assert var_15.max_length is None
    assert var_15.min_length is None
    assert var_15.format == 'email'
    assert var_15.coerce_types is True
    assert var_15.pattern is None
    assert var_15.pattern_regex is None
    var_16 = module_0.to_json_schema(var_15)
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
    var_18 = module_0.to_json_schema(var_17)
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
    var_20 = module_0.to_json_schema(var_19)
    var_21 = 0
    var_22 = 100
    var_23 = module_1.Integer(minimum=var_21, maximum=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Integer'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.minimum == 0
    assert var_23.maximum == 100
    assert var_23.exclusive_minimum is None
    assert var_23.exclusive_maximum is None
    assert var_23.multiple_of is None
    assert var_23.precision is None
    assert var_23.coerce_types is True
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_0.to_json_schema(var_19)
    var_26 = module_1.Integer(multiple_of=var_9)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Integer'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.minimum is None
    assert var_26.maximum is None
    assert var_26.exclusive_minimum is None
    assert var_26.exclusive_maximum is None
    assert var_26.multiple_of == 5
    assert var_26.precision is None
    assert var_26.coerce_types is True
    var_27 = module_0.to_json_schema(var_26)
    var_28 = module_1.Float()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Float'
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
    var_29 = module_0.to_json_schema(var_28)
    var_30 = module_1.Float()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Float'
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
    var_31 = module_0.to_json_schema(var_30)
    var_32 = module_1.Boolean()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_33 = module_0.to_json_schema(var_32)
    var_34 = module_1.Boolean()
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.coerce_types is True
    var_35 = module_0.to_json_schema(var_34)
    var_36 = module_1.Array()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Array'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.items is None
    assert var_36.additional_items is False
    assert var_36.min_items is None
    assert var_36.max_items is None
    assert var_36.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
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
    var_39 = module_1.Array(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Array'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.items).__module__}.{type(var_39.items).__qualname__}' == 'typesystem.fields.String'
    assert var_39.additional_items is False
    assert var_39.min_items is None
    assert var_39.max_items is None
    assert var_39.unique_items is False
    var_40 = module_0.to_json_schema(var_39)
    var_41 = module_1.Array(min_items=var_6, max_items=var_10)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Array'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.items is None
    assert var_41.additional_items is False
    assert var_41.min_items is True
    assert var_41.max_items == 10
    assert var_41.unique_items is False
    var_42 = module_0.to_json_schema(var_41)
    var_43 = var_38.get_default_value()
    module_0.to_json_schema(var_43)

def test_case_30():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_2.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = 5
    var_7 = 10
    var_8 = module_0.to_json_schema(var_5)
    var_9 = '^[a-z]+$'
    var_10 = module_1.String(pattern=var_9)
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
    assert var_10.pattern == '^[a-z]+$'
    assert f'{type(var_10.pattern_regex).__module__}.{type(var_10.pattern_regex).__qualname__}' == 're.Pattern'
    var_11 = module_0.to_json_schema(var_10)
    var_12 = module_1.String(format=var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.allow_blank is False
    assert var_12.trim_whitespace is True
    assert var_12.max_length is None
    assert var_12.min_length is None
    assert var_12.format == '^[a-z]+$'
    assert var_12.coerce_types is True
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    var_13 = module_0.to_json_schema(var_12)
    var_14 = module_1.Integer()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum is None
    assert var_14.maximum is None
    assert var_14.exclusive_minimum is None
    assert var_14.exclusive_maximum is None
    assert var_14.multiple_of is None
    assert var_14.precision is None
    assert var_14.coerce_types is True
    var_15 = module_0.to_json_schema(var_14)
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
    var_17 = module_0.to_json_schema(var_16)
    var_18 = 100
    var_19 = module_0.to_json_schema(var_14)
    var_20 = module_1.Integer(exclusive_minimum=var_6, exclusive_maximum=var_18)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Integer'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.minimum is None
    assert var_20.maximum is None
    assert var_20.exclusive_minimum == 5
    assert var_20.exclusive_maximum == 100
    assert var_20.multiple_of is None
    assert var_20.precision is None
    assert var_20.coerce_types is True
    var_21 = module_0.to_json_schema(var_20)
    var_22 = module_1.Integer(multiple_of=var_6)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Integer'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.minimum is None
    assert var_22.maximum is None
    assert var_22.exclusive_minimum is None
    assert var_22.exclusive_maximum is None
    assert var_22.multiple_of == 5
    assert var_22.precision is None
    assert var_22.coerce_types is True
    var_23 = module_0.to_json_schema(var_22)
    var_24 = var_15.items()
    var_25 = module_1.Float()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Float'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.minimum is None
    assert var_25.maximum is None
    assert var_25.exclusive_minimum is None
    assert var_25.exclusive_maximum is None
    assert var_25.multiple_of is None
    assert var_25.precision is None
    assert var_25.coerce_types is True
    var_26 = module_0.to_json_schema(var_25)
    var_27 = module_1.Boolean()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_28 = module_0.to_json_schema(var_27)
    var_29 = module_1.Array(var_12)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Array'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.items).__module__}.{type(var_29.items).__qualname__}' == 'typesystem.fields.String'
    assert var_29.additional_items is False
    assert var_29.min_items is None
    assert var_29.max_items is None
    assert var_29.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_30 = module_0.to_json_schema(var_29)
    var_31 = module_1.Array(min_items=var_4, max_items=var_7)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Array'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.items is None
    assert var_31.additional_items is False
    assert var_31.min_items is True
    assert var_31.max_items == 10
    assert var_31.unique_items is False
    var_32 = var_12.get_default_value()

def test_case_31():
    var_0 = module_1.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = 'email'
    var_2 = module_1.String(format=var_1)
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_32():
    var_0 = module_1.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_2.NeverMatch()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
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
    var_6 = module_0.to_json_schema(var_3)
    var_7 = 'email'
    var_8 = module_1.String(format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length is None
    assert var_8.min_length is None
    assert var_8.format == 'email'
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    var_9 = module_0.to_json_schema(var_8)
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
    var_11 = module_0.to_json_schema(var_10)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_1.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_4.purge()
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
    var_2 = module_4.purge()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = True
    var_6 = 5
    var_7 = 10
    var_8 = module_0.to_json_schema(var_3)
    var_9 = '^[a-z]+$'
    var_10 = module_1.String(format=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length is None
    assert var_10.min_length is None
    assert var_10.format == '^[a-z]+$'
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = module_1.Integer()
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
    var_12 = module_0.get_standard_properties(var_10)
    var_13 = module_1.Integer()
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
    assert var_13.coerce_types is True
    var_14 = 0
    var_15 = 100
    var_16 = module_1.Integer(minimum=var_14, maximum=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Integer'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum == 0
    assert var_16.maximum == 100
    assert var_16.exclusive_minimum is None
    assert var_16.exclusive_maximum is None
    assert var_16.multiple_of is None
    assert var_16.precision is None
    assert var_16.coerce_types is True
    var_17 = module_0.to_json_schema(var_13)
    var_18 = None
    var_19 = module_1.Integer(minimum=var_15, exclusive_minimum=var_15, multiple_of=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Integer'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.minimum == 100
    assert var_19.maximum is None
    assert var_19.exclusive_minimum == 100
    assert var_19.exclusive_maximum is None
    assert var_19.multiple_of is None
    assert var_19.precision is None
    assert var_19.coerce_types is True
    var_20 = module_5._EnumDict()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'enum._EnumDict'
    assert len(var_20) == 0
    var_21 = module_1.Float()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Float'
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
    var_22 = module_1.Boolean()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_23 = module_1.Boolean()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.coerce_types is True
    var_24 = module_4.purge()
    var_25 = module_1.Array(unique_items=var_5)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Array'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.items is None
    assert var_25.additional_items is False
    assert var_25.min_items is None
    assert var_25.max_items is None
    assert var_25.unique_items is True
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_26 = module_0.to_json_schema(var_25)
    var_27 = False
    var_28 = module_1.Array(additional_items=var_27)
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
    var_29 = module_0.to_json_schema(var_28)
    var_30 = module_1.String()
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
    var_31 = module_1.Array(additional_items=var_30)
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
    var_32 = module_0.to_json_schema(var_31)
    var_33 = module_1.Object()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Object'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.properties == {}
    assert var_33.pattern_properties == {}
    assert var_33.additional_properties is True
    assert var_33.property_names is None
    assert var_33.min_properties is None
    assert var_33.max_properties is None
    assert var_33.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_34 = module_0.to_json_schema(var_33)
    var_35 = 'namDe'
    var_36 = 'age'
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
    var_38 = {var_35: var_3, var_36: var_37}
    var_39 = module_1.Object(properties=var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Object'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.properties).__module__}.{type(var_39.properties).__qualname__}' == 'builtins.dict'
    assert len(var_39.properties) == 2
    assert var_39.pattern_properties == {}
    assert var_39.additional_properties is True
    assert var_39.property_names is None
    assert var_39.min_properties is None
    assert var_39.max_properties is None
    assert var_39.required == []
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
    var_42 = [var_35]
    var_43 = module_1.Object(properties=var_20, required=var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Object'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.properties == {}
    assert var_43.pattern_properties == {}
    assert var_43.additional_properties is True
    assert var_43.property_names is None
    assert var_43.min_properties is None
    assert var_43.max_properties is None
    assert var_43.required == ['namDe']
    var_44 = module_0.to_json_schema(var_43)
    var_45 = False
    var_46 = module_1.Object(additional_properties=var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Object'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert var_46.properties == {}
    assert var_46.pattern_properties == {}
    assert var_46.additional_properties is False
    assert var_46.property_names is None
    assert var_46.min_properties is None
    assert var_46.max_properties is None
    assert var_46.required == []
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
    var_49 = module_1.Object(additional_properties=var_48)
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
    var_50 = module_1.String(pattern=var_9)
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
    assert var_50.pattern == '^[a-z]+$'
    assert f'{type(var_50.pattern_regex).__module__}.{type(var_50.pattern_regex).__qualname__}' == 're.Pattern'
    var_51 = module_1.Object(property_names=var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Object'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert var_51.properties == {}
    assert var_51.pattern_properties == {}
    assert var_51.additional_properties is True
    assert f'{type(var_51.property_names).__module__}.{type(var_51.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_51.min_properties is None
    assert var_51.max_properties is None
    assert var_51.required == []
    var_52 = module_0.to_json_schema(var_51)
    var_53 = module_1.Object(min_properties=var_5, max_properties=var_6)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.Object'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert var_53.properties == {}
    assert var_53.pattern_properties == {}
    assert var_53.additional_properties is True
    assert var_53.property_names is None
    assert var_53.min_properties is True
    assert var_53.max_properties == 5
    assert var_53.required == []
    var_54 = 'constant_value'
    var_55 = module_1.Const(var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Const'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.const == 'constant_value'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
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
    var_57 = module_1.Integer()
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
    var_58 = var_31.serialize(var_5)
    assert var_58 is True
    var_59 = module_1.Integer()
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.fields.Integer'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert var_59.minimum is None
    assert var_59.maximum is None
    assert var_59.exclusive_minimum is None
    assert var_59.exclusive_maximum is None
    assert var_59.multiple_of is None
    assert var_59.precision is None
    assert var_59.coerce_types is True
    var_60 = module_1.String(max_length=var_7)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.String'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.allow_blank is False
    assert var_60.trim_whitespace is True
    assert var_60.max_length == 10
    assert var_60.min_length is None
    assert var_60.format is None
    assert var_60.coerce_types is True
    assert var_60.pattern is None
    assert var_60.pattern_regex is None
    var_61 = [var_56, var_60]
    var_62 = module_2.AllOf(var_61)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert f'{type(var_62.all_of).__module__}.{type(var_62.all_of).__qualname__}' == 'builtins.list'
    assert len(var_62.all_of) == 2
    var_63 = module_0.to_json_schema(var_62)
    var_64 = 'allOf'
    var_65 = var_63[var_64]
    var_66 = len(var_65)
    assert var_66 == 2
    var_67 = module_1.String()
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.fields.String'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    assert var_67.allow_blank is False
    assert var_67.trim_whitespace is True
    assert var_67.max_length is None
    assert var_67.min_length is None
    assert var_67.format is None
    assert var_67.coerce_types is True
    assert var_67.pattern is None
    assert var_67.pattern_regex is None
    var_68 = module_2.Not(var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.composites.Not'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert f'{type(var_68.negated).__module__}.{type(var_68.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_69 = module_0.to_json_schema(var_68)
    var_70 = module_1.String()
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.fields.String'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert var_70.allow_blank is False
    assert var_70.trim_whitespace is True
    assert var_70.max_length is None
    assert var_70.min_length is None
    assert var_70.format is None
    assert var_70.coerce_types is True
    assert var_70.pattern is None
    assert var_70.pattern_regex is None
    var_71 = module_1.Integer()
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Integer'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.minimum is None
    assert var_71.maximum is None
    assert var_71.exclusive_minimum is None
    assert var_71.exclusive_maximum is None
    assert var_71.multiple_of is None
    assert var_71.precision is None
    assert var_71.coerce_types is True
    var_72 = module_1.Boolean()
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert var_72.coerce_types is True
    var_73 = module_2.IfThenElse(var_70, var_71, var_72)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert f'{type(var_73.if_clause).__module__}.{type(var_73.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_73.then_clause).__module__}.{type(var_73.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_73.else_clause).__module__}.{type(var_73.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_74 = module_0.to_json_schema(var_73)
    var_75 = module_2.IfThenElse(var_58, var_66)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_75.title == ''
    assert var_75.description == ''
    assert var_75.allow_null is False
    assert var_75.read_only is False
    assert var_75.if_clause is True
    assert var_75.then_clause == 2
    assert f'{type(var_75.else_clause).__module__}.{type(var_75.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_4.findall(var_65, var_66)

@pytest.mark.xfail(strict=True)
def test_case_34():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_2.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_5._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    var_4 = True
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = 5
    var_7 = 10
    var_8 = module_0.to_json_schema(var_5)
    var_9 = '^[a-z]+$'
    var_10 = module_1.String(pattern=var_9)
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
    assert var_10.pattern == '^[a-z]+$'
    assert f'{type(var_10.pattern_regex).__module__}.{type(var_10.pattern_regex).__qualname__}' == 're.Pattern'
    var_11 = module_0.to_json_schema(var_10)
    var_12 = module_1.String(format=var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.allow_blank is False
    assert var_12.trim_whitespace is True
    assert var_12.max_length is None
    assert var_12.min_length is None
    assert var_12.format == '^[a-z]+$'
    assert var_12.coerce_types is True
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    var_13 = module_0.to_json_schema(var_12)
    var_14 = module_1.Integer()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum is None
    assert var_14.maximum is None
    assert var_14.exclusive_minimum is None
    assert var_14.exclusive_maximum is None
    assert var_14.multiple_of is None
    assert var_14.precision is None
    assert var_14.coerce_types is True
    var_15 = module_0.to_json_schema(var_14)
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
    var_17 = module_0.to_json_schema(var_16)
    var_18 = 0
    var_19 = 100
    var_20 = module_1.Integer(minimum=var_18, maximum=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Integer'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.minimum == 0
    assert var_20.maximum == 100
    assert var_20.exclusive_minimum is None
    assert var_20.exclusive_maximum is None
    assert var_20.multiple_of is None
    assert var_20.precision is None
    assert var_20.coerce_types is True
    var_21 = module_0.to_json_schema(var_20)
    var_22 = module_0.to_json_schema(var_20)
    var_23 = module_1.Integer(multiple_of=var_6)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Integer'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.minimum is None
    assert var_23.maximum is None
    assert var_23.exclusive_minimum is None
    assert var_23.exclusive_maximum is None
    assert var_23.multiple_of == 5
    assert var_23.precision is None
    assert var_23.coerce_types is True
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_1.Float()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Float'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.minimum is None
    assert var_25.maximum is None
    assert var_25.exclusive_minimum is None
    assert var_25.exclusive_maximum is None
    assert var_25.multiple_of is None
    assert var_25.precision is None
    assert var_25.coerce_types is True
    var_26 = module_0.to_json_schema(var_25)
    var_27 = module_1.Boolean()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_28 = module_0.to_json_schema(var_27)
    var_29 = module_1.Array()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Array'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert var_29.items is None
    assert var_29.additional_items is False
    assert var_29.min_items is None
    assert var_29.max_items is None
    assert var_29.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
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
    var_32 = module_1.Array(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Array'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.items).__module__}.{type(var_32.items).__qualname__}' == 'typesystem.fields.String'
    assert var_32.additional_items is False
    assert var_32.min_items is None
    assert var_32.max_items is None
    assert var_32.unique_items is False
    var_33 = module_0.to_json_schema(var_32)
    var_34 = module_1.Array(min_items=var_4, max_items=var_7)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Array'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.items is None
    assert var_34.additional_items is False
    assert var_34.min_items is True
    assert var_34.max_items == 10
    assert var_34.unique_items is False
    var_35 = var_31.get_default_value()
    module_0.to_json_schema(var_35)

def test_case_35():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_2.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = True
    var_4 = 5
    var_5 = 10
    var_6 = 'email'
    var_7 = module_1.String(format=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length is None
    assert var_7.min_length is None
    assert var_7.format == 'email'
    assert var_7.coerce_types is True
    assert var_7.pattern is None
    assert var_7.pattern_regex is None
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_0.to_json_schema(var_7)
    var_9 = module_1.Integer()
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
    assert var_9.coerce_types is True
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_0.to_json_schema(var_9)
    var_12 = 0
    var_13 = 100
    var_14 = module_0.to_json_schema(var_9)
    var_15 = module_1.Integer(exclusive_minimum=var_12, exclusive_maximum=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Integer'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum is None
    assert var_15.maximum is None
    assert var_15.exclusive_minimum == 0
    assert var_15.exclusive_maximum == 100
    assert var_15.multiple_of is None
    assert var_15.precision is None
    assert var_15.coerce_types is True
    var_16 = module_0.to_json_schema(var_15)
    var_17 = module_1.Integer(multiple_of=var_4)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Integer'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum is None
    assert var_17.maximum is None
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of == 5
    assert var_17.precision is None
    assert var_17.coerce_types is True
    var_18 = module_0.to_json_schema(var_17)
    var_19 = module_1.Float()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Float'
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
    var_20 = module_1.Float()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Float'
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
    var_21 = module_1.Boolean()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_22 = module_0.to_json_schema(var_21)
    var_23 = module_1.Boolean()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.coerce_types is True
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_1.String()
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
    var_26 = module_1.Array(min_items=var_3, max_items=var_5)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Array'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.items is None
    assert var_26.additional_items is False
    assert var_26.min_items is True
    assert var_26.max_items == 10
    assert var_26.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_27 = module_0.to_json_schema(var_26)
    var_28 = module_1.Array(unique_items=var_3)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Array'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.items is None
    assert var_28.additional_items is False
    assert var_28.min_items is None
    assert var_28.max_items is None
    assert var_28.unique_items is True
    var_29 = module_0.to_json_schema(var_28)
    var_30 = False
    var_31 = module_1.Array(additional_items=var_30)
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
    var_32 = module_0.to_json_schema(var_31)
    var_33 = module_1.String()
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
    var_34 = module_1.Array(additional_items=var_33)
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
    var_35 = module_0.to_json_schema(var_34)
    var_36 = module_0.to_json_schema(var_19)
    var_37 = 'namDe'
    var_38 = 'age'
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
    var_40 = {var_37: var_21, var_38: var_39}
    var_41 = module_1.Object(properties=var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Object'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.properties).__module__}.{type(var_41.properties).__qualname__}' == 'builtins.dict'
    assert len(var_41.properties) == 2
    assert var_41.pattern_properties == {}
    assert var_41.additional_properties is True
    assert var_41.property_names is None
    assert var_41.min_properties is None
    assert var_41.max_properties is None
    assert var_41.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_42 = module_0.to_json_schema(var_41)
    var_43 = module_1.String()
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
    var_44 = {var_37: var_43}
    var_45 = [var_37]
    var_46 = module_1.Object(properties=var_44, required=var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Object'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.properties).__module__}.{type(var_46.properties).__qualname__}' == 'builtins.dict'
    assert len(var_46.properties) == 1
    assert var_46.pattern_properties == {}
    assert var_46.additional_properties is True
    assert var_46.property_names is None
    assert var_46.min_properties is None
    assert var_46.max_properties is None
    assert var_46.required == ['namDe']
    var_47 = module_0.to_json_schema(var_46)
    var_48 = False
    var_49 = module_1.Object(additional_properties=var_48)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.fields.Object'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert var_49.properties == {}
    assert var_49.pattern_properties == {}
    assert var_49.additional_properties is False
    assert var_49.property_names is None
    assert var_49.min_properties is None
    assert var_49.max_properties is None
    assert var_49.required == []
    var_50 = module_0.to_json_schema(var_49)
    var_51 = module_1.String()
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
    var_52 = module_1.Object(additional_properties=var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Object'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert var_52.properties == {}
    assert var_52.pattern_properties == {}
    assert f'{type(var_52.additional_properties).__module__}.{type(var_52.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_52.property_names is None
    assert var_52.min_properties is None
    assert var_52.max_properties is None
    assert var_52.required == []
    var_53 = module_1.Object(property_names=var_7)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.Object'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert var_53.properties == {}
    assert var_53.pattern_properties == {}
    assert var_53.additional_properties is True
    assert f'{type(var_53.property_names).__module__}.{type(var_53.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_53.min_properties is None
    assert var_53.max_properties is None
    assert var_53.required == []
    var_54 = module_0.to_json_schema(var_53)
    var_55 = module_1.Object(min_properties=var_3, max_properties=var_4)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Object'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.properties == {}
    assert var_55.pattern_properties == {}
    assert var_55.additional_properties is True
    assert var_55.property_names is None
    assert var_55.min_properties is True
    assert var_55.max_properties == 5
    assert var_55.required == []
    var_56 = module_0.to_json_schema(var_55)
    var_57 = 'red'
    var_58 = (var_57, var_57)
    var_59 = (var_14, var_14)
    var_60 = (var_6, var_6)
    var_61 = [var_58, var_59, var_60]
    var_62 = module_1.Choice(choices=var_61)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.fields.Choice'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert var_62.choices == [('red', 'red'), ({'type': 'integer'}, {'type': 'integer'}), ('email', 'email')]
    assert var_62.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_63 = module_0.to_json_schema(var_62)
    var_64 = module_1.String()
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
    var_65 = [var_64, var_39]
    var_66 = module_1.Union(var_65)
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.fields.Union'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert f'{type(var_66.any_of).__module__}.{type(var_66.any_of).__qualname__}' == 'builtins.list'
    assert len(var_66.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_67 = module_0.to_json_schema(var_66)
    var_68 = 'anyOf'
    var_69 = var_67[var_68]
    var_70 = len(var_69)
    assert var_70 == 2
    var_71 = module_1.String()
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.String'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.allow_blank is False
    assert var_71.trim_whitespace is True
    assert var_71.max_length is None
    assert var_71.min_length is None
    assert var_71.format is None
    assert var_71.coerce_types is True
    assert var_71.pattern is None
    assert var_71.pattern_regex is None
    var_72 = module_1.Integer()
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.Integer'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert var_72.minimum is None
    assert var_72.maximum is None
    assert var_72.exclusive_minimum is None
    assert var_72.exclusive_maximum is None
    assert var_72.multiple_of is None
    assert var_72.precision is None
    assert var_72.coerce_types is True
    var_73 = module_2.OneOf(var_65)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert f'{type(var_73.one_of).__module__}.{type(var_73.one_of).__qualname__}' == 'builtins.list'
    assert len(var_73.one_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_74 = module_0.to_json_schema(var_73)
    var_75 = 'oneOf'
    var_76 = var_74[var_75]
    var_77 = len(var_76)
    assert var_77 == 2
    var_78 = module_1.String()
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
    var_79 = module_2.Not(var_78)
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.composites.Not'
    assert var_79.title == ''
    assert var_79.description == ''
    assert var_79.allow_null is False
    assert var_79.read_only is False
    assert f'{type(var_79.negated).__module__}.{type(var_79.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_80 = module_0.to_json_schema(var_79)
    var_81 = module_1.String()
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
    var_82 = module_1.Integer()
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.Integer'
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
    assert var_82.coerce_types is True
    var_83 = module_1.Boolean()
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert var_83.coerce_types is True
    var_84 = module_2.IfThenElse(var_81, var_82, var_83)
    assert f'{type(var_84).__module__}.{type(var_84).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_84.title == ''
    assert var_84.description == ''
    assert var_84.allow_null is False
    assert var_84.read_only is False
    assert f'{type(var_84.if_clause).__module__}.{type(var_84.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_84.then_clause).__module__}.{type(var_84.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_84.else_clause).__module__}.{type(var_84.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_85 = module_0.to_json_schema(var_84)
    var_86 = module_1.String()
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
    assert var_86.pattern is None
    assert var_86.pattern_regex is None
    var_87 = module_1.Integer()
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.fields.Integer'
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
    assert var_87.coerce_types is True
    var_88 = module_2.IfThenElse(var_86, var_87)
    assert f'{type(var_88).__module__}.{type(var_88).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_88.title == ''
    assert var_88.description == ''
    assert var_88.allow_null is False
    assert var_88.read_only is False
    assert f'{type(var_88.if_clause).__module__}.{type(var_88.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_88.then_clause).__module__}.{type(var_88.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_88.else_clause).__module__}.{type(var_88.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_89 = module_0.to_json_schema(var_88)
    var_90 = module_1.String()
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
    var_91 = module_1.Integer()
    assert f'{type(var_91).__module__}.{type(var_91).__qualname__}' == 'typesystem.fields.Integer'
    assert var_91.title == ''
    assert var_91.description == ''
    assert var_91.allow_null is False
    assert var_91.read_only is False
    assert var_91.minimum is None
    assert var_91.maximum is None
    assert var_91.exclusive_minimum is None
    assert var_91.exclusive_maximum is None
    assert var_91.multiple_of is None
    assert var_91.precision is None
    assert var_91.coerce_types is True
    var_92 = {var_37: var_33}
    var_93 = module_1.Object(properties=var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.Object'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert f'{type(var_93.properties).__module__}.{type(var_93.properties).__qualname__}' == 'builtins.dict'
    assert len(var_93.properties) == 1
    assert var_93.pattern_properties == {}
    assert var_93.additional_properties is True
    assert var_93.property_names is None
    assert var_93.min_properties is None
    assert var_93.max_properties is None
    assert var_93.required == []
    var_94 = module_1.String()
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
    var_95 = module_1.Integer()
    assert f'{type(var_95).__module__}.{type(var_95).__qualname__}' == 'typesystem.fields.Integer'
    assert var_95.title == ''
    assert var_95.description == ''
    assert var_95.allow_null is False
    assert var_95.read_only is False
    assert var_95.minimum is None
    assert var_95.maximum is None
    assert var_95.exclusive_minimum is None
    assert var_95.exclusive_maximum is None
    assert var_95.multiple_of is None
    assert var_95.precision is None
    assert var_95.coerce_types is True
    var_96 = {var_37: var_94, var_38: var_95}
    var_97 = module_3.Schema(var_96)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_97.title == ''
    assert var_97.description == ''
    assert var_97.allow_null is False
    assert var_97.read_only is False
    assert f'{type(var_97.fields).__module__}.{type(var_97.fields).__qualname__}' == 'builtins.dict'
    assert len(var_97.fields) == 2
    assert var_97.required == ['namDe', 'age']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_98 = module_0.to_json_schema(var_97)

def test_case_36():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_2.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = 5
    var_7 = 10
    var_8 = module_0.to_json_schema(var_5)
    var_9 = '^[a-z]+$'
    var_10 = module_1.String(pattern=var_9)
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
    assert var_10.pattern == '^[a-z]+$'
    assert f'{type(var_10.pattern_regex).__module__}.{type(var_10.pattern_regex).__qualname__}' == 're.Pattern'
    var_11 = module_0.to_json_schema(var_10)
    var_12 = module_1.String(format=var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.allow_blank is False
    assert var_12.trim_whitespace is True
    assert var_12.max_length is None
    assert var_12.min_length is None
    assert var_12.format == '^[a-z]+$'
    assert var_12.coerce_types is True
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    var_13 = module_0.to_json_schema(var_12)
    var_14 = module_1.Integer()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum is None
    assert var_14.maximum is None
    assert var_14.exclusive_minimum is None
    assert var_14.exclusive_maximum is None
    assert var_14.multiple_of is None
    assert var_14.precision is None
    assert var_14.coerce_types is True
    var_15 = module_0.to_json_schema(var_14)
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
    var_17 = module_0.to_json_schema(var_16)
    var_18 = 0
    var_19 = 100
    var_20 = module_1.Integer(minimum=var_18, maximum=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Integer'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.minimum == 0
    assert var_20.maximum == 100
    assert var_20.exclusive_minimum is None
    assert var_20.exclusive_maximum is None
    assert var_20.multiple_of is None
    assert var_20.precision is None
    assert var_20.coerce_types is True
    var_21 = module_0.to_json_schema(var_20)
    var_22 = module_1.Integer(exclusive_minimum=var_18, exclusive_maximum=var_19)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Integer'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.minimum is None
    assert var_22.maximum is None
    assert var_22.exclusive_minimum == 0
    assert var_22.exclusive_maximum == 100
    assert var_22.multiple_of is None
    assert var_22.precision is None
    assert var_22.coerce_types is True
    var_23 = module_0.to_json_schema(var_22)
    var_24 = module_1.Integer(multiple_of=var_6)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Integer'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.minimum is None
    assert var_24.maximum is None
    assert var_24.exclusive_minimum is None
    assert var_24.exclusive_maximum is None
    assert var_24.multiple_of == 5
    assert var_24.precision is None
    assert var_24.coerce_types is True
    var_25 = module_0.to_json_schema(var_24)
    var_26 = module_1.Float()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Float'
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
    var_27 = module_0.to_json_schema(var_26)
    var_28 = module_1.Boolean()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = module_1.Array(var_12)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Array'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.items).__module__}.{type(var_30.items).__qualname__}' == 'typesystem.fields.String'
    assert var_30.additional_items is False
    assert var_30.min_items is None
    assert var_30.max_items is None
    assert var_30.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_31 = module_0.to_json_schema(var_30)
    var_32 = module_1.Array(min_items=var_4, max_items=var_7)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Array'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert var_32.items is None
    assert var_32.additional_items is False
    assert var_32.min_items is True
    assert var_32.max_items == 10
    assert var_32.unique_items is False
    var_33 = var_12.get_default_value()

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
    var_1 = module_2.NeverMatch()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
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
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = 'maxLength'
    var_5 = True
    var_6 = module_1.String()
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
    var_7 = module_0.to_json_schema(var_6)
    var_8 = 5
    var_9 = 10
    var_10 = module_1.String(max_length=var_9, min_length=var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length == 10
    assert var_10.min_length == 5
    assert var_10.format is None
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = module_0.to_json_schema(var_10)
    var_12 = module_1.String(pattern=var_4)
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
    assert var_12.pattern == 'maxLength'
    assert f'{type(var_12.pattern_regex).__module__}.{type(var_12.pattern_regex).__qualname__}' == 're.Pattern'
    var_13 = module_0.to_json_schema(var_12)
    var_14 = 'email'
    var_15 = module_1.String(format=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.String'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.allow_blank is False
    assert var_15.trim_whitespace is True
    assert var_15.max_length is None
    assert var_15.min_length is None
    assert var_15.format == 'email'
    assert var_15.coerce_types is True
    assert var_15.pattern is None
    assert var_15.pattern_regex is None
    var_16 = module_0.to_json_schema(var_15)
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
    var_18 = module_0.to_json_schema(var_17)

@pytest.mark.xfail(strict=True)
def test_case_38():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_2.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
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
    var_5 = True
    var_6 = module_1.String()
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
    var_7 = module_0.to_json_schema(var_6)
    var_8 = 5
    var_9 = 10
    var_10 = module_1.String(max_length=var_9, min_length=var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length == 10
    assert var_10.min_length == 5
    assert var_10.format is None
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = module_0.to_json_schema(var_10)
    var_12 = '^[a-z]+$'
    var_13 = module_1.String(pattern=var_12)
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
    assert var_13.pattern == '^[a-z]+$'
    assert f'{type(var_13.pattern_regex).__module__}.{type(var_13.pattern_regex).__qualname__}' == 're.Pattern'
    var_14 = 'email'
    var_15 = module_1.String(format=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.String'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.allow_blank is False
    assert var_15.trim_whitespace is True
    assert var_15.max_length is None
    assert var_15.min_length is None
    assert var_15.format == 'email'
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
    var_17 = module_0.to_json_schema(var_16)
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
    var_19 = 0
    var_20 = 100
    var_21 = module_1.Integer(minimum=var_19, maximum=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Integer'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.minimum == 0
    assert var_21.maximum == 100
    assert var_21.exclusive_minimum is None
    assert var_21.exclusive_maximum is None
    assert var_21.multiple_of is None
    assert var_21.precision is None
    assert var_21.coerce_types is True
    var_22 = module_0.to_json_schema(var_21)
    var_23 = var_16.get_default_value()
    var_24 = module_1.Integer(multiple_of=var_8)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Integer'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.minimum is None
    assert var_24.maximum is None
    assert var_24.exclusive_minimum is None
    assert var_24.exclusive_maximum is None
    assert var_24.multiple_of == 5
    assert var_24.precision is None
    assert var_24.coerce_types is True
    var_25 = module_0.to_json_schema(var_24)
    var_26 = module_1.Float()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Float'
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
    var_27 = module_0.to_json_schema(var_26)
    var_28 = module_1.Boolean()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = module_1.Boolean()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.coerce_types is True
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
    var_33 = module_1.Array(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Array'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.items).__module__}.{type(var_33.items).__qualname__}' == 'typesystem.fields.String'
    assert var_33.additional_items is False
    assert var_33.min_items is None
    assert var_33.max_items is None
    assert var_33.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_34 = module_0.to_json_schema(var_33)
    var_35 = module_0.to_json_schema(var_33)
    var_36 = module_1.Array(unique_items=var_5)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Array'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.items is None
    assert var_36.additional_items is False
    assert var_36.min_items is None
    assert var_36.max_items is None
    assert var_36.unique_items is True
    var_37 = module_0.to_json_schema(var_36)
    var_38 = False
    var_39 = module_1.Array(additional_items=var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Array'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.items is None
    assert var_39.additional_items is False
    assert var_39.min_items is None
    assert var_39.max_items is None
    assert var_39.unique_items is False
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
    var_42 = module_1.Array(additional_items=var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.Array'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert var_42.items is None
    assert f'{type(var_42.additional_items).__module__}.{type(var_42.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_42.min_items is None
    assert var_42.max_items is None
    assert var_42.unique_items is False
    var_43 = module_0.to_json_schema(var_42)
    var_44 = module_1.Object()
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Object'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert var_44.properties == {}
    assert var_44.pattern_properties == {}
    assert var_44.additional_properties is True
    assert var_44.property_names is None
    assert var_44.min_properties is None
    assert var_44.max_properties is None
    assert var_44.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_45 = module_0.to_json_schema(var_44)
    var_46 = 'namDe'
    var_47 = 'age'
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
    var_50 = {var_46: var_48, var_47: var_49}
    var_51 = module_1.Object(properties=var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Object'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.properties).__module__}.{type(var_51.properties).__qualname__}' == 'builtins.dict'
    assert len(var_51.properties) == 2
    assert var_51.pattern_properties == {}
    assert var_51.additional_properties is True
    assert var_51.property_names is None
    assert var_51.min_properties is None
    assert var_51.max_properties is None
    assert var_51.required == []
    var_52 = module_1.String()
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
    var_53 = {var_46: var_52}
    var_54 = [var_46]
    var_55 = module_1.Object(properties=var_53, required=var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Object'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert f'{type(var_55.properties).__module__}.{type(var_55.properties).__qualname__}' == 'builtins.dict'
    assert len(var_55.properties) == 1
    assert var_55.pattern_properties == {}
    assert var_55.additional_properties is True
    assert var_55.property_names is None
    assert var_55.min_properties is None
    assert var_55.max_properties is None
    assert var_55.required == ['namDe']
    var_56 = module_0.to_json_schema(var_55)
    var_57 = module_1.Object(additional_properties=var_1)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.Object'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert var_57.properties == {}
    assert var_57.pattern_properties == {}
    assert var_57.additional_properties is True
    assert var_57.property_names is None
    assert var_57.min_properties is None
    assert var_57.max_properties is None
    assert var_57.required == []
    var_58 = module_0.to_json_schema(var_57)
    var_59 = module_1.String()
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.fields.String'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert var_59.allow_blank is False
    assert var_59.trim_whitespace is True
    assert var_59.max_length is None
    assert var_59.min_length is None
    assert var_59.format is None
    assert var_59.coerce_types is True
    assert var_59.pattern is None
    assert var_59.pattern_regex is None
    var_60 = module_1.Object(additional_properties=var_59)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.Object'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.properties == {}
    assert var_60.pattern_properties == {}
    assert f'{type(var_60.additional_properties).__module__}.{type(var_60.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_60.property_names is None
    assert var_60.min_properties is None
    assert var_60.max_properties is None
    assert var_60.required == []
    var_61 = module_0.to_json_schema(var_60)
    var_62 = module_1.String(pattern=var_12)
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
    assert var_62.pattern == '^[a-z]+$'
    assert f'{type(var_62.pattern_regex).__module__}.{type(var_62.pattern_regex).__qualname__}' == 're.Pattern'
    var_63 = module_1.Object(property_names=var_62)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.fields.Object'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert var_63.properties == {}
    assert var_63.pattern_properties == {}
    assert var_63.additional_properties is True
    assert f'{type(var_63.property_names).__module__}.{type(var_63.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_63.min_properties is None
    assert var_63.max_properties is None
    assert var_63.required == []
    var_64 = module_0.to_json_schema(var_63)
    var_65 = module_1.Object(min_properties=var_5, max_properties=var_8)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Object'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.properties == {}
    assert var_65.pattern_properties == {}
    assert var_65.additional_properties is True
    assert var_65.property_names is None
    assert var_65.min_properties is True
    assert var_65.max_properties == 5
    assert var_65.required == []
    var_66 = 'constalnt_value'
    var_67 = module_1.Const(var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.fields.Const'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    assert var_67.const == 'constalnt_value'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_68 = module_1.String()
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
    var_69 = module_1.Integer()
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
    var_70 = module_1.String()
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.fields.String'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert var_70.allow_blank is False
    assert var_70.trim_whitespace is True
    assert var_70.max_length is None
    assert var_70.min_length is None
    assert var_70.format is None
    assert var_70.coerce_types is True
    assert var_70.pattern is None
    assert var_70.pattern_regex is None
    var_71 = module_1.Integer()
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Integer'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.minimum is None
    assert var_71.maximum is None
    assert var_71.exclusive_minimum is None
    assert var_71.exclusive_maximum is None
    assert var_71.multiple_of is None
    assert var_71.precision is None
    assert var_71.coerce_types is True
    var_72 = [var_70, var_71]
    var_73 = module_2.OneOf(var_72)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert f'{type(var_73.one_of).__module__}.{type(var_73.one_of).__qualname__}' == 'builtins.list'
    assert len(var_73.one_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_74 = module_0.to_json_schema(var_73)
    var_75 = 'oneOf'
    var_76 = var_74[var_75]
    var_77 = len(var_76)
    assert var_77 == 2
    var_78 = [var_68, var_32]
    var_79 = module_2.AllOf(var_78)
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_79.title == ''
    assert var_79.description == ''
    assert var_79.allow_null is False
    assert var_79.read_only is False
    assert f'{type(var_79.all_of).__module__}.{type(var_79.all_of).__qualname__}' == 'builtins.list'
    assert len(var_79.all_of) == 2
    var_80 = module_0.to_json_schema(var_79)
    var_81 = 'allOf'
    var_82 = var_80[var_81]
    var_83 = len(var_82)
    assert var_83 == 2
    var_84 = module_1.String()
    assert f'{type(var_84).__module__}.{type(var_84).__qualname__}' == 'typesystem.fields.String'
    assert var_84.title == ''
    assert var_84.description == ''
    assert var_84.allow_null is False
    assert var_84.read_only is False
    assert var_84.allow_blank is False
    assert var_84.trim_whitespace is True
    assert var_84.max_length is None
    assert var_84.min_length is None
    assert var_84.format is None
    assert var_84.coerce_types is True
    assert var_84.pattern is None
    assert var_84.pattern_regex is None
    var_85 = module_2.Not(var_84)
    assert f'{type(var_85).__module__}.{type(var_85).__qualname__}' == 'typesystem.composites.Not'
    assert var_85.title == ''
    assert var_85.description == ''
    assert var_85.allow_null is False
    assert var_85.read_only is False
    assert f'{type(var_85.negated).__module__}.{type(var_85.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_86 = module_0.to_json_schema(var_85)
    var_87 = module_1.Integer()
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.fields.Integer'
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
    assert var_87.coerce_types is True
    var_88 = module_1.Boolean()
    assert f'{type(var_88).__module__}.{type(var_88).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_88.title == ''
    assert var_88.description == ''
    assert var_88.allow_null is False
    assert var_88.read_only is False
    assert var_88.coerce_types is True
    var_89 = module_2.IfThenElse(var_12, var_87, var_88)
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_89.title == ''
    assert var_89.description == ''
    assert var_89.allow_null is False
    assert var_89.read_only is False
    assert var_89.if_clause == '^[a-z]+$'
    assert f'{type(var_89.then_clause).__module__}.{type(var_89.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_89.else_clause).__module__}.{type(var_89.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    module_0.to_json_schema(var_89)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = module_1.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_1 = module_4.purge()
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
    var_2 = module_4.purge()
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
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = True
    var_6 = 5
    var_7 = 10
    var_8 = module_0.to_json_schema(var_3)
    var_9 = '^[a-z]+$'
    var_10 = module_1.String(format=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.String'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.allow_blank is False
    assert var_10.trim_whitespace is True
    assert var_10.max_length is None
    assert var_10.min_length is None
    assert var_10.format == '^[a-z]+$'
    assert var_10.coerce_types is True
    assert var_10.pattern is None
    assert var_10.pattern_regex is None
    var_11 = module_1.Integer()
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
    var_12 = module_0.get_standard_properties(var_10)
    var_13 = module_1.Integer()
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
    assert var_13.coerce_types is True
    var_14 = 0
    var_15 = 100
    var_16 = module_1.Integer(minimum=var_14, maximum=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Integer'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum == 0
    assert var_16.maximum == 100
    assert var_16.exclusive_minimum is None
    assert var_16.exclusive_maximum is None
    assert var_16.multiple_of is None
    assert var_16.precision is None
    assert var_16.coerce_types is True
    var_17 = module_0.to_json_schema(var_13)
    var_18 = None
    var_19 = module_1.Integer(minimum=var_15, exclusive_minimum=var_15, multiple_of=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Integer'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.minimum == 100
    assert var_19.maximum is None
    assert var_19.exclusive_minimum == 100
    assert var_19.exclusive_maximum is None
    assert var_19.multiple_of is None
    assert var_19.precision is None
    assert var_19.coerce_types is True
    var_20 = module_5._EnumDict()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'enum._EnumDict'
    assert len(var_20) == 0
    var_21 = module_1.Float()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Float'
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
    var_22 = module_1.Boolean()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_23 = module_1.Boolean()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.coerce_types is True
    var_24 = module_4.purge()
    var_25 = module_1.Array(unique_items=var_5)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Array'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.items is None
    assert var_25.additional_items is False
    assert var_25.min_items is None
    assert var_25.max_items is None
    assert var_25.unique_items is True
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_26 = module_0.to_json_schema(var_25)
    var_27 = False
    var_28 = module_1.Array(additional_items=var_27)
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
    var_29 = module_0.to_json_schema(var_28)
    var_30 = module_1.String()
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
    var_31 = module_1.Array(additional_items=var_30)
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
    var_32 = module_0.to_json_schema(var_31)
    var_33 = module_1.Object()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Object'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.properties == {}
    assert var_33.pattern_properties == {}
    assert var_33.additional_properties is True
    assert var_33.property_names is None
    assert var_33.min_properties is None
    assert var_33.max_properties is None
    assert var_33.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_34 = module_0.to_json_schema(var_33)
    var_35 = 'namDe'
    var_36 = 'age'
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
    var_38 = {var_35: var_3, var_36: var_37}
    var_39 = module_1.Object(properties=var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Object'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.properties).__module__}.{type(var_39.properties).__qualname__}' == 'builtins.dict'
    assert len(var_39.properties) == 2
    assert var_39.pattern_properties == {}
    assert var_39.additional_properties is True
    assert var_39.property_names is None
    assert var_39.min_properties is None
    assert var_39.max_properties is None
    assert var_39.required == []
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
    var_42 = [var_35]
    var_43 = module_1.Object(properties=var_20, required=var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Object'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.properties == {}
    assert var_43.pattern_properties == {}
    assert var_43.additional_properties is True
    assert var_43.property_names is None
    assert var_43.min_properties is None
    assert var_43.max_properties is None
    assert var_43.required == ['namDe']
    var_44 = module_0.to_json_schema(var_43)
    var_45 = False
    var_46 = module_1.Object(additional_properties=var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Object'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert var_46.properties == {}
    assert var_46.pattern_properties == {}
    assert var_46.additional_properties is False
    assert var_46.property_names is None
    assert var_46.min_properties is None
    assert var_46.max_properties is None
    assert var_46.required == []
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
    var_49 = module_1.Object(additional_properties=var_48)
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
    var_50 = module_0.to_json_schema(var_49)
    var_51 = module_1.String(pattern=var_9)
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
    var_52 = module_1.Object(property_names=var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.fields.Object'
    assert var_52.title == ''
    assert var_52.description == ''
    assert var_52.allow_null is False
    assert var_52.read_only is False
    assert var_52.properties == {}
    assert var_52.pattern_properties == {}
    assert var_52.additional_properties is True
    assert f'{type(var_52.property_names).__module__}.{type(var_52.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_52.min_properties is None
    assert var_52.max_properties is None
    assert var_52.required == []
    var_53 = module_0.to_json_schema(var_52)
    var_54 = module_1.Object(min_properties=var_5, max_properties=var_6)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Object'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert var_54.properties == {}
    assert var_54.pattern_properties == {}
    assert var_54.additional_properties is True
    assert var_54.property_names is None
    assert var_54.min_properties is True
    assert var_54.max_properties == 5
    assert var_54.required == []
    var_55 = 'constant_value'
    var_56 = module_1.Const(var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Const'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert var_56.const == 'constant_value'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_57 = module_1.String()
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
    var_58 = module_1.Integer()
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
    var_59 = var_31.serialize(var_5)
    assert var_59 is True
    var_60 = module_1.Integer()
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.Integer'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.minimum is None
    assert var_60.maximum is None
    assert var_60.exclusive_minimum is None
    assert var_60.exclusive_maximum is None
    assert var_60.multiple_of is None
    assert var_60.precision is None
    assert var_60.coerce_types is True
    var_61 = module_1.String(max_length=var_7)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.fields.String'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert var_61.allow_blank is False
    assert var_61.trim_whitespace is True
    assert var_61.max_length == 10
    assert var_61.min_length is None
    assert var_61.format is None
    assert var_61.coerce_types is True
    assert var_61.pattern is None
    assert var_61.pattern_regex is None
    var_62 = [var_57, var_61]
    var_63 = module_2.AllOf(var_62)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert f'{type(var_63.all_of).__module__}.{type(var_63.all_of).__qualname__}' == 'builtins.list'
    assert len(var_63.all_of) == 2
    var_64 = module_0.to_json_schema(var_63)
    var_65 = 'allOf'
    var_66 = var_64[var_65]
    var_67 = len(var_66)
    assert var_67 == 2
    var_68 = module_1.String()
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
    var_69 = module_2.Not(var_68)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.composites.Not'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert f'{type(var_69.negated).__module__}.{type(var_69.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_70 = module_0.to_json_schema(var_69)
    var_71 = module_1.String()
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.String'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.allow_blank is False
    assert var_71.trim_whitespace is True
    assert var_71.max_length is None
    assert var_71.min_length is None
    assert var_71.format is None
    assert var_71.coerce_types is True
    assert var_71.pattern is None
    assert var_71.pattern_regex is None
    var_72 = module_1.Integer()
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.Integer'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert var_72.minimum is None
    assert var_72.maximum is None
    assert var_72.exclusive_minimum is None
    assert var_72.exclusive_maximum is None
    assert var_72.multiple_of is None
    assert var_72.precision is None
    assert var_72.coerce_types is True
    var_73 = module_1.Boolean()
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert var_73.coerce_types is True
    var_74 = module_2.IfThenElse(var_71, var_72, var_73)
    assert f'{type(var_74).__module__}.{type(var_74).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_74.title == ''
    assert var_74.description == ''
    assert var_74.allow_null is False
    assert var_74.read_only is False
    assert f'{type(var_74.if_clause).__module__}.{type(var_74.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_74.then_clause).__module__}.{type(var_74.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_74.else_clause).__module__}.{type(var_74.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_75 = module_0.to_json_schema(var_74)
    var_76 = module_2.IfThenElse(var_59, var_67)
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_76.title == ''
    assert var_76.description == ''
    assert var_76.allow_null is False
    assert var_76.read_only is False
    assert var_76.if_clause is True
    assert var_76.then_clause == 2
    assert f'{type(var_76.else_clause).__module__}.{type(var_76.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_4.findall(var_66, var_67)

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
    var_1 = module_2.NeverMatch()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
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
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = 'maxLength'
    var_5 = True
    var_6 = module_1.String()
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
    var_7 = module_0.to_json_schema(var_6)
    var_8 = module_0.to_json_schema(var_6)
    var_9 = module_1.String(pattern=var_4)
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
    assert var_9.pattern == 'maxLength'
    assert f'{type(var_9.pattern_regex).__module__}.{type(var_9.pattern_regex).__qualname__}' == 're.Pattern'
    var_10 = module_0.to_json_schema(var_9)
    var_11 = 'email'
    var_12 = module_1.String(format=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.allow_blank is False
    assert var_12.trim_whitespace is True
    assert var_12.max_length is None
    assert var_12.min_length is None
    assert var_12.format == 'email'
    assert var_12.coerce_types is True
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    var_13 = module_0.to_json_schema(var_12)
    var_14 = module_1.Integer()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum is None
    assert var_14.maximum is None
    assert var_14.exclusive_minimum is None
    assert var_14.exclusive_maximum is None
    assert var_14.multiple_of is None
    assert var_14.precision is None
    assert var_14.coerce_types is True
    var_15 = module_0.to_json_schema(var_14)

@pytest.mark.xfail(strict=True)
def test_case_41():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_2.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
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
    var_6 = True
    var_7 = module_1.String()
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
    var_8 = module_0.to_json_schema(var_7)
    var_9 = 5
    var_10 = 10
    var_11 = module_1.String(max_length=var_10, min_length=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.String'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.allow_blank is False
    assert var_11.trim_whitespace is True
    assert var_11.max_length == 10
    assert var_11.min_length == 5
    assert var_11.format is None
    assert var_11.coerce_types is True
    assert var_11.pattern is None
    assert var_11.pattern_regex is None
    var_12 = module_0.to_json_schema(var_11)
    var_13 = '^[a-z]+$'
    var_14 = module_1.String(pattern=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.String'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.allow_blank is False
    assert var_14.trim_whitespace is True
    assert var_14.max_length is None
    assert var_14.min_length is None
    assert var_14.format is None
    assert var_14.coerce_types is True
    assert var_14.pattern == '^[a-z]+$'
    assert f'{type(var_14.pattern_regex).__module__}.{type(var_14.pattern_regex).__qualname__}' == 're.Pattern'
    var_15 = module_0.to_json_schema(var_14)
    var_16 = 'email'
    var_17 = module_1.String(format=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.String'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.allow_blank is False
    assert var_17.trim_whitespace is True
    assert var_17.max_length is None
    assert var_17.min_length is None
    assert var_17.format == 'email'
    assert var_17.coerce_types is True
    assert var_17.pattern is None
    assert var_17.pattern_regex is None
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
    var_19 = module_0.to_json_schema(var_18)
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
    var_21 = 0
    var_22 = 100
    var_23 = module_1.Integer(minimum=var_21, maximum=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Integer'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.minimum == 0
    assert var_23.maximum == 100
    assert var_23.exclusive_minimum is None
    assert var_23.exclusive_maximum is None
    assert var_23.multiple_of is None
    assert var_23.precision is None
    assert var_23.coerce_types is True
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_1.Integer(exclusive_minimum=var_21, exclusive_maximum=var_22)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Integer'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.minimum is None
    assert var_25.maximum is None
    assert var_25.exclusive_minimum == 0
    assert var_25.exclusive_maximum == 100
    assert var_25.multiple_of is None
    assert var_25.precision is None
    assert var_25.coerce_types is True
    var_26 = module_0.to_json_schema(var_25)
    var_27 = module_1.Integer(multiple_of=var_9)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Integer'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.minimum is None
    assert var_27.maximum is None
    assert var_27.exclusive_minimum is None
    assert var_27.exclusive_maximum is None
    assert var_27.multiple_of == 5
    assert var_27.precision is None
    assert var_27.coerce_types is True
    var_28 = module_0.to_json_schema(var_27)
    var_29 = module_1.Float()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.fields.Float'
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
    var_30 = module_0.to_json_schema(var_29)
    var_31 = module_1.Float()
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
    assert var_31.coerce_types is True
    var_32 = module_0.to_json_schema(var_31)
    var_33 = module_1.Boolean()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_34 = module_0.to_json_schema(var_33)
    var_35 = module_1.Boolean()
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.coerce_types is True
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
    var_38 = module_1.Array(var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Array'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert f'{type(var_38.items).__module__}.{type(var_38.items).__qualname__}' == 'typesystem.fields.String'
    assert var_38.additional_items is False
    assert var_38.min_items is None
    assert var_38.max_items is None
    assert var_38.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_39 = module_0.to_json_schema(var_38)
    var_40 = module_1.Array(min_items=var_6, max_items=var_10)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Array'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert var_40.items is None
    assert var_40.additional_items is False
    assert var_40.min_items is True
    assert var_40.max_items == 10
    assert var_40.unique_items is False
    var_41 = module_0.to_json_schema(var_40)
    var_42 = module_1.Array(unique_items=var_6)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.Array'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert var_42.items is None
    assert var_42.additional_items is False
    assert var_42.min_items is None
    assert var_42.max_items is None
    assert var_42.unique_items is True
    var_43 = module_0.to_json_schema(var_42)
    var_44 = False
    var_45 = module_1.Array(additional_items=var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Array'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.items is None
    assert var_45.additional_items is False
    assert var_45.min_items is None
    assert var_45.max_items is None
    assert var_45.unique_items is False
    var_46 = module_0.to_json_schema(var_45)
    var_47 = module_1.String()
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
    var_48 = module_1.Array(additional_items=var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.Array'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert var_48.items is None
    assert f'{type(var_48.additional_items).__module__}.{type(var_48.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_48.min_items is None
    assert var_48.max_items is None
    assert var_48.unique_items is False
    var_49 = module_0.to_json_schema(var_48)
    var_50 = module_1.Object()
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.fields.Object'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert var_50.properties == {}
    assert var_50.pattern_properties == {}
    assert var_50.additional_properties is True
    assert var_50.property_names is None
    assert var_50.min_properties is None
    assert var_50.max_properties is None
    assert var_50.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_51 = module_0.to_json_schema(var_50)
    var_52 = 'namDe'
    var_53 = 'age'
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
    var_55 = module_1.Integer()
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Integer'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.minimum is None
    assert var_55.maximum is None
    assert var_55.exclusive_minimum is None
    assert var_55.exclusive_maximum is None
    assert var_55.multiple_of is None
    assert var_55.precision is None
    assert var_55.coerce_types is True
    var_56 = {var_52: var_54, var_53: var_55}
    var_57 = module_1.Object(properties=var_56)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.Object'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert f'{type(var_57.properties).__module__}.{type(var_57.properties).__qualname__}' == 'builtins.dict'
    assert len(var_57.properties) == 2
    assert var_57.pattern_properties == {}
    assert var_57.additional_properties is True
    assert var_57.property_names is None
    assert var_57.min_properties is None
    assert var_57.max_properties is None
    assert var_57.required == []
    var_58 = module_0.to_json_schema(var_57)
    var_59 = module_1.String()
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.fields.String'
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert var_59.allow_blank is False
    assert var_59.trim_whitespace is True
    assert var_59.max_length is None
    assert var_59.min_length is None
    assert var_59.format is None
    assert var_59.coerce_types is True
    assert var_59.pattern is None
    assert var_59.pattern_regex is None
    var_60 = {var_52: var_59}
    var_61 = [var_52]
    var_62 = module_1.Object(properties=var_60, required=var_61)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.fields.Object'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert f'{type(var_62.properties).__module__}.{type(var_62.properties).__qualname__}' == 'builtins.dict'
    assert len(var_62.properties) == 1
    assert var_62.pattern_properties == {}
    assert var_62.additional_properties is True
    assert var_62.property_names is None
    assert var_62.min_properties is None
    assert var_62.max_properties is None
    assert var_62.required == ['namDe']
    var_63 = module_0.to_json_schema(var_62)
    var_64 = False
    var_65 = module_1.Object(additional_properties=var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Object'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.properties == {}
    assert var_65.pattern_properties == {}
    assert var_65.additional_properties is False
    assert var_65.property_names is None
    assert var_65.min_properties is None
    assert var_65.max_properties is None
    assert var_65.required == []
    var_66 = module_0.to_json_schema(var_65)
    var_67 = module_4.purge()
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
    var_68 = module_1.Object(additional_properties=var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.fields.Object'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert var_68.properties == {}
    assert var_68.pattern_properties == {}
    assert var_68.additional_properties is None
    assert var_68.property_names is None
    assert var_68.min_properties is None
    assert var_68.max_properties is None
    assert var_68.required == []
    var_69 = module_0.to_json_schema(var_68)
    var_70 = module_1.String(pattern=var_13)
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.fields.String'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert var_70.allow_blank is False
    assert var_70.trim_whitespace is True
    assert var_70.max_length is None
    assert var_70.min_length is None
    assert var_70.format is None
    assert var_70.coerce_types is True
    assert var_70.pattern == '^[a-z]+$'
    assert f'{type(var_70.pattern_regex).__module__}.{type(var_70.pattern_regex).__qualname__}' == 're.Pattern'
    var_71 = module_1.Object(property_names=var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Object'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.properties == {}
    assert var_71.pattern_properties == {}
    assert var_71.additional_properties is True
    assert f'{type(var_71.property_names).__module__}.{type(var_71.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_71.min_properties is None
    assert var_71.max_properties is None
    assert var_71.required == []
    var_72 = module_0.to_json_schema(var_71)
    var_73 = module_1.Object(min_properties=var_6, max_properties=var_9)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.Object'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert var_73.properties == {}
    assert var_73.pattern_properties == {}
    assert var_73.additional_properties is True
    assert var_73.property_names is None
    assert var_73.min_properties is True
    assert var_73.max_properties == 5
    assert var_73.required == []
    var_74 = 'red'
    var_75 = (var_74, var_74)
    var_76 = 'green'
    var_77 = (var_76, var_76)
    var_78 = 'blue'
    var_79 = (var_78, var_78)
    var_80 = [var_75, var_77, var_79]
    var_81 = module_1.Choice(choices=var_80)
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'typesystem.fields.Choice'
    assert var_81.title == ''
    assert var_81.description == ''
    assert var_81.allow_null is False
    assert var_81.read_only is False
    assert var_81.choices == [('red', 'red'), ('green', 'green'), ('blue', 'blue')]
    assert var_81.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_82 = module_0.to_json_schema(var_81)
    var_83 = 'constant_value'
    var_84 = module_1.Const(var_83)
    assert f'{type(var_84).__module__}.{type(var_84).__qualname__}' == 'typesystem.fields.Const'
    assert var_84.title == ''
    assert var_84.description == ''
    assert var_84.allow_null is False
    assert var_84.read_only is False
    assert var_84.const == 'constant_value'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_85 = module_1.String()
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
    var_86 = module_1.Integer()
    assert f'{type(var_86).__module__}.{type(var_86).__qualname__}' == 'typesystem.fields.Integer'
    assert var_86.title == ''
    assert var_86.description == ''
    assert var_86.allow_null is False
    assert var_86.read_only is False
    assert var_86.minimum is None
    assert var_86.maximum is None
    assert var_86.exclusive_minimum is None
    assert var_86.exclusive_maximum is None
    assert var_86.multiple_of is None
    assert var_86.precision is None
    assert var_86.coerce_types is True
    var_87 = [var_85, var_86]
    var_88 = module_1.Union(var_87)
    assert f'{type(var_88).__module__}.{type(var_88).__qualname__}' == 'typesystem.fields.Union'
    assert var_88.title == ''
    assert var_88.description == ''
    assert var_88.allow_null is False
    assert var_88.read_only is False
    assert f'{type(var_88.any_of).__module__}.{type(var_88.any_of).__qualname__}' == 'builtins.list'
    assert len(var_88.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_89 = module_0.to_json_schema(var_88)
    var_90 = 'anyOf'
    var_91 = var_89[var_90]
    var_92 = len(var_91)
    assert var_92 == 2
    var_93 = module_1.String()
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
    assert var_93.pattern is None
    assert var_93.pattern_regex is None
    var_94 = module_1.Integer()
    assert f'{type(var_94).__module__}.{type(var_94).__qualname__}' == 'typesystem.fields.Integer'
    assert var_94.title == ''
    assert var_94.description == ''
    assert var_94.allow_null is False
    assert var_94.read_only is False
    assert var_94.minimum is None
    assert var_94.maximum is None
    assert var_94.exclusive_minimum is None
    assert var_94.exclusive_maximum is None
    assert var_94.multiple_of is None
    assert var_94.precision is None
    assert var_94.coerce_types is True
    var_95 = [var_93, var_94]
    var_96 = module_2.OneOf(var_95)
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_96.title == ''
    assert var_96.description == ''
    assert var_96.allow_null is False
    assert var_96.read_only is False
    assert f'{type(var_96.one_of).__module__}.{type(var_96.one_of).__qualname__}' == 'builtins.list'
    assert len(var_96.one_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_97 = module_0.to_json_schema(var_96)
    var_98 = 'oneOf'
    var_99 = var_97[var_98]
    var_100 = len(var_99)
    assert var_100 == 2
    var_101 = module_1.String(max_length=var_10)
    assert f'{type(var_101).__module__}.{type(var_101).__qualname__}' == 'typesystem.fields.String'
    assert var_101.title == ''
    assert var_101.description == ''
    assert var_101.allow_null is False
    assert var_101.read_only is False
    assert var_101.allow_blank is False
    assert var_101.trim_whitespace is True
    assert var_101.max_length == 10
    assert var_101.min_length is None
    assert var_101.format is None
    assert var_101.coerce_types is True
    assert var_101.pattern is None
    assert var_101.pattern_regex is None
    var_102 = [var_85, var_101]
    var_103 = module_2.AllOf(var_102)
    assert f'{type(var_103).__module__}.{type(var_103).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_103.title == ''
    assert var_103.description == ''
    assert var_103.allow_null is False
    assert var_103.read_only is False
    assert f'{type(var_103.all_of).__module__}.{type(var_103.all_of).__qualname__}' == 'builtins.list'
    assert len(var_103.all_of) == 2
    var_104 = module_0.to_json_schema(var_103)
    var_105 = 'allOf'
    var_106 = var_104[var_105]
    var_107 = len(var_106)
    assert var_107 == 2
    var_108 = module_1.String()
    assert f'{type(var_108).__module__}.{type(var_108).__qualname__}' == 'typesystem.fields.String'
    assert var_108.title == ''
    assert var_108.description == ''
    assert var_108.allow_null is False
    assert var_108.read_only is False
    assert var_108.allow_blank is False
    assert var_108.trim_whitespace is True
    assert var_108.max_length is None
    assert var_108.min_length is None
    assert var_108.format is None
    assert var_108.coerce_types is True
    assert var_108.pattern is None
    assert var_108.pattern_regex is None
    var_109 = module_2.Not(var_108)
    assert f'{type(var_109).__module__}.{type(var_109).__qualname__}' == 'typesystem.composites.Not'
    assert var_109.title == ''
    assert var_109.description == ''
    assert var_109.allow_null is False
    assert var_109.read_only is False
    assert f'{type(var_109.negated).__module__}.{type(var_109.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_110 = module_0.to_json_schema(var_109)
    var_111 = module_1.String()
    assert f'{type(var_111).__module__}.{type(var_111).__qualname__}' == 'typesystem.fields.String'
    assert var_111.title == ''
    assert var_111.description == ''
    assert var_111.allow_null is False
    assert var_111.read_only is False
    assert var_111.allow_blank is False
    assert var_111.trim_whitespace is True
    assert var_111.max_length is None
    assert var_111.min_length is None
    assert var_111.format is None
    assert var_111.coerce_types is True
    assert var_111.pattern is None
    assert var_111.pattern_regex is None
    var_112 = module_1.Integer()
    assert f'{type(var_112).__module__}.{type(var_112).__qualname__}' == 'typesystem.fields.Integer'
    assert var_112.title == ''
    assert var_112.description == ''
    assert var_112.allow_null is False
    assert var_112.read_only is False
    assert var_112.minimum is None
    assert var_112.maximum is None
    assert var_112.exclusive_minimum is None
    assert var_112.exclusive_maximum is None
    assert var_112.multiple_of is None
    assert var_112.precision is None
    assert var_112.coerce_types is True
    var_113 = module_1.Boolean()
    assert f'{type(var_113).__module__}.{type(var_113).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_113.title == ''
    assert var_113.description == ''
    assert var_113.allow_null is False
    assert var_113.read_only is False
    assert var_113.coerce_types is True
    var_114 = module_2.IfThenElse(var_111, var_112, var_113)
    assert f'{type(var_114).__module__}.{type(var_114).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_114.title == ''
    assert var_114.description == ''
    assert var_114.allow_null is False
    assert var_114.read_only is False
    assert f'{type(var_114.if_clause).__module__}.{type(var_114.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_114.then_clause).__module__}.{type(var_114.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_114.else_clause).__module__}.{type(var_114.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_115 = module_0.to_json_schema(var_114)
    var_116 = module_1.String()
    assert f'{type(var_116).__module__}.{type(var_116).__qualname__}' == 'typesystem.fields.String'
    assert var_116.title == ''
    assert var_116.description == ''
    assert var_116.allow_null is False
    assert var_116.read_only is False
    assert var_116.allow_blank is False
    assert var_116.trim_whitespace is True
    assert var_116.max_length is None
    assert var_116.min_length is None
    assert var_116.format is None
    assert var_116.coerce_types is True
    assert var_116.pattern is None
    assert var_116.pattern_regex is None
    var_117 = module_2.IfThenElse(var_116, var_107)
    assert f'{type(var_117).__module__}.{type(var_117).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_117.title == ''
    assert var_117.description == ''
    assert var_117.allow_null is False
    assert var_117.read_only is False
    assert f'{type(var_117.if_clause).__module__}.{type(var_117.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert var_117.then_clause == 2
    assert f'{type(var_117.else_clause).__module__}.{type(var_117.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_0.to_json_schema(var_117)

def test_case_42():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 0
    var_8 = 100
    var_9 = 10
    var_10 = 90
    var_11 = 5
    var_12 = 50
    var_13 = {var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = 'number'
    var_15 = False
    var_16 = module_0.from_json_schema_type(var_13, var_14, var_15, var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Float'
    assert var_16.default == 50
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum == 0
    assert var_16.maximum == 100
    assert var_16.exclusive_minimum == 10
    assert var_16.exclusive_maximum == 90
    assert var_16.multiple_of == 5
    assert var_16.precision is None
    assert var_16.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_17 = True
    var_18 = module_0.from_json_schema_type(var_13, var_14, var_17, var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Float'
    assert var_18.default == 50
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is True
    assert var_18.read_only is False
    assert var_18.minimum == 0
    assert var_18.maximum == 100
    assert var_18.exclusive_minimum == 10
    assert var_18.exclusive_maximum == 90
    assert var_18.multiple_of == 5
    assert var_18.precision is None
    assert var_18.coerce_types is False
    var_19 = {var_1: var_17, var_2: var_9, var_6: var_11}
    var_20 = 'integer'
    var_21 = False
    var_22 = module_0.from_json_schema_type(var_19, var_20, var_21, var_0)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Integer'
    assert var_22.default == 5
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.minimum is True
    assert var_22.maximum == 10
    assert var_22.exclusive_minimum is None
    assert var_22.exclusive_maximum is None
    assert var_22.multiple_of is None
    assert var_22.precision is None
    assert var_22.coerce_types is False
    var_23 = 'minLength'
    var_24 = 'maxLength'
    var_25 = 'pattern'
    var_26 = 'format'
    var_27 = 2
    var_28 = '^[a-z]+$'
    var_29 = 'email'
    var_30 = 'test'
    var_31 = {var_23: var_27, var_24: var_12, var_25: var_28, var_26: var_29, var_6: var_30}
    var_32 = 'string'
    var_33 = module_0.from_json_schema_type(var_31, var_32, var_21, var_0)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.String'
    assert var_33.default == 'test'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.allow_blank is False
    assert var_33.trim_whitespace is True
    assert var_33.max_length == 50
    assert var_33.min_length == 2
    assert var_33.format == 'email'
    assert var_33.coerce_types is False
    assert var_33.pattern == '^[a-z]+$'
    assert f'{type(var_33.pattern_regex).__module__}.{type(var_33.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_34 = {var_23: var_15}
    var_35 = False
    var_36 = module_0.from_json_schema_type(var_34, var_32, var_35, var_0)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.String'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.default == ''
    assert var_36.allow_blank is True
    assert var_36.trim_whitespace is True
    assert var_36.max_length is None
    assert var_36.min_length is None
    assert var_36.format is None
    assert var_36.coerce_types is False
    assert var_36.pattern is None
    assert var_36.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_37 = {var_23: var_17}
    var_38 = False
    var_39 = module_0.from_json_schema_type(var_37, var_32, var_38, var_0)
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
    assert var_39.coerce_types is False
    assert var_39.pattern is None
    assert var_39.pattern_regex is None
    var_40 = 'boolean'
    var_41 = False
    var_42 = module_0.from_json_schema_type(var_34, var_40, var_41, var_0)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_42.title == ''
    assert var_42.description == ''
    assert var_42.allow_null is False
    assert var_42.read_only is False
    assert var_42.coerce_types is False
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_43 = module_0.from_json_schema_type(var_31, var_40, var_17, var_0)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_43.default == 'test'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is True
    assert var_43.read_only is False
    assert var_43.coerce_types is False
    var_44 = {}
    var_45 = 'array'
    var_46 = False
    var_47 = module_0.from_json_schema_type(var_44, var_45, var_46, var_0)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.Array'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert var_47.items is None
    assert var_47.additional_items is True
    assert var_47.min_items == 0
    assert var_47.max_items is None
    assert var_47.unique_items is False
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_48 = 'items'
    var_49 = 'minItems'
    var_50 = 'maxItems'
    var_51 = 'uniqueItems'
    var_52 = 'type'
    var_53 = {var_52: var_32}
    var_54 = 'a'
    var_55 = 'b'
    var_56 = [var_54, var_55]
    var_57 = {var_48: var_53, var_49: var_17, var_50: var_9, var_51: var_17, var_6: var_56}
    var_58 = False
    var_59 = module_0.from_json_schema_type(var_57, var_45, var_58, var_0)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.fields.Array'
    assert var_59.default == ['a', 'b']
    assert var_59.title == ''
    assert var_59.description == ''
    assert var_59.allow_null is False
    assert var_59.read_only is False
    assert f'{type(var_59.items).__module__}.{type(var_59.items).__qualname__}' == 'typesystem.fields.String'
    assert var_59.additional_items is True
    assert var_59.min_items is True
    assert var_59.max_items == 10
    assert var_59.unique_items is True
    var_60 = {var_52: var_32}
    var_61 = {var_52: var_20}
    var_62 = [var_60, var_61]
    var_63 = {var_48: var_62}
    var_64 = False
    var_65 = module_0.from_json_schema_type(var_63, var_45, var_64, var_0)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Array'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert f'{type(var_65.items).__module__}.{type(var_65.items).__qualname__}' == 'builtins.list'
    assert len(var_65.items) == 2
    assert var_65.additional_items is True
    assert var_65.min_items == 0
    assert var_65.max_items is None
    assert var_65.unique_items is False
    var_66 = var_65.items
    var_67 = var_65.items
    var_68 = len(var_67)
    assert var_68 == 2
    var_69 = 'additionalItems'
    var_70 = False
    var_71 = {var_69: var_70}
    var_72 = False
    var_73 = module_0.from_json_schema_type(var_71, var_45, var_72, var_0)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.Array'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert var_73.items is None
    assert var_73.additional_items is False
    assert var_73.min_items == 0
    assert var_73.max_items is None
    assert var_73.unique_items is False
    var_74 = {var_52: var_14}
    var_75 = {var_69: var_74}
    var_76 = False
    var_77 = module_0.from_json_schema_type(var_75, var_45, var_76, var_0)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.Array'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert var_77.items is None
    assert f'{type(var_77.additional_items).__module__}.{type(var_77.additional_items).__qualname__}' == 'typesystem.fields.Float'
    assert var_77.min_items == 0
    assert var_77.max_items is None
    assert var_77.unique_items is False
    var_78 = var_77.additional_items
    var_79 = {}
    var_80 = 'object'
    var_81 = False
    var_82 = module_0.from_json_schema_type(var_79, var_80, var_81, var_0)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.fields.Object'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert var_82.properties == {}
    assert var_82.pattern_properties == {}
    assert var_82.additional_properties is None
    assert var_82.property_names is None
    assert var_82.min_properties is None
    assert var_82.max_properties is None
    assert var_82.required == []
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_83 = 'properties'
    var_84 = 'required'
    var_85 = 'minProperties'
    var_86 = 'maxProperties'
    var_87 = '4|\ndUF'
    var_88 = 'age'
    var_89 = {var_52: var_32}
    var_90 = {var_52: var_20}
    var_91 = {var_87: var_89, var_88: var_90}
    var_92 = [var_87]
    var_93 = {var_83: var_91, var_84: var_92, var_85: var_17, var_86: var_11}
    var_94 = False
    var_95 = module_0.from_json_schema_type(var_93, var_80, var_94, var_0)
    assert f'{type(var_95).__module__}.{type(var_95).__qualname__}' == 'typesystem.fields.Object'
    assert var_95.title == ''
    assert var_95.description == ''
    assert var_95.allow_null is False
    assert var_95.read_only is False
    assert f'{type(var_95.properties).__module__}.{type(var_95.properties).__qualname__}' == 'builtins.dict'
    assert len(var_95.properties) == 2
    assert var_95.pattern_properties == {}
    assert var_95.additional_properties is None
    assert var_95.property_names is None
    assert var_95.min_properties is True
    assert var_95.max_properties == 5
    assert var_95.required == ['4|\ndUF']
    var_96 = 'patternProperties'
    var_97 = '^S_'
    var_98 = '^I_'
    var_99 = {var_52: var_32}
    var_100 = {var_52: var_20}
    var_101 = {var_97: var_99, var_98: var_100}
    var_102 = {var_96: var_101}
    var_103 = False
    var_104 = module_0.from_json_schema_type(var_102, var_80, var_103, var_0)
    assert f'{type(var_104).__module__}.{type(var_104).__qualname__}' == 'typesystem.fields.Object'
    assert var_104.title == ''
    assert var_104.description == ''
    assert var_104.allow_null is False
    assert var_104.read_only is False
    assert var_104.properties == {}
    assert f'{type(var_104.pattern_properties).__module__}.{type(var_104.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_104.pattern_properties) == 2
    assert var_104.additional_properties is None
    assert var_104.property_names is None
    assert var_104.min_properties is None
    assert var_104.max_properties is None
    assert var_104.required == []
    var_105 = 'additionalProperties'
    var_106 = False
    var_107 = {var_105: var_106}
    var_108 = False
    var_109 = module_0.from_json_schema_type(var_107, var_80, var_108, var_0)
    assert f'{type(var_109).__module__}.{type(var_109).__qualname__}' == 'typesystem.fields.Object'
    assert var_109.title == ''
    assert var_109.description == ''
    assert var_109.allow_null is False
    assert var_109.read_only is False
    assert var_109.properties == {}
    assert var_109.pattern_properties == {}
    assert var_109.additional_properties is False
    assert var_109.property_names is None
    assert var_109.min_properties is None
    assert var_109.max_properties is None
    assert var_109.required == []
    var_110 = {var_52: var_32}
    var_111 = {var_105: var_110}
    var_112 = False
    var_113 = module_0.from_json_schema_type(var_111, var_80, var_112, var_0)
    assert f'{type(var_113).__module__}.{type(var_113).__qualname__}' == 'typesystem.fields.Object'
    assert var_113.title == ''
    assert var_113.description == ''
    assert var_113.allow_null is False
    assert var_113.read_only is False
    assert var_113.properties == {}
    assert var_113.pattern_properties == {}
    assert f'{type(var_113.additional_properties).__module__}.{type(var_113.additional_properties).__qualname__}' == 'typesystem.fields.String'
    assert var_113.property_names is None
    assert var_113.min_properties is None
    assert var_113.max_properties is None
    assert var_113.required == []
    var_114 = var_113.additional_properties
    var_115 = 'propertyNames'
    var_116 = {var_25: var_28}
    var_117 = {var_115: var_116}
    var_118 = False
    var_119 = module_0.from_json_schema_type(var_117, var_80, var_118, var_0)
    assert f'{type(var_119).__module__}.{type(var_119).__qualname__}' == 'typesystem.fields.Object'
    assert var_119.title == ''
    assert var_119.description == ''
    assert var_119.allow_null is False
    assert var_119.read_only is False
    assert var_119.properties == {}
    assert var_119.pattern_properties == {}
    assert var_119.additional_properties is None
    assert f'{type(var_119.property_names).__module__}.{type(var_119.property_names).__qualname__}' == 'typesystem.fields.Union'
    assert var_119.min_properties is None
    assert var_119.max_properties is None
    assert var_119.required == []
    var_120 = var_119.property_names
    var_121 = {}
    var_122 = module_0.from_json_schema_type(var_121, var_80, var_17, var_0)
    assert f'{type(var_122).__module__}.{type(var_122).__qualname__}' == 'typesystem.fields.Object'
    assert var_122.default is None
    assert var_122.title == ''
    assert var_122.description == ''
    assert var_122.allow_null is True
    assert var_122.read_only is False
    assert var_122.properties == {}
    assert var_122.pattern_properties == {}
    assert var_122.additional_properties is None
    assert var_122.property_names is None
    assert var_122.min_properties is None
    assert var_122.max_properties is None
    assert var_122.required == []

def test_case_43():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = 'schemas/User'
    var_3 = {var_1: var_2}
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_3, var_0)

@pytest.mark.xfail(strict=True)
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_2.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = '#n'
    var_9 = module_1.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.String'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.allow_blank is False
    assert var_9.trim_whitespace is True
    assert var_9.max_length == 10
    assert var_9.min_length == 1
    assert var_9.format == '#n'
    assert var_9.coerce_types is True
    assert var_9.pattern == '^[a-z]+$'
    assert f'{type(var_9.pattern_regex).__module__}.{type(var_9.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = module_0.to_json_schema(var_9)
    var_12 = True
    var_13 = None
    var_14 = module_1.String(allow_blank=var_12, min_length=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.String'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.default == ''
    assert var_14.allow_blank is True
    assert var_14.trim_whitespace is True
    assert var_14.max_length is None
    assert var_14.min_length is None
    assert var_14.format is None
    assert var_14.coerce_types is True
    assert var_14.pattern is None
    assert var_14.pattern_regex is None
    var_15 = module_0.to_json_schema(var_14)
    var_16 = 100
    var_17 = 99
    var_18 = 5
    var_19 = module_1.Integer(minimum=var_4, maximum=var_16, exclusive_minimum=var_12, exclusive_maximum=var_17, multiple_of=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Integer'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.minimum is False
    assert var_19.maximum == 100
    assert var_19.exclusive_minimum is True
    assert var_19.exclusive_maximum == 99
    assert var_19.multiple_of == 5
    assert var_19.precision is None
    assert var_19.coerce_types is True
    var_20 = module_0.to_json_schema(var_19)
    var_21 = module_1.Float(minimum=var_4, maximum=var_12)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Float'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.minimum is False
    assert var_21.maximum is True
    assert var_21.exclusive_minimum is None
    assert var_21.exclusive_maximum is None
    assert var_21.multiple_of is None
    assert var_21.precision is None
    assert var_21.coerce_types is True
    var_22 = module_0.to_json_schema(var_21)
    var_23 = True
    var_24 = module_1.Boolean()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
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
    var_27 = True
    var_28 = module_1.Array(var_26, min_items=var_23, max_items=var_18, unique_items=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Array'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.items).__module__}.{type(var_28.items).__qualname__}' == 'typesystem.fields.String'
    assert var_28.additional_items is False
    assert var_28.min_items is True
    assert var_28.max_items == 5
    assert var_28.unique_items is True
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = module_1.String()
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
    var_31 = module_1.Integer()
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
    var_33 = module_1.Array(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Array'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.items).__module__}.{type(var_33.items).__qualname__}' == 'builtins.list'
    assert len(var_33.items) == 2
    assert var_33.additional_items is False
    assert var_33.min_items == 2
    assert var_33.max_items == 2
    assert var_33.unique_items is False
    var_34 = module_0.to_json_schema(var_33)
    var_35 = 'items'
    var_36 = var_34[var_35]
    var_37 = var_34[var_35]
    var_38 = len(var_37)
    assert var_38 == 2
    var_39 = module_1.Array(additional_items=var_4)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Array'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.items is None
    assert var_39.additional_items is False
    assert var_39.min_items is None
    assert var_39.max_items is None
    assert var_39.unique_items is False
    var_40 = module_0.to_json_schema(var_39)
    var_41 = 'name'
    var_42 = 'age'
    var_43 = module_1.String()
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
    var_44 = module_1.Integer()
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.fields.Integer'
    assert var_44.title == ''
    assert var_44.description == ''
    assert var_44.allow_null is False
    assert var_44.read_only is False
    assert var_44.minimum is None
    assert var_44.maximum is None
    assert var_44.exclusive_minimum is None
    assert var_44.exclusive_maximum is None
    assert var_44.multiple_of is None
    assert var_44.precision is None
    assert var_44.coerce_types is True
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = [var_41]
    var_47 = module_1.Object(properties=var_45, required=var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.Object'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert f'{type(var_47.properties).__module__}.{type(var_47.properties).__qualname__}' == 'builtins.dict'
    assert len(var_47.properties) == 2
    assert var_47.pattern_properties == {}
    assert var_47.additional_properties is True
    assert var_47.property_names is None
    assert var_47.min_properties is None
    assert var_47.max_properties is None
    assert var_47.required == ['name']
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_48 = module_0.to_json_schema(var_47)
    var_49 = '^S_'
    var_50 = '^I_'
    var_51 = module_1.String()
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
    var_52 = module_1.Integer()
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
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = module_1.Object(pattern_properties=var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Object'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert var_54.properties == {}
    assert f'{type(var_54.pattern_properties).__module__}.{type(var_54.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_54.pattern_properties) == 2
    assert var_54.additional_properties is True
    assert var_54.property_names is None
    assert var_54.min_properties is None
    assert var_54.max_properties is None
    assert var_54.required == []
    var_55 = True
    var_56 = module_1.Object(additional_properties=var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.fields.Object'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert var_56.properties == {}
    assert var_56.pattern_properties == {}
    assert var_56.additional_properties is True
    assert var_56.property_names is None
    assert var_56.min_properties is None
    assert var_56.max_properties is None
    assert var_56.required == []
    var_57 = module_0.to_json_schema(var_56)
    var_58 = module_1.Object(property_names=var_9)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.Object'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.properties == {}
    assert var_58.pattern_properties == {}
    assert var_58.additional_properties is True
    assert f'{type(var_58.property_names).__module__}.{type(var_58.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_58.min_properties is None
    assert var_58.max_properties is None
    assert var_58.required == []
    var_59 = module_0.to_json_schema(var_58)
    var_60 = 'a'
    var_61 = (var_60, var_55)
    var_62 = 'b'
    var_63 = 2
    var_64 = (var_62, var_63)
    var_65 = [var_61, var_64]
    var_66 = module_1.Choice(choices=var_65)
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.fields.Choice'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert var_66.choices == [('a', True), ('b', 2)]
    assert var_66.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_67 = module_0.to_json_schema(var_66)
    var_68 = 'fixed_value'
    var_69 = module_1.Const(var_68)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.fields.Const'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert var_69.const == 'fixed_value'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_70 = module_0.to_json_schema(var_69)
    var_71 = module_1.String()
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.String'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.allow_blank is False
    assert var_71.trim_whitespace is True
    assert var_71.max_length is None
    assert var_71.min_length is None
    assert var_71.format is None
    assert var_71.coerce_types is True
    assert var_71.pattern is None
    assert var_71.pattern_regex is None
    var_72 = module_1.Integer()
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.Integer'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert var_72.minimum is None
    assert var_72.maximum is None
    assert var_72.exclusive_minimum is None
    assert var_72.exclusive_maximum is None
    assert var_72.multiple_of is None
    assert var_72.precision is None
    assert var_72.coerce_types is True
    var_73 = [var_71, var_72]
    var_74 = module_1.Union(var_73)
    assert f'{type(var_74).__module__}.{type(var_74).__qualname__}' == 'typesystem.fields.Union'
    assert var_74.title == ''
    assert var_74.description == ''
    assert var_74.allow_null is False
    assert var_74.read_only is False
    assert f'{type(var_74.any_of).__module__}.{type(var_74.any_of).__qualname__}' == 'builtins.list'
    assert len(var_74.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_75 = module_0.to_json_schema(var_74)
    var_76 = 'anyOf'
    var_77 = var_75[var_76]
    var_78 = len(var_77)
    assert var_78 == 2
    var_79 = module_1.String()
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.fields.String'
    assert var_79.title == ''
    assert var_79.description == ''
    assert var_79.allow_null is False
    assert var_79.read_only is False
    assert var_79.allow_blank is False
    assert var_79.trim_whitespace is True
    assert var_79.max_length is None
    assert var_79.min_length is None
    assert var_79.format is None
    assert var_79.coerce_types is True
    assert var_79.pattern is None
    assert var_79.pattern_regex is None
    var_80 = module_1.Integer()
    assert f'{type(var_80).__module__}.{type(var_80).__qualname__}' == 'typesystem.fields.Integer'
    assert var_80.title == ''
    assert var_80.description == ''
    assert var_80.allow_null is False
    assert var_80.read_only is False
    assert var_80.minimum is None
    assert var_80.maximum is None
    assert var_80.exclusive_minimum is None
    assert var_80.exclusive_maximum is None
    assert var_80.multiple_of is None
    assert var_80.precision is None
    assert var_80.coerce_types is True
    var_81 = [var_79, var_80]
    var_82 = module_2.OneOf(var_81)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_82.title == ''
    assert var_82.description == ''
    assert var_82.allow_null is False
    assert var_82.read_only is False
    assert f'{type(var_82.one_of).__module__}.{type(var_82.one_of).__qualname__}' == 'builtins.list'
    assert len(var_82.one_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_83 = module_0.to_json_schema(var_82)
    var_84 = len(var_67)
    var_85 = module_1.String(min_length=var_18)
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
    var_86 = [var_14, var_85]
    var_87 = module_2.AllOf(var_86)
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_87.title == ''
    assert var_87.description == ''
    assert var_87.allow_null is False
    assert var_87.read_only is False
    assert f'{type(var_87.all_of).__module__}.{type(var_87.all_of).__qualname__}' == 'builtins.list'
    assert len(var_87.all_of) == 2
    var_88 = module_0.to_json_schema(var_87)
    var_89 = 'allOf'
    var_90 = var_88[var_89]
    var_91 = len(var_90)
    assert var_91 == 2
    var_92 = module_1.String()
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
    var_93 = module_1.Integer()
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.fields.Integer'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert var_93.minimum is None
    assert var_93.maximum is None
    assert var_93.exclusive_minimum is None
    assert var_93.exclusive_maximum is None
    assert var_93.multiple_of is None
    assert var_93.precision is None
    assert var_93.coerce_types is True
    var_94 = module_1.Boolean()
    assert f'{type(var_94).__module__}.{type(var_94).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_94.title == ''
    assert var_94.description == ''
    assert var_94.allow_null is False
    assert var_94.read_only is False
    assert var_94.coerce_types is True
    var_95 = module_2.IfThenElse(var_92, var_93, var_94)
    assert f'{type(var_95).__module__}.{type(var_95).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_95.title == ''
    assert var_95.description == ''
    assert var_95.allow_null is False
    assert var_95.read_only is False
    assert f'{type(var_95.if_clause).__module__}.{type(var_95.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_95.then_clause).__module__}.{type(var_95.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_95.else_clause).__module__}.{type(var_95.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_96 = module_0.to_json_schema(var_95)
    var_97 = module_1.String()
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
    var_98 = module_1.Integer()
    assert f'{type(var_98).__module__}.{type(var_98).__qualname__}' == 'typesystem.fields.Integer'
    assert var_98.title == ''
    assert var_98.description == ''
    assert var_98.allow_null is False
    assert var_98.read_only is False
    assert var_98.minimum is None
    assert var_98.maximum is None
    assert var_98.exclusive_minimum is None
    assert var_98.exclusive_maximum is None
    assert var_98.multiple_of is None
    assert var_98.precision is None
    assert var_98.coerce_types is True
    var_99 = module_2.IfThenElse(var_97, var_98)
    assert f'{type(var_99).__module__}.{type(var_99).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_99.title == ''
    assert var_99.description == ''
    assert var_99.allow_null is False
    assert var_99.read_only is False
    assert f'{type(var_99.if_clause).__module__}.{type(var_99.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_99.then_clause).__module__}.{type(var_99.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_99.else_clause).__module__}.{type(var_99.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_100 = module_0.to_json_schema(var_99)
    var_101 = module_2.Not(var_71)
    assert f'{type(var_101).__module__}.{type(var_101).__qualname__}' == 'typesystem.composites.Not'
    assert var_101.title == ''
    assert var_101.description == ''
    assert var_101.allow_null is False
    assert var_101.read_only is False
    assert f'{type(var_101.negated).__module__}.{type(var_101.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_102 = module_0.to_json_schema(var_101)
    var_103 = module_3.Definitions()
    assert f'{type(var_103).__module__}.{type(var_103).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_103) == 0
    var_104 = module_0.to_json_schema(var_103)
    var_105 = module_3.Definitions()
    assert f'{type(var_105).__module__}.{type(var_105).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_105) == 0
    module_0.to_json_schema(var_38)

@pytest.mark.xfail(strict=True)
def test_case_45():
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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_2.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_1.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
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
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = True
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
    var_13 = module_0.to_json_schema(var_12)
    var_14 = True
    var_15 = None
    var_16 = module_1.String(allow_blank=var_14, min_length=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.String'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.default == ''
    assert var_16.allow_blank is True
    assert var_16.trim_whitespace is True
    assert var_16.max_length is None
    assert var_16.min_length is None
    assert var_16.format is None
    assert var_16.coerce_types is True
    assert var_16.pattern is None
    assert var_16.pattern_regex is None
    var_17 = module_0.to_json_schema(var_16)
    var_18 = 100
    var_19 = 5
    var_20 = module_1.Integer(minimum=var_4, maximum=var_18, exclusive_minimum=var_14, exclusive_maximum=var_1, multiple_of=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Integer'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.minimum is False
    assert var_20.maximum == 100
    assert var_20.exclusive_minimum is True
    assert var_20.exclusive_maximum is True
    assert var_20.multiple_of == 5
    assert var_20.precision is None
    assert var_20.coerce_types is True
    var_21 = module_0.to_json_schema(var_20)
    var_22 = module_1.Float(minimum=var_4, maximum=var_14)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Float'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert var_22.minimum is False
    assert var_22.maximum is True
    assert var_22.exclusive_minimum is None
    assert var_22.exclusive_maximum is None
    assert var_22.multiple_of is None
    assert var_22.precision is None
    assert var_22.coerce_types is True
    var_23 = module_1.Boolean()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_24 = module_0.to_json_schema(var_23)
    var_25 = True
    var_26 = module_1.Boolean()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.coerce_types is True
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
    var_29 = True
    var_30 = module_1.Array(var_28, min_items=var_25, max_items=var_19, unique_items=var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Array'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.items).__module__}.{type(var_30.items).__qualname__}' == 'typesystem.fields.String'
    assert var_30.additional_items is False
    assert var_30.min_items is True
    assert var_30.max_items == 5
    assert var_30.unique_items is True
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
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
    var_35 = module_1.Array(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Array'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.items).__module__}.{type(var_35.items).__qualname__}' == 'builtins.list'
    assert len(var_35.items) == 2
    assert var_35.additional_items is False
    assert var_35.min_items == 2
    assert var_35.max_items == 2
    assert var_35.unique_items is False
    var_36 = module_0.to_json_schema(var_35)
    var_37 = 'items'
    var_38 = var_36[var_37]
    var_39 = var_36[var_37]
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = module_1.Array(additional_items=var_4)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Array'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.items is None
    assert var_41.additional_items is False
    assert var_41.min_items is None
    assert var_41.max_items is None
    assert var_41.unique_items is False
    var_42 = 'name'
    var_43 = 'age'
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
    var_45 = module_1.Integer()
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Integer'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.minimum is None
    assert var_45.maximum is None
    assert var_45.exclusive_minimum is None
    assert var_45.exclusive_maximum is None
    assert var_45.multiple_of is None
    assert var_45.precision is None
    assert var_45.coerce_types is True
    var_46 = {var_42: var_44, var_43: var_45}
    var_47 = [var_42]
    var_48 = module_1.Object(properties=var_46, required=var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.Object'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert f'{type(var_48.properties).__module__}.{type(var_48.properties).__qualname__}' == 'builtins.dict'
    assert len(var_48.properties) == 2
    assert var_48.pattern_properties == {}
    assert var_48.additional_properties is True
    assert var_48.property_names is None
    assert var_48.min_properties is None
    assert var_48.max_properties is None
    assert var_48.required == ['name']
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_49 = module_0.to_json_schema(var_48)
    var_50 = '^S_'
    var_51 = '^I_'
    var_52 = module_1.String()
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
    var_53 = module_1.Integer()
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
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = module_1.Object(pattern_properties=var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Object'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.properties == {}
    assert f'{type(var_55.pattern_properties).__module__}.{type(var_55.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_55.pattern_properties) == 2
    assert var_55.additional_properties is True
    assert var_55.property_names is None
    assert var_55.min_properties is None
    assert var_55.max_properties is None
    assert var_55.required == []
    var_56 = module_0.to_json_schema(var_55)
    var_57 = True
    var_58 = module_1.Object(additional_properties=var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.Object'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.properties == {}
    assert var_58.pattern_properties == {}
    assert var_58.additional_properties is True
    assert var_58.property_names is None
    assert var_58.min_properties is None
    assert var_58.max_properties is None
    assert var_58.required == []
    var_59 = module_0.to_json_schema(var_58)
    var_60 = module_1.String(pattern=var_7)
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
    assert var_60.pattern == '^[a-z]+$'
    assert f'{type(var_60.pattern_regex).__module__}.{type(var_60.pattern_regex).__qualname__}' == 're.Pattern'
    var_61 = module_1.Object(property_names=var_60)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.fields.Object'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert var_61.properties == {}
    assert var_61.pattern_properties == {}
    assert var_61.additional_properties is True
    assert f'{type(var_61.property_names).__module__}.{type(var_61.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_61.min_properties is None
    assert var_61.max_properties is None
    assert var_61.required == []
    var_62 = module_0.to_json_schema(var_61)
    var_63 = 'a'
    var_64 = (var_63, var_57)
    var_65 = 'b'
    var_66 = 2
    var_67 = (var_65, var_66)
    var_68 = [var_64, var_67]
    var_69 = module_1.Choice(choices=var_68)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'typesystem.fields.Choice'
    assert var_69.title == ''
    assert var_69.description == ''
    assert var_69.allow_null is False
    assert var_69.read_only is False
    assert var_69.choices == [('a', True), ('b', 2)]
    assert var_69.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_70 = module_0.to_json_schema(var_69)
    var_71 = 'fixed_value'
    var_72 = module_1.Const(var_71)
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'typesystem.fields.Const'
    assert var_72.title == ''
    assert var_72.description == ''
    assert var_72.allow_null is False
    assert var_72.read_only is False
    assert var_72.const == 'fixed_value'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_73 = module_0.to_json_schema(var_72)
    var_74 = module_1.String()
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
    var_75 = module_1.Integer()
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
    var_76 = [var_74, var_75]
    var_77 = module_1.Union(var_76)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.Union'
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert f'{type(var_77.any_of).__module__}.{type(var_77.any_of).__qualname__}' == 'builtins.list'
    assert len(var_77.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_78 = module_0.to_json_schema(var_77)
    var_79 = 'anyOf'
    var_80 = var_78[var_79]
    var_81 = len(var_80)
    assert var_81 == 2
    var_82 = module_1.String()
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
    var_83 = module_1.Integer()
    assert f'{type(var_83).__module__}.{type(var_83).__qualname__}' == 'typesystem.fields.Integer'
    assert var_83.title == ''
    assert var_83.description == ''
    assert var_83.allow_null is False
    assert var_83.read_only is False
    assert var_83.minimum is None
    assert var_83.maximum is None
    assert var_83.exclusive_minimum is None
    assert var_83.exclusive_maximum is None
    assert var_83.multiple_of is None
    assert var_83.precision is None
    assert var_83.coerce_types is True
    var_84 = [var_82, var_83]
    var_85 = module_2.OneOf(var_84)
    assert f'{type(var_85).__module__}.{type(var_85).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_85.title == ''
    assert var_85.description == ''
    assert var_85.allow_null is False
    assert var_85.read_only is False
    assert f'{type(var_85.one_of).__module__}.{type(var_85.one_of).__qualname__}' == 'builtins.list'
    assert len(var_85.one_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_86 = module_0.to_json_schema(var_85)
    var_87 = 'oneOf'
    var_88 = var_86[var_87]
    var_89 = len(var_88)
    assert var_89 == 2
    var_90 = module_1.String()
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
    var_91 = module_1.String(min_length=var_19)
    assert f'{type(var_91).__module__}.{type(var_91).__qualname__}' == 'typesystem.fields.String'
    assert var_91.title == ''
    assert var_91.description == ''
    assert var_91.allow_null is False
    assert var_91.read_only is False
    assert var_91.allow_blank is False
    assert var_91.trim_whitespace is True
    assert var_91.max_length is None
    assert var_91.min_length == 5
    assert var_91.format is None
    assert var_91.coerce_types is True
    assert var_91.pattern is None
    assert var_91.pattern_regex is None
    var_92 = [var_90, var_91]
    var_93 = module_2.AllOf(var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_93.title == ''
    assert var_93.description == ''
    assert var_93.allow_null is False
    assert var_93.read_only is False
    assert f'{type(var_93.all_of).__module__}.{type(var_93.all_of).__qualname__}' == 'builtins.list'
    assert len(var_93.all_of) == 2
    var_94 = module_0.to_json_schema(var_93)
    var_95 = 'allOf'
    var_96 = var_94[var_95]
    var_97 = len(var_96)
    assert var_97 == 2
    var_98 = module_1.String()
    assert f'{type(var_98).__module__}.{type(var_98).__qualname__}' == 'typesystem.fields.String'
    assert var_98.title == ''
    assert var_98.description == ''
    assert var_98.allow_null is False
    assert var_98.read_only is False
    assert var_98.allow_blank is False
    assert var_98.trim_whitespace is True
    assert var_98.max_length is None
    assert var_98.min_length is None
    assert var_98.format is None
    assert var_98.coerce_types is True
    assert var_98.pattern is None
    assert var_98.pattern_regex is None
    var_99 = module_1.Integer()
    assert f'{type(var_99).__module__}.{type(var_99).__qualname__}' == 'typesystem.fields.Integer'
    assert var_99.title == ''
    assert var_99.description == ''
    assert var_99.allow_null is False
    assert var_99.read_only is False
    assert var_99.minimum is None
    assert var_99.maximum is None
    assert var_99.exclusive_minimum is None
    assert var_99.exclusive_maximum is None
    assert var_99.multiple_of is None
    assert var_99.precision is None
    assert var_99.coerce_types is True
    var_100 = module_1.Boolean()
    assert f'{type(var_100).__module__}.{type(var_100).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_100.title == ''
    assert var_100.description == ''
    assert var_100.allow_null is False
    assert var_100.read_only is False
    assert var_100.coerce_types is True
    var_101 = module_2.IfThenElse(var_98, var_99, var_100)
    assert f'{type(var_101).__module__}.{type(var_101).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_101.title == ''
    assert var_101.description == ''
    assert var_101.allow_null is False
    assert var_101.read_only is False
    assert f'{type(var_101.if_clause).__module__}.{type(var_101.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_101.then_clause).__module__}.{type(var_101.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_101.else_clause).__module__}.{type(var_101.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_102 = module_0.to_json_schema(var_101)
    var_103 = module_1.String()
    assert f'{type(var_103).__module__}.{type(var_103).__qualname__}' == 'typesystem.fields.String'
    assert var_103.title == ''
    assert var_103.description == ''
    assert var_103.allow_null is False
    assert var_103.read_only is False
    assert var_103.allow_blank is False
    assert var_103.trim_whitespace is True
    assert var_103.max_length is None
    assert var_103.min_length is None
    assert var_103.format is None
    assert var_103.coerce_types is True
    assert var_103.pattern is None
    assert var_103.pattern_regex is None
    var_104 = module_1.Integer()
    assert f'{type(var_104).__module__}.{type(var_104).__qualname__}' == 'typesystem.fields.Integer'
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
    var_105 = module_1.String()
    assert f'{type(var_105).__module__}.{type(var_105).__qualname__}' == 'typesystem.fields.String'
    assert var_105.title == ''
    assert var_105.description == ''
    assert var_105.allow_null is False
    assert var_105.read_only is False
    assert var_105.allow_blank is False
    assert var_105.trim_whitespace is True
    assert var_105.max_length is None
    assert var_105.min_length is None
    assert var_105.format is None
    assert var_105.coerce_types is True
    assert var_105.pattern is None
    assert var_105.pattern_regex is None
    var_106 = module_2.Not(var_105)
    assert f'{type(var_106).__module__}.{type(var_106).__qualname__}' == 'typesystem.composites.Not'
    assert var_106.title == ''
    assert var_106.description == ''
    assert var_106.allow_null is False
    assert var_106.read_only is False
    assert f'{type(var_106.negated).__module__}.{type(var_106.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_107 = module_0.to_json_schema(var_106)
    var_108 = module_3.Definitions()
    assert f'{type(var_108).__module__}.{type(var_108).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_108) == 0
    var_109 = module_0.to_json_schema(var_108)
    var_110 = 'CustomType'
    var_111 = module_3.Definitions()
    assert f'{type(var_111).__module__}.{type(var_111).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_111) == 0
    var_112 = module_3.Reference(var_110, var_111)
    assert f'{type(var_112).__module__}.{type(var_112).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_112.title == ''
    assert var_112.description == ''
    assert var_112.allow_null is False
    assert var_112.read_only is False
    assert var_112.to == 'CustomType'
    assert f'{type(var_112.definitions).__module__}.{type(var_112.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_112.definitions) == 0
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    module_0.to_json_schema(var_112)

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
    assert module_0.TYPE_CONSTRAINTS == {'contains', 'minLength', 'maxProperties', 'uniqueItems', 'additionalProperties', 'items', 'exclusiveMinimum', 'required', 'boolean_schema', 'additionalItems', 'type', 'minItems', 'patternProperties', 'pattern', 'multipleOf', 'maximum', 'properties', 'maxItems', 'propertyNames', 'maxLength', 'exclusiveMaximum', 'minimum', 'minProperties', 'dependencies'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_2.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_2.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = False
    var_4 = 1
    var_5 = 10
    var_6 = '^[a-z]+$'
    var_7 = '#n'
    var_8 = module_1.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.String'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.allow_blank is False
    assert var_8.trim_whitespace is True
    assert var_8.max_length == 10
    assert var_8.min_length == 1
    assert var_8.format == '#n'
    assert var_8.coerce_types is True
    assert var_8.pattern == '^[a-z]+$'
    assert f'{type(var_8.pattern_regex).__module__}.{type(var_8.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = module_0.to_json_schema(var_8)
    var_10 = module_0.to_json_schema(var_8)
    var_11 = True
    var_12 = None
    var_13 = module_1.String(allow_blank=var_11, min_length=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.String'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.default == ''
    assert var_13.allow_blank is True
    assert var_13.trim_whitespace is True
    assert var_13.max_length is None
    assert var_13.min_length is None
    assert var_13.format is None
    assert var_13.coerce_types is True
    assert var_13.pattern is None
    assert var_13.pattern_regex is None
    var_14 = module_0.to_json_schema(var_13)
    var_15 = 100
    var_16 = 99
    var_17 = 5
    var_18 = module_1.Integer(minimum=var_3, maximum=var_15, exclusive_minimum=var_11, exclusive_maximum=var_16, multiple_of=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Integer'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.minimum is False
    assert var_18.maximum == 100
    assert var_18.exclusive_minimum is True
    assert var_18.exclusive_maximum == 99
    assert var_18.multiple_of == 5
    assert var_18.precision is None
    assert var_18.coerce_types is True
    var_19 = module_0.to_json_schema(var_18)
    var_20 = module_1.Float(minimum=var_3, maximum=var_11)
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
    var_21 = module_0.to_json_schema(var_20)
    var_22 = True
    var_23 = module_1.Boolean()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.coerce_types is True
    assert module_1.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_1.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_1.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_24 = module_0.to_json_schema(var_23)
    var_25 = module_1.String()
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
    var_27 = module_1.Array(var_25, min_items=var_22, max_items=var_17, unique_items=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Array'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert f'{type(var_27.items).__module__}.{type(var_27.items).__qualname__}' == 'typesystem.fields.String'
    assert var_27.additional_items is False
    assert var_27.min_items is True
    assert var_27.max_items == 5
    assert var_27.unique_items is True
    assert module_1.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_28 = module_0.to_json_schema(var_27)
    var_29 = module_1.String()
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
    var_30 = module_1.Integer()
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
    var_31 = [var_29, var_30]
    var_32 = module_1.Array(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.fields.Array'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.items).__module__}.{type(var_32.items).__qualname__}' == 'builtins.list'
    assert len(var_32.items) == 2
    assert var_32.additional_items is False
    assert var_32.min_items == 2
    assert var_32.max_items == 2
    assert var_32.unique_items is False
    var_33 = module_0.to_json_schema(var_32)
    var_34 = 'items'
    var_35 = var_33[var_34]
    var_36 = var_33[var_34]
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = module_1.Array(additional_items=var_3)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Array'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.items is None
    assert var_38.additional_items is False
    assert var_38.min_items is None
    assert var_38.max_items is None
    assert var_38.unique_items is False
    var_39 = module_0.to_json_schema(var_38)
    var_40 = 'name'
    var_41 = 'age'
    var_42 = module_1.String()
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
    var_43 = module_1.Integer()
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
    var_44 = {var_40: var_42, var_41: var_43}
    var_45 = [var_40]
    var_46 = module_1.Object(properties=var_44, required=var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Object'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.properties).__module__}.{type(var_46.properties).__qualname__}' == 'builtins.dict'
    assert len(var_46.properties) == 2
    assert var_46.pattern_properties == {}
    assert var_46.additional_properties is True
    assert var_46.property_names is None
    assert var_46.min_properties is None
    assert var_46.max_properties is None
    assert var_46.required == ['name']
    assert module_1.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_47 = module_0.to_json_schema(var_46)
    var_48 = '^S_'
    var_49 = '^I_'
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
    var_51 = module_1.Integer()
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.fields.Integer'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert var_51.minimum is None
    assert var_51.maximum is None
    assert var_51.exclusive_minimum is None
    assert var_51.exclusive_maximum is None
    assert var_51.multiple_of is None
    assert var_51.precision is None
    assert var_51.coerce_types is True
    var_52 = {var_48: var_50, var_49: var_51}
    var_53 = module_1.Object(pattern_properties=var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.fields.Object'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert var_53.properties == {}
    assert f'{type(var_53.pattern_properties).__module__}.{type(var_53.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_53.pattern_properties) == 2
    assert var_53.additional_properties is True
    assert var_53.property_names is None
    assert var_53.min_properties is None
    assert var_53.max_properties is None
    assert var_53.required == []
    var_54 = True
    var_55 = module_1.Object(additional_properties=var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Object'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.properties == {}
    assert var_55.pattern_properties == {}
    assert var_55.additional_properties is True
    assert var_55.property_names is None
    assert var_55.min_properties is None
    assert var_55.max_properties is None
    assert var_55.required == []
    var_56 = module_0.to_json_schema(var_55)
    var_57 = module_1.Object(property_names=var_8)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.fields.Object'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert var_57.properties == {}
    assert var_57.pattern_properties == {}
    assert var_57.additional_properties is True
    assert f'{type(var_57.property_names).__module__}.{type(var_57.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_57.min_properties is None
    assert var_57.max_properties is None
    assert var_57.required == []
    var_58 = module_0.to_json_schema(var_57)
    var_59 = 'a'
    var_60 = (var_59, var_54)
    var_61 = 'b'
    var_62 = 2
    var_63 = (var_61, var_62)
    var_64 = [var_60, var_63]
    var_65 = module_1.Choice(choices=var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.fields.Choice'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert var_65.choices == [('a', True), ('b', 2)]
    assert var_65.coerce_types is True
    assert module_1.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_66 = module_0.to_json_schema(var_65)
    var_67 = 'fixed_value'
    var_68 = module_1.Const(var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.fields.Const'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert var_68.const == 'fixed_value'
    assert module_1.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_69 = module_0.to_json_schema(var_68)
    var_70 = module_1.String()
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.fields.String'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert var_70.allow_blank is False
    assert var_70.trim_whitespace is True
    assert var_70.max_length is None
    assert var_70.min_length is None
    assert var_70.format is None
    assert var_70.coerce_types is True
    assert var_70.pattern is None
    assert var_70.pattern_regex is None
    var_71 = module_1.Integer()
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.fields.Integer'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert var_71.minimum is None
    assert var_71.maximum is None
    assert var_71.exclusive_minimum is None
    assert var_71.exclusive_maximum is None
    assert var_71.multiple_of is None
    assert var_71.precision is None
    assert var_71.coerce_types is True
    var_72 = [var_70, var_71]
    var_73 = module_1.Union(var_72)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'typesystem.fields.Union'
    assert var_73.title == ''
    assert var_73.description == ''
    assert var_73.allow_null is False
    assert var_73.read_only is False
    assert f'{type(var_73.any_of).__module__}.{type(var_73.any_of).__qualname__}' == 'builtins.list'
    assert len(var_73.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_74 = module_0.to_json_schema(var_73)
    var_75 = 'anyOf'
    var_76 = var_74[var_75]
    var_77 = len(var_76)
    assert var_77 == 2
    var_78 = module_1.String()
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
    var_79 = module_1.Integer()
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'typesystem.fields.Integer'
    assert var_79.title == ''
    assert var_79.description == ''
    assert var_79.allow_null is False
    assert var_79.read_only is False
    assert var_79.minimum is None
    assert var_79.maximum is None
    assert var_79.exclusive_minimum is None
    assert var_79.exclusive_maximum is None
    assert var_79.multiple_of is None
    assert var_79.precision is None
    assert var_79.coerce_types is True
    var_80 = [var_78, var_79]
    var_81 = module_2.OneOf(var_80)
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_81.title == ''
    assert var_81.description == ''
    assert var_81.allow_null is False
    assert var_81.read_only is False
    assert f'{type(var_81.one_of).__module__}.{type(var_81.one_of).__qualname__}' == 'builtins.list'
    assert len(var_81.one_of) == 2
    assert module_2.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_82 = module_0.to_json_schema(var_81)
    var_83 = 'oneOf'
    var_84 = var_82[var_83]
    var_85 = len(var_84)
    assert var_85 == 2
    var_86 = module_1.String()
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
    assert var_86.pattern is None
    assert var_86.pattern_regex is None
    var_87 = module_1.String(min_length=var_17)
    assert f'{type(var_87).__module__}.{type(var_87).__qualname__}' == 'typesystem.fields.String'
    assert var_87.title == ''
    assert var_87.description == ''
    assert var_87.allow_null is False
    assert var_87.read_only is False
    assert var_87.allow_blank is False
    assert var_87.trim_whitespace is True
    assert var_87.max_length is None
    assert var_87.min_length == 5
    assert var_87.format is None
    assert var_87.coerce_types is True
    assert var_87.pattern is None
    assert var_87.pattern_regex is None
    var_88 = [var_86, var_87]
    var_89 = module_2.AllOf(var_88)
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_89.title == ''
    assert var_89.description == ''
    assert var_89.allow_null is False
    assert var_89.read_only is False
    assert f'{type(var_89.all_of).__module__}.{type(var_89.all_of).__qualname__}' == 'builtins.list'
    assert len(var_89.all_of) == 2
    var_90 = module_0.to_json_schema(var_89)
    var_91 = 'allOf'
    var_92 = var_90[var_91]
    var_93 = len(var_92)
    assert var_93 == 2
    var_94 = module_1.String()
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
    var_95 = module_1.Integer()
    assert f'{type(var_95).__module__}.{type(var_95).__qualname__}' == 'typesystem.fields.Integer'
    assert var_95.title == ''
    assert var_95.description == ''
    assert var_95.allow_null is False
    assert var_95.read_only is False
    assert var_95.minimum is None
    assert var_95.maximum is None
    assert var_95.exclusive_minimum is None
    assert var_95.exclusive_maximum is None
    assert var_95.multiple_of is None
    assert var_95.precision is None
    assert var_95.coerce_types is True
    var_96 = module_1.Boolean()
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_96.title == ''
    assert var_96.description == ''
    assert var_96.allow_null is False
    assert var_96.read_only is False
    assert var_96.coerce_types is True
    var_97 = module_2.IfThenElse(var_94, var_95, var_96)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_97.title == ''
    assert var_97.description == ''
    assert var_97.allow_null is False
    assert var_97.read_only is False
    assert f'{type(var_97.if_clause).__module__}.{type(var_97.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_97.then_clause).__module__}.{type(var_97.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_97.else_clause).__module__}.{type(var_97.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_98 = module_0.to_json_schema(var_97)
    var_99 = module_1.String()
    assert f'{type(var_99).__module__}.{type(var_99).__qualname__}' == 'typesystem.fields.String'
    assert var_99.title == ''
    assert var_99.description == ''
    assert var_99.allow_null is False
    assert var_99.read_only is False
    assert var_99.allow_blank is False
    assert var_99.trim_whitespace is True
    assert var_99.max_length is None
    assert var_99.min_length is None
    assert var_99.format is None
    assert var_99.coerce_types is True
    assert var_99.pattern is None
    assert var_99.pattern_regex is None
    var_100 = module_1.Integer()
    assert f'{type(var_100).__module__}.{type(var_100).__qualname__}' == 'typesystem.fields.Integer'
    assert var_100.title == ''
    assert var_100.description == ''
    assert var_100.allow_null is False
    assert var_100.read_only is False
    assert var_100.minimum is None
    assert var_100.maximum is None
    assert var_100.exclusive_minimum is None
    assert var_100.exclusive_maximum is None
    assert var_100.multiple_of is None
    assert var_100.precision is None
    assert var_100.coerce_types is True
    var_101 = module_2.IfThenElse(var_99, var_100)
    assert f'{type(var_101).__module__}.{type(var_101).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_101.title == ''
    assert var_101.description == ''
    assert var_101.allow_null is False
    assert var_101.read_only is False
    assert f'{type(var_101.if_clause).__module__}.{type(var_101.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_101.then_clause).__module__}.{type(var_101.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_101.else_clause).__module__}.{type(var_101.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_102 = module_0.to_json_schema(var_101)
    var_103 = module_2.Not(var_70)
    assert f'{type(var_103).__module__}.{type(var_103).__qualname__}' == 'typesystem.composites.Not'
    assert var_103.title == ''
    assert var_103.description == ''
    assert var_103.allow_null is False
    assert var_103.read_only is False
    assert f'{type(var_103.negated).__module__}.{type(var_103.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_2.Not.errors == {'negated': 'Must not match.'}
    var_104 = module_0.to_json_schema(var_103)
    var_105 = module_3.Definitions()
    assert f'{type(var_105).__module__}.{type(var_105).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_105) == 0
    var_106 = module_0.to_json_schema(var_105)
    var_107 = 'CustomType'
    var_108 = module_3.Definitions()
    assert f'{type(var_108).__module__}.{type(var_108).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_108) == 0
    var_109 = module_3.Reference(var_107, var_108)
    assert f'{type(var_109).__module__}.{type(var_109).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_109.title == ''
    assert var_109.description == ''
    assert var_109.allow_null is False
    assert var_109.read_only is False
    assert var_109.to == 'CustomType'
    assert f'{type(var_109.definitions).__module__}.{type(var_109.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_109.definitions) == 0
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    module_0.to_json_schema(var_109)