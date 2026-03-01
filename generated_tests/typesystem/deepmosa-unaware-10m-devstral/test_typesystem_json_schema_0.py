# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import enum as module_1
import typesystem.fields as module_2
import typesystem.composites as module_3
import re as module_4
import typesystem.schemas as module_5

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_1 = module_0.get_valid_types(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_0 = None
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    module_0.not_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

def test_case_13():
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_14():
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_2 = module_3.IfThenElse(var_1, else_clause=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = module_0.to_json_schema(var_2)

def test_case_15():
    var_0 = module_2.Boolean()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_1 = module_0.to_json_schema(var_0, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_3 = module_0.to_json_schema(var_2, var_0)
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

def test_case_17():
    var_0 = module_2.Boolean()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    with pytest.raises(ValueError):
        module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = {var_0: var_0}
    var_2 = module_1._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = module_3.AllOf(var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.all_of == {None: None}
    module_0.to_json_schema(var_3)

def test_case_20():
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_0 = 'email'
    var_1 = module_2.String(format=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length is None
    assert var_1.min_length is None
    assert var_1.format == 'email'
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_0.to_json_schema(var_1, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_22():
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
def test_case_24():
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_3.Not(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.negated is None
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_2.Boolean()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_0.get_standard_properties(var_3)
    module_0.to_json_schema(var_2)

def test_case_25():
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
    var_2 = module_3.IfThenElse(var_1, else_clause=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.fields.Const'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Const'
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_26():
    var_0 = module_2.Boolean()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_3 = module_0.type_from_json_schema(var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.coerce_types is False

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_2.Integer()
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
    var_2 = module_3.IfThenElse(var_1, else_clause=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = module_0.to_json_schema(var_2, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_0.to_json_schema(var_0)

def test_case_28():
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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

def test_case_29():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_5.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.fields).__module__}.{type(var_2.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.fields) == 0
    assert var_2.required == []
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_3.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.schemas.Schema'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_4 = module_0.from_json_schema(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_3.IfThenElse(var_3, else_clause=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.if_clause).__module__}.{type(var_5.if_clause).__qualname__}' == 'typesystem.composites.Not'
    assert f'{type(var_5.then_clause).__module__}.{type(var_5.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_5.else_clause).__module__}.{type(var_5.else_clause).__qualname__}' == 'typesystem.schemas.Schema'
    var_6 = module_0.to_json_schema(var_5)
    var_7 = module_0.from_json_schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.if_clause).__module__}.{type(var_7.if_clause).__qualname__}' == 'typesystem.composites.Not'
    assert f'{type(var_7.then_clause).__module__}.{type(var_7.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_7.else_clause).__module__}.{type(var_7.else_clause).__qualname__}' == 'typesystem.fields.Object'
    var_8 = module_0.from_json_schema(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Any'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    var_9 = var_4.validate_or_error(var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None

def test_case_32():
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
    var_2 = module_3.OneOf(var_0, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.one_of).__module__}.{type(var_2.one_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.one_of) == 0
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_3 = module_0.from_json_schema(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_3.NeverMatch()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_5 = module_3.IfThenElse(var_2, else_clause=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.if_clause).__module__}.{type(var_5.if_clause).__qualname__}' == 'typesystem.composites.OneOf'
    assert f'{type(var_5.then_clause).__module__}.{type(var_5.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_5.else_clause).__module__}.{type(var_5.else_clause).__qualname__}' == 'typesystem.schemas.Schema'
    var_6 = module_3.IfThenElse(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.if_clause).__module__}.{type(var_6.if_clause).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert f'{type(var_6.then_clause).__module__}.{type(var_6.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_6.else_clause).__module__}.{type(var_6.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_7 = module_0.to_json_schema(var_5)

def test_case_33():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 10
    var_4 = False
    var_5 = module_5.Definitions()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 0
    var_6 = 'integer'
    var_7 = {var_0: var_6, var_1: var_4, var_2: var_3}
    var_8 = False
    var_9 = module_5.Definitions()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_9) == 0
    var_10 = module_0.from_json_schema_type(var_7, var_6, var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Integer'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.minimum is False
    assert var_10.maximum == 10
    assert var_10.exclusive_minimum is None
    assert var_10.exclusive_maximum is None
    assert var_10.multiple_of is None
    assert var_10.precision is None
    assert var_10.coerce_types is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_11 = 'minLength'
    var_12 = 'maxLength'
    var_13 = 'string'
    var_14 = 5
    var_15 = {var_0: var_13, var_11: var_14, var_12: var_3}
    var_16 = False
    var_17 = module_5.Definitions()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_17) == 0
    var_18 = module_0.from_json_schema_type(var_15, var_13, var_16, var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.String'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.allow_blank is False
    assert var_18.trim_whitespace is True
    assert var_18.max_length == 10
    assert var_18.min_length == 5
    assert var_18.format is None
    assert var_18.coerce_types is False
    assert var_18.pattern is None
    assert var_18.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_19 = 'boolean'
    var_20 = {var_0: var_19}
    var_21 = False
    var_22 = module_5.Definitions()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_22) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_23 = module_0.from_json_schema_type(var_20, var_19, var_21, var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert var_23.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_24 = 'items'
    var_25 = 'array'
    var_26 = {var_0: var_13}
    var_27 = {var_0: var_25, var_24: var_26}
    var_28 = False
    var_29 = module_5.Definitions()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_29) == 0
    var_30 = module_0.from_json_schema_type(var_27, var_25, var_28, var_29)
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
    var_31 = var_30.items
    var_32 = 'properties'
    var_33 = 'object'
    var_34 = 'name'
    var_35 = {var_0: var_13}
    var_36 = {var_34: var_35}
    var_37 = {var_0: var_33, var_32: var_36}
    var_38 = False
    var_39 = module_5.Definitions()
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_39) == 0
    var_40 = module_0.from_json_schema_type(var_37, var_33, var_38, var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Object'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert f'{type(var_40.properties).__module__}.{type(var_40.properties).__qualname__}' == 'builtins.dict'
    assert len(var_40.properties) == 1
    assert var_40.pattern_properties == {}
    assert var_40.additional_properties is None
    assert var_40.property_names is None
    assert var_40.min_properties is None
    assert var_40.max_properties is None
    assert var_40.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_41 = var_40.properties[var_34]

def test_case_34():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = '#/components/schemas/TestSchema'
    var_3 = {var_1: var_2}
    var_4 = module_0.ref_from_json_schema(var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == '#/components/schemas/TestSchema'
    assert f'{type(var_4.definitions).__module__}.{type(var_4.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4.definitions) == 0
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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

def test_case_35():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
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
    with pytest.raises(AttributeError):
        var_34 = var_33.schemas

def test_case_36():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
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
    var_31 = 'value'
    var_32 = {var_30: var_31}
    var_33 = module_0.from_json_schema(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Const'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.const == 'value'
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
    with pytest.raises(AttributeError):
        var_42 = var_41.constraints

def test_case_37():
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_5.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.fields).__module__}.{type(var_2.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.fields) == 0
    assert var_2.required == []
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_3.AllOf(var_1, **var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.all_of).__module__}.{type(var_3.all_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_3.all_of) == 0
    var_4 = module_3.IfThenElse(var_2, else_clause=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.if_clause).__module__}.{type(var_4.if_clause).__qualname__}' == 'typesystem.schemas.Schema'
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.composites.AllOf'
    var_5 = var_3.validate(var_0)
    var_6 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_7 = module_0.from_json_schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.if_clause).__module__}.{type(var_7.if_clause).__qualname__}' == 'typesystem.fields.Object'
    assert f'{type(var_7.then_clause).__module__}.{type(var_7.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_7.else_clause).__module__}.{type(var_7.else_clause).__qualname__}' == 'typesystem.composites.AllOf'

def test_case_38():
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
    var_1 = 'oneOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'number'
    var_5 = {var_3: var_4}
    var_6 = [var_5, var_5]
    var_7 = 'default_value'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = module_0.one_of_from_json_schema(var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_9.default == 'default_value'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.one_of).__module__}.{type(var_9.one_of).__qualname__}' == 'builtins.list'
    assert len(var_9.one_of) == 2
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    with pytest.raises(TypeError):
        var_12 = var_9.one_of[var_2]

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_5.Schema(var_1, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.fields).__module__}.{type(var_2.fields).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.fields) == 0
    assert var_2.required == []
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_3.OneOf(var_1, **var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.one_of).__module__}.{type(var_3.one_of).__qualname__}' == 'enum._EnumDict'
    assert len(var_3.one_of) == 0
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_4 = module_0.from_json_schema(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_5 = module_3.NeverMatch()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_6 = module_3.IfThenElse(var_3, else_clause=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.if_clause).__module__}.{type(var_6.if_clause).__qualname__}' == 'typesystem.composites.OneOf'
    assert f'{type(var_6.then_clause).__module__}.{type(var_6.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_6.else_clause).__module__}.{type(var_6.else_clause).__qualname__}' == 'typesystem.schemas.Schema'
    var_7 = module_0.to_json_schema(var_6)
    var_8 = module_0.from_json_schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.if_clause).__module__}.{type(var_8.if_clause).__qualname__}' == 'typesystem.composites.OneOf'
    assert f'{type(var_8.then_clause).__module__}.{type(var_8.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_8.else_clause).__module__}.{type(var_8.else_clause).__qualname__}' == 'typesystem.fields.Object'
    module_0.to_json_schema(var_0, var_0)

def test_case_40():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
    var_4 = module_5.Definitions()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = 'type'
    var_6 = 'string'
    var_7 = '$ref'
    var_8 = '#/components/schemas/Test'
    var_9 = {var_7: var_8}
    var_10 = module_0.from_json_schema(var_9, var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.to == '#/components/schemas/Test'
    assert f'{type(var_10.definitions).__module__}.{type(var_10.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_10.definitions) == 0
    assert module_5.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_5.Reference.target).__module__}.{type(module_5.Reference.target).__qualname__}' == 'builtins.property'
    var_11 = {var_5: var_6}
    var_12 = module_0.from_json_schema(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.String'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.default == ''
    assert var_12.allow_blank is True
    assert var_12.trim_whitespace is True
    assert var_12.max_length is None
    assert var_12.min_length is None
    assert var_12.format is None
    assert var_12.coerce_types is False
    assert var_12.pattern is None
    assert var_12.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_13 = 'integer'
    var_14 = {var_5: var_13}
    var_15 = module_0.from_json_schema(var_14)
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
    assert var_15.coerce_types is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_16 = 'number'
    var_17 = {var_5: var_16}
    var_18 = module_0.from_json_schema(var_17)
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
    var_20 = {var_5: var_19}
    var_21 = module_0.from_json_schema(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert var_21.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'none', 'null'}
    var_22 = 'array'
    var_23 = {var_5: var_22}
    var_24 = module_0.from_json_schema(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Array'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert var_24.items is None
    assert var_24.additional_items is True
    assert var_24.min_items == 0
    assert var_24.max_items is None
    assert var_24.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_25 = 'object'
    var_26 = {var_5: var_25}
    var_27 = module_0.from_json_schema(var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.fields.Object'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.properties == {}
    assert var_27.pattern_properties == {}
    assert var_27.additional_properties is None
    assert var_27.property_names is None
    assert var_27.min_properties is None
    assert var_27.max_properties is None
    assert var_27.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_28 = 'enum'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = 'c'
    var_32 = [var_29, var_30, var_31]
    var_33 = {var_28: var_32}
    var_34 = module_0.from_json_schema(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Choice'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_34.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_35 = module_0.from_json_schema(var_33)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Choice'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_35.coerce_types is True
    var_36 = var_35.choices
    var_37 = 'const'
    var_38 = 'test'
    var_39 = {var_37: var_38}
    var_40 = module_0.from_json_schema(var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Const'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert var_40.const == 'test'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_41 = module_0.from_json_schema(var_39)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Const'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.const == 'test'
    with pytest.raises(AttributeError):
        var_42 = var_41.value
    assert var_42 == 'test'

def test_case_41():
    var_0 = 9
    var_1 = 'email'
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == 9
    assert var_2.min_length == 9
    assert var_2.format == 'email'
    assert var_2.coerce_types is True
    assert var_2.pattern == 'email'
    assert f'{type(var_2.pattern_regex).__module__}.{type(var_2.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.String'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.allow_blank is False
    assert var_4.trim_whitespace is True
    assert var_4.max_length == 9
    assert var_4.min_length == 9
    assert var_4.format == 'email'
    assert var_4.coerce_types is False
    assert var_4.pattern == 'email'
    assert f'{type(var_4.pattern_regex).__module__}.{type(var_4.pattern_regex).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_4.purge()
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
    var_3 = module_0.type_from_json_schema(var_1, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Union'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert f'{type(var_3.any_of).__module__}.{type(var_3.any_of).__qualname__}' == 'builtins.list'
    assert len(var_3.any_of) == 5
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_4 = module_0.from_json_schema(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_5 = module_3.IfThenElse(var_3, else_clause=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.if_clause).__module__}.{type(var_5.if_clause).__qualname__}' == 'typesystem.fields.Union'
    assert f'{type(var_5.then_clause).__module__}.{type(var_5.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_5.else_clause).__module__}.{type(var_5.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_6 = module_3.IfThenElse(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.if_clause).__module__}.{type(var_6.if_clause).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert f'{type(var_6.then_clause).__module__}.{type(var_6.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_6.else_clause).__module__}.{type(var_6.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_7 = module_0.to_json_schema(var_5)
    var_8 = module_0.to_json_schema(var_4)
    assert var_8 is True
    var_9 = module_5.Definitions()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_9) == 0
    var_10 = var_9.__setitem__(var_0, var_0)
    assert len(var_9) == 1
    module_0.to_json_schema(var_9)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = module_5.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'minLength'
    var_8 = 8
    var_9 = {var_4: var_5, var_7: var_8}
    var_10 = 'number'
    var_11 = {var_4: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = module_0.if_then_else_from_json_schema(var_12, var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.if_clause).__module__}.{type(var_13.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_13.then_clause).__module__}.{type(var_13.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_13.else_clause).__module__}.{type(var_13.else_clause).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_17 = {var_4: var_5}
    var_18 = {var_4: var_5, var_2: var_10, var_7: var_8}
    var_19 = {var_1: var_17, var_2: var_18}
    var_20 = module_0.if_then_else_from_json_schema(var_19, var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert f'{type(var_20.if_clause).__module__}.{type(var_20.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_20.then_clause).__module__}.{type(var_20.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_20.else_clause).__module__}.{type(var_20.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_21 = var_20.if_clause
    var_22 = var_20.then_clause
    var_23 = {var_4: var_5}
    var_24 = {var_1: var_23}
    var_25 = module_0.if_then_else_from_json_schema(var_24, var_0)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert f'{type(var_25.if_clause).__module__}.{type(var_25.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_25.then_clause).__module__}.{type(var_25.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_25.else_clause).__module__}.{type(var_25.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_26 = var_25.if_clause
    var_22.validate(var_26)

@pytest.mark.xfail(strict=True)
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_11 = True
    var_12 = module_2.Integer(minimum=var_10, maximum=var_11, exclusive_maximum=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Integer'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.minimum == 0
    assert var_12.maximum is True
    assert var_12.exclusive_minimum is None
    assert var_12.exclusive_maximum is True
    assert var_12.multiple_of is None
    assert var_12.precision is None
    assert var_12.coerce_types is True
    var_13 = True
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
    var_21 = False
    var_22 = module_2.Array(var_19, min_items=var_13, max_items=var_20, unique_items=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Array'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.items).__module__}.{type(var_22.items).__qualname__}' == 'typesystem.fields.String'
    assert var_22.additional_items is False
    assert var_22.min_items is True
    assert var_22.max_items == 5
    assert var_22.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_23 = module_0.to_json_schema(var_22)
    var_24 = 'name'
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
    var_26 = {var_24: var_25}
    var_27 = [var_24]
    var_28 = module_2.Object(properties=var_26, required=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Object'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.properties).__module__}.{type(var_28.properties).__qualname__}' == 'builtins.dict'
    assert len(var_28.properties) == 1
    assert var_28.pattern_properties == {}
    assert var_28.additional_properties is True
    assert var_28.property_names is None
    assert var_28.min_properties is None
    assert var_28.max_properties is None
    assert var_28.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = module_4.purge()
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
    var_31 = module_2.Choice(choices=var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Choice'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.choices == []
    assert var_31.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_32 = 'fixed'
    var_33 = module_2.Const(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Const'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.const == 'fixed'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_34 = module_0.to_json_schema(var_33)
    var_35 = module_2.String()
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
    var_36 = module_2.Integer()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Integer'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.minimum is None
    assert var_36.maximum is None
    assert var_36.exclusive_minimum is None
    assert var_36.exclusive_maximum is None
    assert var_36.multiple_of is None
    assert var_36.precision is None
    assert var_36.coerce_types is True
    var_30.validate(var_30)

def test_case_45():
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = module_0.to_json_schema(var_6)
    var_8 = 0
    var_9 = 100
    var_10 = True
    var_11 = module_2.Integer(minimum=var_8, maximum=var_9, exclusive_maximum=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Integer'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.minimum == 0
    assert var_11.maximum == 100
    assert var_11.exclusive_minimum is None
    assert var_11.exclusive_maximum is True
    assert var_11.multiple_of is None
    assert var_11.precision is None
    assert var_11.coerce_types is True
    var_12 = True
    var_13 = module_0.to_json_schema(var_11)
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
    var_20 = True
    var_21 = module_2.Array(var_19, min_items=var_12, max_items=var_1, unique_items=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Array'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.items).__module__}.{type(var_21.items).__qualname__}' == 'typesystem.fields.String'
    assert var_21.additional_items is False
    assert var_21.min_items is True
    assert var_21.max_items is False
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
    var_38 = 'test'
    var_39 = module_2.Const(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Const'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert var_39.const == 'test'
    var_40 = [var_37, var_39]
    var_41 = module_3.AllOf(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.all_of).__module__}.{type(var_41.all_of).__qualname__}' == 'builtins.list'
    assert len(var_41.all_of) == 2
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
    var_44 = module_2.Integer()
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
    var_45 = [var_43, var_44]
    var_46 = module_3.OneOf(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.one_of).__module__}.{type(var_46.one_of).__qualname__}' == 'builtins.list'
    assert len(var_46.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
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
    var_49 = module_3.Not(var_48)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.composites.Not'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert f'{type(var_49.negated).__module__}.{type(var_49.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
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
    var_53 = module_3.IfThenElse(var_51, var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert f'{type(var_53.if_clause).__module__}.{type(var_53.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_53.then_clause).__module__}.{type(var_53.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_53.else_clause).__module__}.{type(var_53.else_clause).__qualname__}' == 'typesystem.fields.Any'
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
    var_56 = module_0.to_json_schema(var_53)
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
    var_58 = {var_23: var_57}
    var_59 = [var_23]
    var_60 = module_5.Schema(var_58)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert f'{type(var_60.fields).__module__}.{type(var_60.fields).__qualname__}' == 'builtins.dict'
    assert len(var_60.fields) == 1
    assert var_60.required == ['name']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_61 = module_0.to_json_schema(var_60)

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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_14 = 0.1
    var_15 = module_2.Float(minimum=var_10, maximum=var_4, multiple_of=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Float'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.minimum == 0
    assert var_15.maximum == 1
    assert var_15.exclusive_minimum is None
    assert var_15.exclusive_maximum is None
    assert var_15.multiple_of == pytest.approx(0.1, abs=0.01, rel=0.01)
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
    var_21 = module_2.Array(var_19, min_items=var_4, max_items=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Array'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    assert f'{type(var_21.items).__module__}.{type(var_21.items).__qualname__}' == 'typesystem.fields.String'
    assert var_21.additional_items is False
    assert var_21.min_items == 1
    assert var_21.max_items == 5
    assert var_21.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_22 = module_0.to_json_schema(var_21)
    var_23 = 'name'
    var_24 = 'age'
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
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = module_2.Object(properties=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Object'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.properties).__module__}.{type(var_28.properties).__qualname__}' == 'builtins.dict'
    assert len(var_28.properties) == 2
    assert var_28.pattern_properties == {}
    assert var_28.additional_properties is True
    assert var_28.property_names is None
    assert var_28.min_properties is None
    assert var_28.max_properties is None
    assert var_28.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = 'a'
    var_31 = (var_30, var_30)
    var_32 = 'b'
    var_33 = (var_32, var_32)
    var_34 = [var_31, var_33]
    var_35 = module_2.Choice(choices=var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Choice'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.choices == [('a', 'a'), ('b', 'b')]
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
    var_45 = module_2.String()
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.String'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.allow_blank is False
    assert var_45.trim_whitespace is True
    assert var_45.max_length is None
    assert var_45.min_length is None
    assert var_45.format is None
    assert var_45.coerce_types is True
    assert var_45.pattern is None
    assert var_45.pattern_regex is None
    var_46 = 'test'
    var_47 = module_2.Const(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.Const'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert var_47.const == 'test'
    var_48 = [var_45, var_47]
    var_49 = module_3.AllOf(var_48)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_49.title == ''
    assert var_49.description == ''
    assert var_49.allow_null is False
    assert var_49.read_only is False
    assert f'{type(var_49.all_of).__module__}.{type(var_49.all_of).__qualname__}' == 'builtins.list'
    assert len(var_49.all_of) == 2
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
    var_54 = module_3.OneOf(var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert f'{type(var_54.one_of).__module__}.{type(var_54.one_of).__qualname__}' == 'builtins.list'
    assert len(var_54.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
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
    var_57 = module_3.Not(var_56)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.composites.Not'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert f'{type(var_57.negated).__module__}.{type(var_57.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_58 = module_0.to_json_schema(var_57)
    var_59 = module_2.String()
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
    var_60 = module_2.Integer()
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
    var_61 = module_3.IfThenElse(var_59, var_60)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert f'{type(var_61.if_clause).__module__}.{type(var_61.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_61.then_clause).__module__}.{type(var_61.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_61.else_clause).__module__}.{type(var_61.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_62 = module_0.to_json_schema(var_61)
    var_63 = module_5.Definitions()
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_63) == 0
    var_64 = module_5.Reference(var_46, var_63)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert var_64.to == 'test'
    assert f'{type(var_64.definitions).__module__}.{type(var_64.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_64.definitions) == 0
    assert module_5.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_5.Reference.target).__module__}.{type(module_5.Reference.target).__qualname__}' == 'builtins.property'
    module_0.to_json_schema(var_64)

def test_case_47():
    var_0 = module_2.String()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = True
    var_2 = module_2.Array(var_0, min_items=var_1, max_items=var_1, unique_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.String'
    assert var_2.additional_items is False
    assert var_2.min_items is True
    assert var_2.max_items is True
    assert var_2.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = module_0.type_from_json_schema(var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.items).__module__}.{type(var_4.items).__qualname__}' == 'typesystem.fields.String'
    assert var_4.additional_items is False
    assert var_4.min_items is True
    assert var_4.max_items is True
    assert var_4.unique_items is True

def test_case_48():
    var_0 = 9
    var_1 = 'email'
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == 9
    assert var_2.min_length == 9
    assert var_2.format == 'email'
    assert var_2.coerce_types is True
    assert var_2.pattern == 'email'
    assert f'{type(var_2.pattern_regex).__module__}.{type(var_2.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_3 = 1184.1403821259419
    var_4 = module_2.Float(multiple_of=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Float'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum is None
    assert var_4.maximum is None
    assert var_4.exclusive_minimum is None
    assert var_4.exclusive_maximum is None
    assert var_4.multiple_of == pytest.approx(1184.1403821259419, abs=0.01, rel=0.01)
    assert var_4.precision is None
    assert var_4.coerce_types is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
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
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_7 = True
    var_8 = module_2.Array(var_6, min_items=var_2, max_items=var_2, unique_items=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Array'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.items).__module__}.{type(var_8.items).__qualname__}' == 'typesystem.fields.String'
    assert var_8.additional_items is False
    assert var_8.min_items is True
    assert var_8.max_items is True
    assert var_8.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_9 = module_0.to_json_schema(var_8)

def test_case_50():
    var_0 = module_2.String()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = True
    var_2 = module_2.Array(var_0, min_items=var_1, max_items=var_1, unique_items=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Array'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.items).__module__}.{type(var_2.items).__qualname__}' == 'typesystem.fields.String'
    assert var_2.additional_items is False
    assert var_2.min_items is True
    assert var_2.max_items is True
    assert var_2.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
def test_case_51():
    var_0 = 'type'
    var_1 = 'maximum'
    var_2 = 'multipleOf'
    var_3 = 'default'
    var_4 = 'number'
    var_5 = 100
    var_6 = module_5.Definitions()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 0
    var_7 = 'minLength'
    var_8 = 'pattern'
    var_9 = 'string'
    var_10 = 1
    var_11 = 'email'
    var_12 = '^[a-zA-Z0-9]+$'
    var_13 = {var_0: var_9, var_7: var_10, var_8: var_5, var_12: var_11, var_8: var_12, var_3: var_12}
    var_14 = False
    var_15 = module_0.from_json_schema_type(var_13, var_9, var_14, var_6)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.String'
    assert var_15.default == '^[a-zA-Z0-9]+$'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.allow_blank is False
    assert var_15.trim_whitespace is True
    assert var_15.max_length is None
    assert var_15.min_length is None
    assert var_15.format is None
    assert var_15.coerce_types is False
    assert var_15.pattern == '^[a-zA-Z0-9]+$'
    assert f'{type(var_15.pattern_regex).__module__}.{type(var_15.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_16 = True
    var_17 = module_5.Definitions()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_17) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_18 = 'items'
    var_19 = 'additionalItems'
    var_20 = 'minItems'
    var_21 = 'uniqueItems'
    var_22 = 'array'
    var_23 = {var_0: var_9}
    var_24 = False
    var_25 = 10
    var_26 = True
    var_27 = [var_1]
    var_28 = {var_0: var_22, var_18: var_23, var_19: var_24, var_20: var_16, var_10: var_25, var_21: var_26, var_3: var_27}
    var_29 = False
    var_30 = module_0.from_json_schema_type(var_28, var_22, var_29, var_6)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Array'
    assert var_30.default == ['maximum']
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert f'{type(var_30.items).__module__}.{type(var_30.items).__qualname__}' == 'typesystem.fields.String'
    assert var_30.additional_items is False
    assert var_30.min_items is True
    assert var_30.max_items is None
    assert var_30.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_31 = var_30.items
    var_32 = 'patternProperties'
    var_33 = 'additionalProperties'
    var_34 = 'propertyNames'
    var_35 = 'minProperties'
    var_36 = 'object'
    var_37 = {var_0: var_9}
    var_38 = {var_4: var_37}
    var_39 = False
    var_40 = [var_4]
    var_41 = {var_2: var_22}
    var_42 = {var_0: var_36, var_0: var_38, var_32: var_38, var_33: var_39, var_34: var_31, var_35: var_26, var_0: var_25, var_0: var_40, var_3: var_41}
    var_43 = False
    module_0.from_json_schema_type(var_42, var_36, var_43, var_26)

def test_case_52():
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
    var_1 = '$ref'
    var_2 = {var_1: var_1}
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_53():
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_9 = 'maxLength'
    var_10 = module_0.to_json_schema(var_8)
    var_11 = False
    var_12 = 100
    var_13 = -22
    var_14 = module_2.Integer(minimum=var_11, maximum=var_12, exclusive_minimum=var_4, multiple_of=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum is False
    assert var_14.maximum == 100
    assert var_14.exclusive_minimum is True
    assert var_14.exclusive_maximum is None
    assert var_14.multiple_of == -22
    assert var_14.precision is None
    assert var_14.coerce_types is True
    var_15 = module_0.to_json_schema(var_14)
    var_16 = module_2.Float(minimum=var_11, maximum=var_4)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Float'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum is False
    assert var_16.maximum is True
    assert var_16.exclusive_minimum is None
    assert var_16.exclusive_maximum is None
    assert var_16.multiple_of is None
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
    var_20 = 5
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
    var_22 = module_2.Array(var_21, var_11, var_4, var_20, unique_items=var_4)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Array'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.items).__module__}.{type(var_22.items).__qualname__}' == 'typesystem.fields.String'
    assert var_22.additional_items is False
    assert var_22.min_items is True
    assert var_22.max_items == 5
    assert var_22.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_23 = module_0.to_json_schema(var_22)
    var_24 = 'name'
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
    var_26 = {var_24: var_25}
    var_27 = '^S_'
    var_28 = module_2.String()
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
    var_29 = {var_27: var_28}
    var_30 = '[A-Z]+'
    var_31 = module_2.String(pattern=var_30)
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
    assert var_31.pattern == '[A-Z]+'
    assert f'{type(var_31.pattern_regex).__module__}.{type(var_31.pattern_regex).__qualname__}' == 're.Pattern'
    var_32 = [var_24]
    var_33 = module_2.Object(properties=var_26, pattern_properties=var_29, additional_properties=var_11, property_names=var_31, min_properties=var_4, max_properties=var_20, required=var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Object'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.properties).__module__}.{type(var_33.properties).__qualname__}' == 'builtins.dict'
    assert len(var_33.properties) == 1
    assert f'{type(var_33.pattern_properties).__module__}.{type(var_33.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_33.pattern_properties) == 1
    assert var_33.additional_properties is False
    assert f'{type(var_33.property_names).__module__}.{type(var_33.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_33.min_properties is True
    assert var_33.max_properties == 5
    assert var_33.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_34 = module_0.to_json_schema(var_33)
    var_35 = 'a'
    var_36 = (var_35, var_35)
    var_37 = 'b'
    var_38 = (var_37, var_37)
    var_39 = [var_36, var_38]
    var_40 = module_2.Choice(choices=var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Choice'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert var_40.choices == [('a', 'a'), ('b', 'b')]
    assert var_40.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_41 = module_0.to_json_schema(var_40)
    var_42 = 'fixed'
    var_43 = module_2.Const(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Const'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.const == 'fixed'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_44 = module_0.to_json_schema(var_43)
    var_45 = module_2.String()
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.String'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert var_45.allow_blank is False
    assert var_45.trim_whitespace is True
    assert var_45.max_length is None
    assert var_45.min_length is None
    assert var_45.format is None
    assert var_45.coerce_types is True
    assert var_45.pattern is None
    assert var_45.pattern_regex is None
    var_46 = module_2.Integer()
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Integer'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert var_46.minimum is None
    assert var_46.maximum is None
    assert var_46.exclusive_minimum is None
    assert var_46.exclusive_maximum is None
    assert var_46.multiple_of is None
    assert var_46.precision is None
    assert var_46.coerce_types is True
    var_47 = [var_45, var_46]
    var_48 = module_2.Union(var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.fields.Union'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert f'{type(var_48.any_of).__module__}.{type(var_48.any_of).__qualname__}' == 'builtins.list'
    assert len(var_48.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
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
    var_51 = module_2.Integer()
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
    var_53 = 'test'
    var_54 = module_2.Const(var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Const'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert var_54.const == 'test'
    var_55 = [var_52, var_54]
    var_56 = module_3.AllOf(var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.all_of).__module__}.{type(var_56.all_of).__qualname__}' == 'builtins.list'
    assert len(var_56.all_of) == 2
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
    var_59 = module_2.Integer()
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
    var_60 = module_2.Boolean()
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.coerce_types is True
    var_61 = module_3.IfThenElse(var_58, var_59, var_60)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert f'{type(var_61.if_clause).__module__}.{type(var_61.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_61.then_clause).__module__}.{type(var_61.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_61.else_clause).__module__}.{type(var_61.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_62 = module_1._EnumDict()
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'enum._EnumDict'
    assert len(var_62) == 0
    var_63 = module_0.to_json_schema(var_61)
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
    var_65 = module_3.Not(var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.composites.Not'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert f'{type(var_65.negated).__module__}.{type(var_65.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_66 = module_0.to_json_schema(var_65)
    var_67 = module_2.String()
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
    var_69 = {var_24: var_68}
    var_70 = [var_24]
    var_71 = module_5.Schema(var_69)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_71.title == ''
    assert var_71.description == ''
    assert var_71.allow_null is False
    assert var_71.read_only is False
    assert f'{type(var_71.fields).__module__}.{type(var_71.fields).__qualname__}' == 'builtins.dict'
    assert len(var_71.fields) == 1
    assert var_71.required == ['name']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_72 = module_0.to_json_schema(var_71)
    var_56.validate(var_9)

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
    var_1 = module_0.to_json_schema(var_0)
    assert var_1 is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_9 = 'format'
    var_10 = module_0.to_json_schema(var_8)
    var_11 = False
    var_12 = 100
    var_13 = 2
    var_14 = module_2.Integer(minimum=var_11, maximum=var_12, multiple_of=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum is False
    assert var_14.maximum == 100
    assert var_14.exclusive_minimum is None
    assert var_14.exclusive_maximum is None
    assert var_14.multiple_of == 2
    assert var_14.precision is None
    assert var_14.coerce_types is True
    var_15 = module_0.to_json_schema(var_14)
    var_16 = module_2.Float(exclusive_minimum=var_11, exclusive_maximum=var_4)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Float'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.minimum is None
    assert var_16.maximum is None
    assert var_16.exclusive_minimum is False
    assert var_16.exclusive_maximum is True
    assert var_16.multiple_of is None
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
    var_20 = 5
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
    var_22 = module_2.Array(var_21, var_11, var_4, var_20)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Array'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    assert f'{type(var_22.items).__module__}.{type(var_22.items).__qualname__}' == 'typesystem.fields.String'
    assert var_22.additional_items is False
    assert var_22.min_items is True
    assert var_22.max_items == 5
    assert var_22.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_23 = module_0.to_json_schema(var_22)
    var_24 = 'name'
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
    var_26 = {var_24: var_25}
    var_27 = [var_24]
    var_28 = module_2.Object(properties=var_26, additional_properties=var_11, required=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Object'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.properties).__module__}.{type(var_28.properties).__qualname__}' == 'builtins.dict'
    assert len(var_28.properties) == 1
    assert var_28.pattern_properties == {}
    assert var_28.additional_properties is False
    assert var_28.property_names is None
    assert var_28.min_properties is None
    assert var_28.max_properties is None
    assert var_28.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_29 = module_0.to_json_schema(var_28)
    var_30 = 'a'
    var_31 = (var_30, var_30)
    var_32 = 'b'
    var_33 = (var_32, var_32)
    var_34 = [var_31, var_33]
    var_35 = module_2.Choice(choices=var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.fields.Choice'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert var_35.choices == [('a', 'a'), ('b', 'b')]
    assert var_35.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_36 = module_0.to_json_schema(var_35)
    var_37 = module_2.Const(var_9)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Const'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.const == 'format'
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
    var_48 = module_3.AllOf(var_47)
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
    var_51 = module_2.Integer()
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
    var_54 = module_2.Boolean()
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert var_54.coerce_types is True
    var_55 = module_3.IfThenElse(var_52, var_53, var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert f'{type(var_55.if_clause).__module__}.{type(var_55.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_55.then_clause).__module__}.{type(var_55.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_55.else_clause).__module__}.{type(var_55.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_56 = module_0.to_json_schema(var_55)
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
    var_58 = module_3.Not(var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.composites.Not'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert f'{type(var_58.negated).__module__}.{type(var_58.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_59 = module_0.to_json_schema(var_58)
    var_60 = module_2.String()
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
    var_61 = {var_45: var_60}
    var_62 = module_5.Reference(var_45, var_61)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert var_62.to == 'test'
    assert f'{type(var_62.definitions).__module__}.{type(var_62.definitions).__qualname__}' == 'builtins.dict'
    assert len(var_62.definitions) == 1
    assert module_5.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_5.Reference.target).__module__}.{type(module_5.Reference.target).__qualname__}' == 'builtins.property'
    var_63 = module_0.to_json_schema(var_62)
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
    var_65 = {var_24: var_64}
    var_66 = [var_24]
    var_67 = module_5.Schema(var_65)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_67.title == ''
    assert var_67.description == ''
    assert var_67.allow_null is False
    assert var_67.read_only is False
    assert f'{type(var_67.fields).__module__}.{type(var_67.fields).__qualname__}' == 'builtins.dict'
    assert len(var_67.fields) == 1
    assert var_67.required == ['name']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
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
    var_70 = module_2.Integer()
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.fields.Integer'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert var_70.minimum is None
    assert var_70.maximum is None
    assert var_70.exclusive_minimum is None
    assert var_70.exclusive_maximum is None
    assert var_70.multiple_of is None
    assert var_70.precision is None
    assert var_70.coerce_types is True

def test_case_55():
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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_12 = var_9.any_of
    var_13 = var_9.any_of
    var_14 = 'null'
    var_15 = [var_1, var_14]
    var_16 = {var_0: var_15}
    var_17 = module_5.Definitions()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_17) == 0
    var_18 = module_0.type_from_json_schema(var_16, var_17)
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
    var_19 = {}
    var_20 = module_5.Definitions()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_20) == 0
    var_21 = module_0.type_from_json_schema(var_19, var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Union'
    assert var_21.default is None
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is True
    assert var_21.read_only is False
    assert f'{type(var_21.any_of).__module__}.{type(var_21.any_of).__qualname__}' == 'builtins.list'
    assert len(var_21.any_of) == 5
    var_22 = [var_14]
    var_23 = {var_0: var_22}
    var_24 = module_5.Definitions()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_24) == 0
    var_25 = module_0.type_from_json_schema(var_23, var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Const'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.const is None
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_26 = 'minLength'
    var_27 = 'maxLength'
    var_28 = 'pattern'
    var_29 = 5
    var_30 = 10
    var_31 = '^[A-Za-z]+$'
    var_32 = {var_0: var_1, var_26: var_29, var_27: var_30, var_28: var_31}
    var_33 = module_5.Definitions()
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
    assert var_34.pattern == '^[A-Za-z]+$'
    assert f'{type(var_34.pattern_regex).__module__}.{type(var_34.pattern_regex).__qualname__}' == 're.Pattern'
    var_35 = 'properties'
    var_36 = 'required'
    var_37 = 'object'
    var_38 = 'name'
    var_39 = 'age'
    var_40 = {var_0: var_1}
    var_41 = {var_0: var_5}
    var_42 = {var_38: var_40, var_39: var_41}
    var_43 = [var_38]
    var_44 = {var_0: var_37, var_35: var_42, var_36: var_43}
    var_45 = module_5.Definitions()
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_45) == 0
    var_46 = module_0.type_from_json_schema(var_44, var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Object'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.properties).__module__}.{type(var_46.properties).__qualname__}' == 'builtins.dict'
    assert len(var_46.properties) == 2
    assert var_46.pattern_properties == {}
    assert var_46.additional_properties is None
    assert var_46.property_names is None
    assert var_46.min_properties is None
    assert var_46.max_properties is None
    assert var_46.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_47 = 'items'
    var_48 = 'minItems'
    var_49 = 'maxItems'
    var_50 = 'array'
    var_51 = {var_0: var_1}
    var_52 = 1
    var_53 = {var_0: var_50, var_47: var_51, var_48: var_52, var_49: var_29}
    var_54 = module_5.Definitions()
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_54) == 0
    var_55 = module_0.type_from_json_schema(var_53, var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Array'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert f'{type(var_55.items).__module__}.{type(var_55.items).__qualname__}' == 'typesystem.fields.String'
    assert var_55.additional_items is True
    assert var_55.min_items == 1
    assert var_55.max_items == 5
    assert var_55.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_56 = var_55.items

def test_case_56():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    assert module_3.NeverMatch.errors == {'never': 'This never validates.'}
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
    var_14 = module_0.from_json_schema(var_9)
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
    assert var_14.coerce_types is False
    var_15 = 'array'
    var_16 = {var_4: var_15}
    var_17 = module_0.from_json_schema(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Array'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.items is None
    assert var_17.additional_items is True
    assert var_17.min_items == 0
    assert var_17.max_items is None
    assert var_17.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_18 = 'object'
    var_19 = {var_4: var_18}
    var_20 = module_0.from_json_schema(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Object'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.properties == {}
    assert var_20.pattern_properties == {}
    assert var_20.additional_properties is None
    assert var_20.property_names is None
    assert var_20.min_properties is None
    assert var_20.max_properties is None
    assert var_20.required == []
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_21 = [var_5, var_8]
    var_22 = {var_4: var_21}
    var_23 = module_0.from_json_schema(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Union'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.any_of).__module__}.{type(var_23.any_of).__qualname__}' == 'builtins.list'
    assert len(var_23.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_24 = 'enum'
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 'c'
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_24: var_28}
    var_30 = module_0.from_json_schema(var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Choice'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')]
    assert var_30.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_31 = 'const'
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = module_0.from_json_schema(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Const'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.const == 'value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_35 = 'allOf'
    var_36 = {var_4: var_5}
    var_37 = 'minLength'
    var_38 = {var_37: var_0}
    var_39 = [var_36, var_38]
    var_40 = {var_35: var_39}
    var_41 = module_0.from_json_schema(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.all_of).__module__}.{type(var_41.all_of).__qualname__}' == 'builtins.list'
    assert len(var_41.all_of) == 2
    var_42 = 'anyOf'
    var_43 = {var_4: var_5}
    var_44 = {var_4: var_8}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_0.from_json_schema(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'typesystem.fields.Union'
    assert var_47.title == ''
    assert var_47.description == ''
    assert var_47.allow_null is False
    assert var_47.read_only is False
    assert f'{type(var_47.any_of).__module__}.{type(var_47.any_of).__qualname__}' == 'builtins.list'
    assert len(var_47.any_of) == 2
    var_48 = 'oneOf'
    var_49 = {var_4: var_5}
    var_50 = {var_4: var_8}
    var_51 = [var_49, var_50]
    var_52 = {var_48: var_51}
    var_53 = module_0.from_json_schema(var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_53.title == ''
    assert var_53.description == ''
    assert var_53.allow_null is False
    assert var_53.read_only is False
    assert f'{type(var_53.one_of).__module__}.{type(var_53.one_of).__qualname__}' == 'builtins.list'
    assert len(var_53.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_54 = 'not'
    var_55 = {var_4: var_5}
    var_56 = {var_54: var_55}
    var_57 = module_0.from_json_schema(var_56)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.composites.Not'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert f'{type(var_57.negated).__module__}.{type(var_57.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_58 = module_0.from_json_schema(var_19)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.fields.Object'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert var_58.properties == {}
    assert var_58.pattern_properties == {}
    assert var_58.additional_properties is None
    assert var_58.property_names is None
    assert var_58.min_properties is None
    assert var_58.max_properties is None
    assert var_58.required == []
    var_59 = module_5.Definitions()
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_59) == 0
    var_60 = '$ref'
    var_61 = '#/components/schemas/Test'
    var_62 = {var_60: var_61}
    var_63 = module_0.from_json_schema(var_62, var_59)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert var_63.to == '#/components/schemas/Test'
    assert f'{type(var_63.definitions).__module__}.{type(var_63.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_63.definitions) == 0
    assert module_5.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_5.Reference.target).__module__}.{type(module_5.Reference.target).__qualname__}' == 'builtins.property'
    var_64 = [var_25, var_26, var_27]
    var_65 = {var_4: var_5, var_24: var_64}
    var_66 = module_0.from_json_schema(var_65)
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert f'{type(var_66.all_of).__module__}.{type(var_66.all_of).__qualname__}' == 'builtins.list'
    assert len(var_66.all_of) == 2
    var_67 = {}
    var_68 = module_0.from_json_schema(var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.fields.Any'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False

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
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_12 = 2
    var_13 = module_2.Integer(minimum=var_10, maximum=var_11, exclusive_minimum=var_4, multiple_of=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum is False
    assert var_13.maximum == 100
    assert var_13.exclusive_minimum is True
    assert var_13.exclusive_maximum is None
    assert var_13.multiple_of == 2
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = module_0.to_json_schema(var_13)
    var_15 = module_2.Float(minimum=var_10, maximum=var_4)
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
    var_19 = 5
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
    var_21 = module_2.Array(var_20, var_10, var_4, var_19, unique_items=var_4)
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
    var_22 = 'name'
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
    var_24 = '^S_'
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
    var_26 = {var_24: var_25}
    var_27 = '[A-Z]+'
    var_28 = module_2.String(pattern=var_27)
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
    assert var_28.pattern == '[A-Z]+'
    assert f'{type(var_28.pattern_regex).__module__}.{type(var_28.pattern_regex).__qualname__}' == 're.Pattern'
    var_29 = [var_22]
    var_30 = module_2.Object(properties=var_15, pattern_properties=var_26, additional_properties=var_10, property_names=var_28, min_properties=var_4, max_properties=var_19, required=var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Object'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.properties == {}
    assert f'{type(var_30.pattern_properties).__module__}.{type(var_30.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_30.pattern_properties) == 1
    assert f'{type(var_30.additional_properties).__module__}.{type(var_30.additional_properties).__qualname__}' == 'typesystem.fields.Float'
    assert f'{type(var_30.property_names).__module__}.{type(var_30.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_30.min_properties is True
    assert var_30.max_properties == 5
    assert var_30.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_31 = module_0.to_json_schema(var_30)
    var_32 = 'a'
    var_33 = (var_32, var_32)
    var_34 = 'b'
    var_35 = (var_34, var_34)
    var_36 = [var_33, var_35]
    var_37 = module_2.Choice(choices=var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.fields.Choice'
    assert var_37.title == ''
    assert var_37.description == ''
    assert var_37.allow_null is False
    assert var_37.read_only is False
    assert var_37.choices == [('a', 'a'), ('b', 'b')]
    assert var_37.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_38 = module_0.to_json_schema(var_37)
    var_39 = 'fixed'
    var_40 = module_2.Const(var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.fields.Const'
    assert var_40.title == ''
    assert var_40.description == ''
    assert var_40.allow_null is False
    assert var_40.read_only is False
    assert var_40.const == 'fixed'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
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
    var_45 = module_2.Union(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.fields.Union'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert f'{type(var_45.any_of).__module__}.{type(var_45.any_of).__qualname__}' == 'builtins.list'
    assert len(var_45.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
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
    var_49 = [var_47, var_48]
    var_50 = module_3.OneOf(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert f'{type(var_50.one_of).__module__}.{type(var_50.one_of).__qualname__}' == 'builtins.list'
    assert len(var_50.one_of) == 2
    assert module_3.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
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
    var_53 = 'test'
    var_54 = module_2.Const(var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.fields.Const'
    assert var_54.title == ''
    assert var_54.description == ''
    assert var_54.allow_null is False
    assert var_54.read_only is False
    assert var_54.const == 'test'
    var_55 = [var_52, var_54]
    var_56 = module_3.AllOf(var_55)
    assert f'{type(var_56).__module__}.{type(var_56).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_56.title == ''
    assert var_56.description == ''
    assert var_56.allow_null is False
    assert var_56.read_only is False
    assert f'{type(var_56.all_of).__module__}.{type(var_56.all_of).__qualname__}' == 'builtins.list'
    assert len(var_56.all_of) == 2
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
    var_59 = module_2.Integer()
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
    var_60 = module_2.Boolean()
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_60.title == ''
    assert var_60.description == ''
    assert var_60.allow_null is False
    assert var_60.read_only is False
    assert var_60.coerce_types is True
    var_61 = module_3.IfThenElse(var_58, var_59, var_60)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert f'{type(var_61.if_clause).__module__}.{type(var_61.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_61.then_clause).__module__}.{type(var_61.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_61.else_clause).__module__}.{type(var_61.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
    var_62 = module_0.to_json_schema(var_61)
    var_63 = module_2.String()
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.fields.String'
    assert var_63.title == ''
    assert var_63.description == ''
    assert var_63.allow_null is False
    assert var_63.read_only is False
    assert var_63.allow_blank is False
    assert var_63.trim_whitespace is True
    assert var_63.max_length is None
    assert var_63.min_length is None
    assert var_63.format is None
    assert var_63.coerce_types is True
    assert var_63.pattern is None
    assert var_63.pattern_regex is None
    var_64 = module_3.Not(var_63)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.composites.Not'
    assert var_64.title == ''
    assert var_64.description == ''
    assert var_64.allow_null is False
    assert var_64.read_only is False
    assert f'{type(var_64.negated).__module__}.{type(var_64.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_3.Not.errors == {'negated': 'Must not match.'}
    var_65 = module_0.to_json_schema(var_64)
    var_66 = module_2.String()
    assert f'{type(var_66).__module__}.{type(var_66).__qualname__}' == 'typesystem.fields.String'
    assert var_66.title == ''
    assert var_66.description == ''
    assert var_66.allow_null is False
    assert var_66.read_only is False
    assert var_66.allow_blank is False
    assert var_66.trim_whitespace is True
    assert var_66.max_length is None
    assert var_66.min_length is None
    assert var_66.format is None
    assert var_66.coerce_types is True
    assert var_66.pattern is None
    assert var_66.pattern_regex is None
    var_67 = module_2.String()
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
    var_68 = {var_22: var_67}
    var_69 = [var_22]
    var_70 = module_5.Schema(var_68)
    assert f'{type(var_70).__module__}.{type(var_70).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_70.title == ''
    assert var_70.description == ''
    assert var_70.allow_null is False
    assert var_70.read_only is False
    assert f'{type(var_70.fields).__module__}.{type(var_70.fields).__qualname__}' == 'builtins.dict'
    assert len(var_70.fields) == 1
    assert var_70.required == ['name']
    assert module_5.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
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

def test_case_58():
    var_0 = module_2.String()
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = False
    var_2 = None
    var_3 = module_2.Array(var_0, min_items=var_2, max_items=var_2, unique_items=var_1)
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
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    var_6 = module_0.type_from_json_schema(var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.items).__module__}.{type(var_6.items).__qualname__}' == 'typesystem.fields.String'
    assert var_6.additional_items is False
    assert var_6.min_items == 0
    assert var_6.max_items is None
    assert var_6.unique_items is False

@pytest.mark.xfail(strict=True)
def test_case_59():
    var_0 = None
    var_1 = module_2.String(min_length=var_0, pattern=var_0)
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
    var_2 = False
    var_3 = module_2.Array(additional_items=var_1, min_items=var_0, max_items=var_2, exact_items=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.items is None
    assert f'{type(var_3.additional_items).__module__}.{type(var_3.additional_items).__qualname__}' == 'typesystem.fields.String'
    assert var_3.min_items is False
    assert var_3.max_items is False
    assert var_3.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'maxLength', 'minimum', 'pattern', 'propertyNames', 'contains', 'uniqueItems', 'maxItems', 'multipleOf', 'additionalProperties', 'properties', 'boolean_schema', 'required', 'type', 'exclusiveMinimum', 'maximum', 'minItems', 'dependencies', 'minProperties', 'additionalItems', 'patternProperties', 'exclusiveMaximum', 'maxProperties', 'items', 'minLength'}
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
    module_0.type_from_json_schema(var_0, var_5)