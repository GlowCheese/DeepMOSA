# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.json_schema as module_0
import typesystem.composites as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import re as module_4
import enum as module_5

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
def test_case_2():
    var_0 = None
    module_0.from_json_schema(var_0)

def test_case_3():
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_0 = {}
    var_1 = None
    with pytest.raises(AssertionError):
        module_0.from_json_schema_type(var_0, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.to_json_schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.enum_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.const_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.all_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.not_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    module_0.any_of_from_json_schema(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_0.one_of_from_json_schema(var_0, var_0)

def test_case_13():
    var_0 = None
    var_1 = module_2.Array(min_items=var_0, max_items=var_0, exact_items=var_0)
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_3 = module_0.get_standard_properties(var_2)
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    module_0.enum_from_json_schema(var_1, var_0)

def test_case_16():
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
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.coerce_types is False

def test_case_17():
    var_0 = None
    var_1 = module_2.Array(min_items=var_0, max_items=var_0, exact_items=var_0)
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = module_0.to_json_schema(var_1, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Array'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.items is None
    assert var_3.additional_items is False
    assert var_3.min_items == 0
    assert var_3.max_items is None
    assert var_3.unique_items is False

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_2.Array(max_items=var_0)
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_2 = False
    var_3 = None
    var_4 = module_0.from_json_schema(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_5 = module_0.to_json_schema(var_4, var_3)
    assert var_5 is False
    var_6 = var_4.has_default()
    assert var_6 is False
    var_7 = module_0.from_json_schema(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    module_0.type_from_json_schema(var_2, var_3)

def test_case_19():
    var_0 = None
    var_1 = module_2.Integer(exclusive_minimum=var_0, precision=var_0, multiple_of=var_0)
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_0 = None
    var_1 = None
    var_2 = module_2.Integer(exclusive_minimum=var_0, precision=var_0, multiple_of=var_1)
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    assert var_4.coerce_types is False
    var_2.validate(var_0)

def test_case_21():
    var_0 = -20
    var_1 = ''
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == -20
    assert var_2.min_length == -20
    assert var_2.format == ''
    assert var_2.coerce_types is True
    assert var_2.pattern == ''
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
    assert var_3.min_items == -20
    assert var_3.max_items == -20
    assert var_3.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_1 = module_1.Not(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.Not'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.negated is None
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    module_0.to_json_schema(var_1)

def test_case_23():
    var_0 = 'bn7J\nZR'
    var_1 = module_0.from_json_schema(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2

def test_case_24():
    var_0 = module_3.Definitions()
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.one_of[var_12]
    var_14 = 1
    var_15 = var_9.one_of[var_14]
    var_16 = 'properties'
    var_17 = 'object'
    var_18 = 'name'
    var_19 = {var_2: var_3}
    var_20 = {var_18: var_19}
    var_21 = {var_2: var_17, var_16: var_20}
    var_22 = 'array'
    var_23 = 'integer'
    var_24 = {var_2: var_23}
    var_25 = {var_2: var_22, var_3: var_24}
    var_26 = [var_21, var_25]
    var_27 = {var_1: var_26}
    var_28 = module_0.one_of_from_json_schema(var_27, var_0)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert f'{type(var_28.one_of).__module__}.{type(var_28.one_of).__qualname__}' == 'builtins.list'
    assert len(var_28.one_of) == 2
    var_29 = var_28.one_of
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = var_28.one_of[var_12]
    var_32 = var_28.one_of[var_14]
    var_33 = 'default'
    var_34 = 'boolean'
    var_35 = {var_2: var_34}
    var_36 = 'null'
    var_37 = {var_2: var_36}
    var_38 = [var_35, var_37]
    var_39 = True
    var_40 = {var_1: var_38, var_33: var_39}
    var_41 = module_0.one_of_from_json_schema(var_40, var_0)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_41.default is True
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert f'{type(var_41.one_of).__module__}.{type(var_41.one_of).__qualname__}' == 'builtins.list'
    assert len(var_41.one_of) == 2
    var_42 = '$ref'
    var_43 = '#/components/schemas/Test'
    var_44 = {var_42: var_43}
    var_45 = {var_2: var_23}
    var_46 = [var_44, var_45]
    var_47 = {var_1: var_46}
    var_48 = module_0.one_of_from_json_schema(var_47, var_0)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_48.title == ''
    assert var_48.description == ''
    assert var_48.allow_null is False
    assert var_48.read_only is False
    assert f'{type(var_48.one_of).__module__}.{type(var_48.one_of).__qualname__}' == 'builtins.list'
    assert len(var_48.one_of) == 2
    var_49 = var_48.one_of[var_12]
    var_50 = var_48.one_of[var_39]

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_4 = 'string'
    var_5 = module_0.to_json_schema(var_3)
    assert var_5 is False
    var_6 = 'integer'
    var_7 = {var_4: var_6}
    var_8 = module_0.from_json_schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Any'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    var_9 = module_0.from_json_schema(var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    var_10 = {var_1: var_4}
    var_11 = module_0.type_from_json_schema(var_10, var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Union'
    assert var_11.default is None
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is True
    assert var_11.read_only is False
    assert f'{type(var_11.any_of).__module__}.{type(var_11.any_of).__qualname__}' == 'builtins.list'
    assert len(var_11.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_12 = 'fixed_value'
    var_13 = {var_4: var_12}
    var_14 = module_0.from_json_schema(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Any'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_15 = 'allOf'
    var_16 = "'g&f&gH$xwmY"
    var_17 = 5
    var_18 = {var_15: var_4, var_16: var_17}
    var_19 = 10
    var_20 = {var_4: var_4, var_16: var_19}
    var_21 = [var_18, var_20]
    var_22 = {var_15: var_21}
    var_23 = module_0.from_json_schema(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.all_of).__module__}.{type(var_23.all_of).__qualname__}' == 'builtins.list'
    assert len(var_23.all_of) == 2
    var_24 = None
    module_0.to_json_schema(var_24)

def test_case_26():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_4 = 'string'
    var_5 = {var_4: var_4}
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Any'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    var_7 = 'number'
    var_8 = {var_4: var_7}
    var_9 = module_0.from_json_schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Any'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    var_10 = 'integer'
    var_11 = {var_4: var_10}
    var_12 = module_0.from_json_schema(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Any'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    var_13 = 'array'
    var_14 = {var_1: var_13}
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
    var_23 = 'fixed_value'
    var_24 = {var_22: var_23}
    var_25 = module_0.from_json_schema(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.fields.Const'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert var_25.const == 'fixed_value'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_26 = 'allOf'
    var_27 = 'maxLength'
    var_28 = 10
    var_29 = {var_18: var_4, var_27: var_28}
    var_30 = [var_14, var_29]
    var_31 = {var_26: var_30}
    var_32 = module_0.from_json_schema(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_32.title == ''
    assert var_32.description == ''
    assert var_32.allow_null is False
    assert var_32.read_only is False
    assert f'{type(var_32.all_of).__module__}.{type(var_32.all_of).__qualname__}' == 'builtins.list'
    assert len(var_32.all_of) == 2
    with pytest.raises(AttributeError):
        var_33 = var_32.schemas

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Any'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_4 = 'string'
    var_5 = {var_4: var_4}
    var_6 = module_0.from_json_schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Any'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    var_7 = module_0.to_json_schema(var_3)
    assert var_7 is False
    var_8 = module_0.from_json_schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    var_9 = 'integer'
    var_10 = {var_4: var_9}
    var_11 = module_0.from_json_schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Any'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    var_12 = module_0.from_json_schema(var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Any'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    var_13 = 'array'
    var_14 = {var_1: var_13}
    var_15 = module_0.type_from_json_schema(var_14, var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Union'
    assert var_15.default is None
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is True
    assert var_15.read_only is False
    assert f'{type(var_15.any_of).__module__}.{type(var_15.any_of).__qualname__}' == 'builtins.list'
    assert len(var_15.any_of) == 5
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_16 = 'const'
    var_17 = 'fixed_value'
    var_18 = {var_16: var_17}
    var_19 = module_0.from_json_schema(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Const'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.const == 'fixed_value'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_20 = "'g&f&gH$xwmY"
    var_21 = {var_13: var_4, var_20: var_7}
    var_22 = 10
    var_23 = {var_4: var_4, var_20: var_22}
    var_24 = [var_21, var_23]
    var_25 = {var_16: var_24}
    var_26 = module_0.from_json_schema(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.fields.Const'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert var_26.const == [{'array': 'string', "'g&f&gH$xwmY": False}, {'string': 'string', "'g&f&gH$xwmY": 10}]
    var_27 = None
    module_0.to_json_schema(var_27)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'oneOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3, var_1: var_1}
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.one_of[var_12]
    var_14 = 1
    var_15 = var_9.one_of[var_14]
    var_16 = 'properties'
    var_17 = 'object'
    var_18 = 'name'
    var_19 = {var_2: var_3}
    var_20 = {var_18: var_19}
    var_21 = {var_2: var_17, var_16: var_20}
    var_22 = 'items'
    var_23 = 'array'
    var_24 = 'integer'
    var_25 = {var_2: var_24}
    var_26 = {var_2: var_23, var_22: var_25}
    var_27 = [var_21, var_26]
    var_28 = {var_1: var_27}
    var_29 = module_0.one_of_from_json_schema(var_28, var_0)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_29.title == ''
    assert var_29.description == ''
    assert var_29.allow_null is False
    assert var_29.read_only is False
    assert f'{type(var_29.one_of).__module__}.{type(var_29.one_of).__qualname__}' == 'builtins.list'
    assert len(var_29.one_of) == 2
    var_30 = len(var_3)
    var_31 = var_29.one_of[var_12]
    var_32 = var_29.one_of[var_14]
    var_33 = {var_2: var_23}
    var_34 = 'null'
    var_35 = {var_2: var_34}
    var_36 = [var_33, var_35]
    var_37 = True
    var_38 = {var_1: var_36, var_31: var_37}
    var_39 = module_0.one_of_from_json_schema(var_38, var_0)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.one_of).__module__}.{type(var_39.one_of).__qualname__}' == 'builtins.list'
    assert len(var_39.one_of) == 2
    var_40 = {}
    module_0.one_of_from_json_schema(var_40, var_0)

def test_case_29():
    var_0 = -12
    var_1 = 'J'
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == -12
    assert var_2.min_length == -12
    assert var_2.format == 'J'
    assert var_2.coerce_types is True
    assert var_2.pattern == 'J'
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
    assert var_3.min_items == -12
    assert var_3.max_items == -12
    assert var_3.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    assert var_5.min_items == -12
    assert var_5.max_items == -12
    assert var_5.unique_items is False

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_3.Definitions()
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.one_of[var_12]
    var_14 = var_9.one_of[var_12]
    var_15 = 'properties'
    var_16 = 'object'
    var_17 = 'name'
    var_18 = {var_2: var_3}
    var_19 = {var_17: var_18}
    var_20 = {var_2: var_16, var_15: var_19}
    var_21 = 'items'
    var_22 = 'array'
    var_23 = {var_2: var_1}
    var_24 = {var_2: var_22, var_21: var_23}
    var_25 = [var_20, var_24]
    var_26 = {var_1: var_25}
    module_0.one_of_from_json_schema(var_26, var_0)

def test_case_31():
    var_0 = 2
    var_1 = 'X{I/BAe5f'
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == 2
    assert var_2.min_length == 2
    assert var_2.format == 'X{I/BAe5f'
    assert var_2.coerce_types is True
    assert var_2.pattern == 'X{I/BAe5f'
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
    assert var_3.min_items == 2
    assert var_3.max_items == 2
    assert var_3.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = var_2.__or__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_0.to_json_schema(var_3)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.items).__module__}.{type(var_6.items).__qualname__}' == 'typesystem.fields.String'
    assert var_6.additional_items is False
    assert var_6.min_items == 2
    assert var_6.max_items == 2
    assert var_6.unique_items is False

def test_case_32():
    var_0 = 'enum'
    var_1 = module_3.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = 'default'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_0: var_6, var_2: var_4}
    var_8 = module_3.Definitions()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_8) == 0
    var_9 = module_0.enum_from_json_schema(var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Choice'
    assert var_9.default == 2
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.choices == [(1, 1), (2, 2), (3, 3)]
    assert var_9.coerce_types is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'oneOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3, var_1: var_1}
    var_5 = {var_2: var_2}
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_6}
    module_0.one_of_from_json_schema(var_7, var_0)

@pytest.mark.xfail(strict=True)
def test_case_34():
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_13 = True
    var_14 = module_2.Integer(minimum=var_10, maximum=var_11, exclusive_minimum=var_12, exclusive_maximum=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.fields.Integer'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.minimum == 0
    assert var_14.maximum == 100
    assert var_14.exclusive_minimum is True
    assert var_14.exclusive_maximum is True
    assert var_14.multiple_of is None
    assert var_14.precision is None
    assert var_14.coerce_types is True
    var_15 = True
    var_16 = module_0.to_json_schema(var_14)
    var_17 = 0.1
    var_18 = module_2.Float(minimum=var_10, maximum=var_15, multiple_of=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Float'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.minimum == 0
    assert var_18.maximum is True
    assert var_18.exclusive_minimum is None
    assert var_18.exclusive_maximum is None
    assert var_18.multiple_of == pytest.approx(0.1, abs=0.01, rel=0.01)
    assert var_18.precision is None
    assert var_18.coerce_types is True
    var_19 = module_0.to_json_schema(var_18)
    var_20 = module_2.Boolean()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_21 = module_0.to_json_schema(var_20)
    var_22 = module_2.String()
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
    var_23 = True
    var_24 = module_2.Array(var_22, min_items=var_15, max_items=var_5, unique_items=var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Array'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.items).__module__}.{type(var_24.items).__qualname__}' == 'typesystem.fields.String'
    assert var_24.additional_items is False
    assert var_24.min_items is True
    assert var_24.max_items == 10
    assert var_24.unique_items is True
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_25 = True
    var_26 = module_0.to_json_schema(var_24)
    var_27 = 'name'
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
    var_30 = [var_27]
    var_31 = module_2.Object(properties=var_29, min_properties=var_25, max_properties=var_5, required=var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Object'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert f'{type(var_31.properties).__module__}.{type(var_31.properties).__qualname__}' == 'builtins.dict'
    assert len(var_31.properties) == 1
    assert var_31.pattern_properties == {}
    assert var_31.additional_properties is True
    assert var_31.property_names is None
    assert var_31.min_properties is True
    assert var_31.max_properties == 10
    assert var_31.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_32 = module_0.to_json_schema(var_31)
    var_33 = 'a'
    var_34 = (var_33, var_33)
    var_35 = 'b'
    var_36 = (var_35, var_35)
    var_37 = [var_34, var_36]
    var_38 = module_2.Choice(choices=var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.fields.Choice'
    assert var_38.title == ''
    assert var_38.description == ''
    assert var_38.allow_null is False
    assert var_38.read_only is False
    assert var_38.choices == [('a', 'a'), ('b', 'b')]
    assert var_38.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_39 = module_0.to_json_schema(var_38)
    var_40 = 'fixed_value'
    var_41 = module_2.Const(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.fields.Const'
    assert var_41.title == ''
    assert var_41.description == ''
    assert var_41.allow_null is False
    assert var_41.read_only is False
    assert var_41.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
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
    var_46 = module_2.Union(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.fields.Union'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.any_of).__module__}.{type(var_46.any_of).__qualname__}' == 'builtins.list'
    assert len(var_46.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
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
    var_51 = module_1.OneOf(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert f'{type(var_51.one_of).__module__}.{type(var_51.one_of).__qualname__}' == 'builtins.list'
    assert len(var_51.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_52 = module_0.to_json_schema(var_51)
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
    var_54 = 'test'
    var_55 = module_2.Const(var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.fields.Const'
    assert var_55.title == ''
    assert var_55.description == ''
    assert var_55.allow_null is False
    assert var_55.read_only is False
    assert var_55.const == 'test'
    var_56 = [var_53, var_55]
    var_57 = module_1.AllOf(var_56)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_57.title == ''
    assert var_57.description == ''
    assert var_57.allow_null is False
    assert var_57.read_only is False
    assert f'{type(var_57.all_of).__module__}.{type(var_57.all_of).__qualname__}' == 'builtins.list'
    assert len(var_57.all_of) == 2
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
    var_61 = module_2.Boolean()
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert var_61.coerce_types is True
    var_62 = module_1.IfThenElse(var_59, var_60, var_61)
    assert f'{type(var_62).__module__}.{type(var_62).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_62.title == ''
    assert var_62.description == ''
    assert var_62.allow_null is False
    assert var_62.read_only is False
    assert f'{type(var_62.if_clause).__module__}.{type(var_62.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_62.then_clause).__module__}.{type(var_62.then_clause).__qualname__}' == 'typesystem.fields.Integer'
    assert f'{type(var_62.else_clause).__module__}.{type(var_62.else_clause).__qualname__}' == 'typesystem.fields.Boolean'
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
    var_65 = module_1.Not(var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'typesystem.composites.Not'
    assert var_65.title == ''
    assert var_65.description == ''
    assert var_65.allow_null is False
    assert var_65.read_only is False
    assert f'{type(var_65.negated).__module__}.{type(var_65.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_66 = module_0.to_json_schema(var_65)
    var_67 = module_3.Definitions()
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_67) == 0
    var_68 = module_3.Reference(var_54, var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_68.title == ''
    assert var_68.description == ''
    assert var_68.allow_null is False
    assert var_68.read_only is False
    assert var_68.to == 'test'
    assert f'{type(var_68.definitions).__module__}.{type(var_68.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_68.definitions) == 0
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    module_0.to_json_schema(var_68)

def test_case_35():
    var_0 = None
    var_1 = module_2.Object(properties=var_0, pattern_properties=var_0, required=var_0)
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_0 = -4645
    var_1 = 'W:#PW4&=-bTM'
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == -4645
    assert var_2.min_length == -4645
    assert var_2.format == 'W:#PW4&=-bTM'
    assert var_2.coerce_types is True
    assert var_2.pattern == 'W:#PW4&=-bTM'
    assert f'{type(var_2.pattern_regex).__module__}.{type(var_2.pattern_regex).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_1 = module_0.to_json_schema(var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_0 = None
    var_1 = module_2.String(coerce_types=var_0)
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
    assert var_1.coerce_types is None
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
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
    var_3 = module_1.IfThenElse(var_1, else_clause=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.if_clause).__module__}.{type(var_3.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_4 = module_0.to_json_schema(var_3, var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_0 = module_2.Any()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Any'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_1 = 1
    var_2 = 0
    var_3 = 100
    var_4 = module_2.Integer(minimum=var_2, maximum=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Integer'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum == 0
    assert var_4.maximum == 100
    assert var_4.exclusive_minimum is None
    assert var_4.exclusive_maximum is None
    assert var_4.multiple_of is None
    assert var_4.precision is None
    assert var_4.coerce_types is True
    var_5 = module_2.Boolean()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.coerce_types is True
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_6 = module_0.to_json_schema(var_5)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
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
    var_8 = 5
    var_9 = module_2.Array(var_7, min_items=var_1, max_items=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Array'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.items).__module__}.{type(var_9.items).__qualname__}' == 'typesystem.fields.String'
    assert var_9.additional_items is False
    assert var_9.min_items == 1
    assert var_9.max_items == 5
    assert var_9.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_10 = module_0.to_json_schema(var_9)
    var_11 = 'name'
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
    var_13 = {var_11: var_12}
    var_14 = [var_11]
    var_15 = module_2.Object(properties=var_13, required=var_14)
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
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_16 = module_0.to_json_schema(var_15)
    var_17 = 'fixed'
    var_18 = module_2.Const(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.fields.Const'
    assert var_18.title == ''
    assert var_18.description == ''
    assert var_18.allow_null is False
    assert var_18.read_only is False
    assert var_18.const == 'fixed'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
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
    var_21 = module_2.Integer()
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
    var_22 = [var_20, var_21]
    var_23 = module_2.Union(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Union'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.any_of).__module__}.{type(var_23.any_of).__qualname__}' == 'builtins.list'
    assert len(var_23.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_4.validate(var_9)

def test_case_40():
    var_0 = None
    var_1 = module_2.Float(exclusive_maximum=var_0, precision=var_0)
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_0 = module_5._EnumDict()
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_3 = 0
    var_4 = 100
    var_5 = module_2.Integer(minimum=var_3, maximum=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Integer'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.minimum == 0
    assert var_5.maximum == 100
    assert var_5.exclusive_minimum is None
    assert var_5.exclusive_maximum is None
    assert var_5.multiple_of is None
    assert var_5.precision is None
    assert var_5.coerce_types is True
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
    var_7 = module_0.to_json_schema(var_6)
    var_8 = module_2.String()
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
    assert var_8.coerce_types is True
    assert var_8.pattern is None
    assert var_8.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_9 = 5
    var_10 = module_2.Array(var_8, min_items=var_2, max_items=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Array'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.items).__module__}.{type(var_10.items).__qualname__}' == 'typesystem.fields.String'
    assert var_10.additional_items is False
    assert var_10.min_items == 1
    assert var_10.max_items == 5
    assert var_10.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_11 = module_0.to_json_schema(var_10)
    var_12 = 'name'
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
    var_14 = {var_12: var_13}
    var_15 = [var_12]
    var_16 = module_2.Object(properties=var_14, required=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Object'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert f'{type(var_16.properties).__module__}.{type(var_16.properties).__qualname__}' == 'builtins.dict'
    assert len(var_16.properties) == 1
    assert var_16.pattern_properties == {}
    assert var_16.additional_properties is True
    assert var_16.property_names is None
    assert var_16.min_properties is None
    assert var_16.max_properties is None
    assert var_16.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_17 = module_0.to_json_schema(var_16)
    var_18 = 'fixed'
    var_19 = module_2.Const(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Const'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert var_19.const == 'fixed'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
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
    var_23 = [var_21, var_22]
    var_24 = module_2.Union(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.fields.Union'
    assert var_24.title == ''
    assert var_24.description == ''
    assert var_24.allow_null is False
    assert var_24.read_only is False
    assert f'{type(var_24.any_of).__module__}.{type(var_24.any_of).__qualname__}' == 'builtins.list'
    assert len(var_24.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_25 = module_4.purge()
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
    var_27 = 'test'
    var_28 = module_2.Const(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.fields.Const'
    assert var_28.title == ''
    assert var_28.description == ''
    assert var_28.allow_null is False
    assert var_28.read_only is False
    assert var_28.const == 'test'
    var_29 = 'Test'
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
    var_31 = {var_29: var_30}
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
    var_33 = module_3.Schema(var_31)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert f'{type(var_33.fields).__module__}.{type(var_33.fields).__qualname__}' == 'builtins.dict'
    assert len(var_33.fields) == 1
    assert var_33.required == ['Test']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
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
    var_25.clear()

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
    var_1 = -20
    var_2 = '[-z]'
    var_3 = module_2.String(max_length=var_1, min_length=var_1, pattern=var_2, format=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length == -20
    assert var_3.min_length == -20
    assert var_3.format == '[-z]'
    assert var_3.coerce_types is True
    assert var_3.pattern == '[-z]'
    assert f'{type(var_3.pattern_regex).__module__}.{type(var_3.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = module_2.Integer(minimum=var_1, maximum=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Integer'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.minimum == -20
    assert var_4.maximum == -20
    assert var_4.exclusive_minimum is None
    assert var_4.exclusive_maximum is None
    assert var_4.multiple_of is None
    assert var_4.precision is None
    assert var_4.coerce_types is True
    var_5 = 5
    var_6 = module_2.Array(var_3, min_items=var_1, max_items=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.items).__module__}.{type(var_6.items).__qualname__}' == 'typesystem.fields.String'
    assert var_6.additional_items is False
    assert var_6.min_items == -20
    assert var_6.max_items == 5
    assert var_6.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_7 = module_0.to_json_schema(var_6)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_8 = var_0.validate(var_1)
    assert var_8 == -20
    var_9 = module_0.to_json_schema(var_4)

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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_4 = 10
    var_5 = '[a-z]+'
    var_6 = 'email'
    var_7 = module_2.String(max_length=var_4, min_length=var_4, pattern=var_5, format=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.String'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert var_7.allow_blank is False
    assert var_7.trim_whitespace is True
    assert var_7.max_length == 10
    assert var_7.min_length == 10
    assert var_7.format == 'email'
    assert var_7.coerce_types is True
    assert var_7.pattern == '[a-z]+'
    assert f'{type(var_7.pattern_regex).__module__}.{type(var_7.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_0.to_json_schema(var_7)
    var_9 = 0
    var_10 = 100
    var_11 = True
    var_12 = True
    var_13 = module_2.Integer(minimum=var_9, maximum=var_10, exclusive_minimum=var_11, exclusive_maximum=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum == 0
    assert var_13.maximum == 100
    assert var_13.exclusive_minimum is True
    assert var_13.exclusive_maximum is True
    assert var_13.multiple_of is None
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = True
    var_15 = module_0.to_json_schema(var_13)
    var_16 = 0.1
    var_17 = module_2.Float(minimum=var_9, maximum=var_14, multiple_of=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Float'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum == 0
    assert var_17.maximum is True
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of == pytest.approx(0.1, abs=0.01, rel=0.01)
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
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
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
    var_23 = module_2.Array(var_21, min_items=var_14, max_items=var_4, unique_items=var_22)
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
    var_27 = [var_25]
    var_28 = 'a'
    var_29 = (var_28, var_28)
    var_30 = 'b'
    var_31 = (var_30, var_30)
    var_32 = [var_29, var_31]
    var_33 = module_2.Choice(choices=var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Choice'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.choices == [('a', 'a'), ('b', 'b')]
    assert var_33.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_34 = module_0.to_json_schema(var_33)
    var_35 = 'fixed_value'
    var_36 = module_2.Const(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Const'
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
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
    var_46 = module_1.OneOf(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_46.title == ''
    assert var_46.description == ''
    assert var_46.allow_null is False
    assert var_46.read_only is False
    assert f'{type(var_46.one_of).__module__}.{type(var_46.one_of).__qualname__}' == 'builtins.list'
    assert len(var_46.one_of) == 2
    assert module_1.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
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
    var_49 = 'test'
    var_50 = module_2.Const(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.fields.Const'
    assert var_50.title == ''
    assert var_50.description == ''
    assert var_50.allow_null is False
    assert var_50.read_only is False
    assert var_50.const == 'test'
    var_51 = module_1.AllOf(var_27)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_51.title == ''
    assert var_51.description == ''
    assert var_51.allow_null is False
    assert var_51.read_only is False
    assert var_51.all_of == ['name']
    module_0.to_json_schema(var_51)

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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    assert var_7.format == '[a-z]+'
    assert var_7.coerce_types is True
    assert var_7.pattern == '[a-z]+'
    assert f'{type(var_7.pattern_regex).__module__}.{type(var_7.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_8 = module_0.to_json_schema(var_7)
    var_9 = 0
    var_10 = 100
    var_11 = True
    var_12 = True
    var_13 = module_2.Integer(minimum=var_9, maximum=var_10, exclusive_minimum=var_11, exclusive_maximum=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Integer'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.minimum == 0
    assert var_13.maximum == 100
    assert var_13.exclusive_minimum is True
    assert var_13.exclusive_maximum is True
    assert var_13.multiple_of is None
    assert var_13.precision is None
    assert var_13.coerce_types is True
    var_14 = True
    var_15 = module_0.to_json_schema(var_13)
    var_16 = 0.1
    var_17 = module_2.Float(minimum=var_9, maximum=var_14, multiple_of=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Float'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert var_17.minimum == 0
    assert var_17.maximum is True
    assert var_17.exclusive_minimum is None
    assert var_17.exclusive_maximum is None
    assert var_17.multiple_of == pytest.approx(0.1, abs=0.01, rel=0.01)
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
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
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
    var_23 = module_2.Array(var_21, min_items=var_14, max_items=var_5, unique_items=var_22)
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
    var_26 = 'a'
    var_27 = (var_26, var_26)
    var_28 = 'b'
    var_29 = (var_28, var_28)
    var_30 = [var_27, var_29]
    var_31 = module_2.Choice(choices=var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.fields.Choice'
    assert var_31.title == ''
    assert var_31.description == ''
    assert var_31.allow_null is False
    assert var_31.read_only is False
    assert var_31.choices == [('a', 'a'), ('b', 'b')]
    assert var_31.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_32 = module_0.to_json_schema(var_31)
    var_33 = 'fixed_value'
    var_34 = module_2.Const(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.fields.Const'
    assert var_34.title == ''
    assert var_34.description == ''
    assert var_34.allow_null is False
    assert var_34.read_only is False
    assert var_34.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_35 = module_0.to_json_schema(var_34)
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
    var_38 = [var_36, var_37]
    var_39 = module_2.Union(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Union'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.any_of).__module__}.{type(var_39.any_of).__qualname__}' == 'builtins.list'
    assert len(var_39.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
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
    var_55 = module_1.IfThenElse(var_52, var_53, var_54)
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
    var_58 = module_1.Not(var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.composites.Not'
    assert var_58.title == ''
    assert var_58.description == ''
    assert var_58.allow_null is False
    assert var_58.read_only is False
    assert f'{type(var_58.negated).__module__}.{type(var_58.negated).__qualname__}' == 'typesystem.fields.String'
    assert module_1.Not.errors == {'negated': 'Must not match.'}
    var_59 = module_0.to_json_schema(var_58)
    var_60 = module_3.Definitions()
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_60) == 0
    var_61 = module_3.Reference(var_47, var_60)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_61.title == ''
    assert var_61.description == ''
    assert var_61.allow_null is False
    assert var_61.read_only is False
    assert var_61.to == 'test'
    assert f'{type(var_61.definitions).__module__}.{type(var_61.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_61.definitions) == 0
    assert module_3.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_3.Reference.target).__module__}.{type(module_3.Reference.target).__qualname__}' == 'builtins.property'
    module_0.to_json_schema(var_61)

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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_14 = module_0.to_json_schema(var_13)
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
    var_16 = 5
    var_17 = module_2.Array(var_15, min_items=var_4, max_items=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Array'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.items).__module__}.{type(var_17.items).__qualname__}' == 'typesystem.fields.String'
    assert var_17.additional_items is False
    assert var_17.min_items == 1
    assert var_17.max_items == 5
    assert var_17.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_18 = module_0.to_json_schema(var_17)
    var_19 = 'name'
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
    var_21 = {var_19: var_20}
    var_22 = [var_19]
    var_23 = module_2.Object(properties=var_21, required=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Object'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    assert f'{type(var_23.properties).__module__}.{type(var_23.properties).__qualname__}' == 'builtins.dict'
    assert len(var_23.properties) == 1
    assert var_23.pattern_properties == {}
    assert var_23.additional_properties is True
    assert var_23.property_names is None
    assert var_23.min_properties is None
    assert var_23.max_properties is None
    assert var_23.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_24 = module_0.to_json_schema(var_23)
    var_25 = 'a'
    var_26 = (var_25, var_25)
    var_27 = 'b'
    var_28 = (var_27, var_27)
    var_29 = [var_26, var_28]
    var_30 = module_2.Choice(choices=var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.Choice'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.choices == [('a', 'a'), ('b', 'b')]
    assert var_30.coerce_types is True
    assert module_2.Choice.errors == {'null': 'May not be null.', 'required': 'This field is required.', 'choice': 'Not a valid choice.'}
    var_31 = module_0.to_json_schema(var_30)
    var_32 = 'fixed'
    var_33 = module_2.Const(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.fields.Const'
    assert var_33.title == ''
    assert var_33.description == ''
    assert var_33.allow_null is False
    assert var_33.read_only is False
    assert var_33.const == 'fixed'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_34 = 'const'
    var_35 = module_0.to_json_schema(var_33)
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
    var_38 = [var_36, var_37]
    var_39 = module_2.Union(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.fields.Union'
    assert var_39.title == ''
    assert var_39.description == ''
    assert var_39.allow_null is False
    assert var_39.read_only is False
    assert f'{type(var_39.any_of).__module__}.{type(var_39.any_of).__qualname__}' == 'builtins.list'
    assert len(var_39.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
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
    var_42 = 'test'
    var_43 = module_2.Const(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.fields.Const'
    assert var_43.title == ''
    assert var_43.description == ''
    assert var_43.allow_null is False
    assert var_43.read_only is False
    assert var_43.const == 'test'
    var_44 = [var_41, var_43]
    var_45 = module_1.AllOf(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_45.title == ''
    assert var_45.description == ''
    assert var_45.allow_null is False
    assert var_45.read_only is False
    assert f'{type(var_45.all_of).__module__}.{type(var_45.all_of).__qualname__}' == 'builtins.list'
    assert len(var_45.all_of) == 2
    var_46 = {var_34: var_42}
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
    module_3.Schema(var_46)

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = -43
    var_1 = None
    var_2 = module_2.Const(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Const'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.const is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_3 = module_0.to_json_schema(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = 'J'
    var_5 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_4, format=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.String'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.allow_blank is False
    assert var_5.trim_whitespace is True
    assert var_5.max_length == -43
    assert var_5.min_length == -43
    assert var_5.format == 'J'
    assert var_5.coerce_types is True
    assert var_5.pattern == 'J'
    assert f'{type(var_5.pattern_regex).__module__}.{type(var_5.pattern_regex).__qualname__}' == 're.Pattern'
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_6 = module_2.Array(var_5, min_items=var_0, max_items=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Array'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.items).__module__}.{type(var_6.items).__qualname__}' == 'typesystem.fields.String'
    assert var_6.additional_items is False
    assert var_6.min_items == -43
    assert var_6.max_items == -43
    assert var_6.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_7 = module_0.to_json_schema(var_6)
    var_8 = module_0.from_json_schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Array'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.items).__module__}.{type(var_8.items).__qualname__}' == 'typesystem.fields.String'
    assert var_8.additional_items is False
    assert var_8.min_items == -43
    assert var_8.max_items == -43
    assert var_8.unique_items is False
    module_0.not_from_json_schema(var_4, var_8)

def test_case_48():
    var_0 = -20
    var_1 = ''
    var_2 = module_2.String(max_length=var_0, min_length=var_0, pattern=var_1, format=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length == -20
    assert var_2.min_length == -20
    assert var_2.format == ''
    assert var_2.coerce_types is True
    assert var_2.pattern == ''
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
    assert var_3.min_items == -20
    assert var_3.max_items == -20
    assert var_3.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_4 = var_2.__or__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 2
    assert module_2.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_6 = None
    var_7 = module_0.from_json_schema(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Union'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.any_of).__module__}.{type(var_7.any_of).__qualname__}' == 'builtins.list'
    assert len(var_7.any_of) == 2

def test_case_49():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_2 = 'string'
    var_3 = {var_2: var_2}
    var_4 = module_0.from_json_schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Any'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    var_5 = module_0.to_json_schema(var_1)
    assert var_5 is False
    var_6 = var_1.has_default()
    assert var_6 is False
    var_7 = module_0.to_json_schema(var_1)
    assert var_7 is False
    var_8 = module_0.from_json_schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    var_9 = 'integer'
    var_10 = {var_2: var_9}
    var_11 = module_0.from_json_schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Any'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    var_12 = module_0.from_json_schema(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.fields.Any'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    var_13 = 'const'
    var_14 = 'fixed_value'
    var_15 = {var_13: var_14}
    var_16 = module_0.from_json_schema(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.fields.Const'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.const == 'fixed_value'
    assert module_2.Const.errors == {'only_null': 'Must be null.', 'const': "Must be the value '{const}'."}
    var_17 = 'allOf'
    var_18 = "'g&f&gH$xwmY"
    var_19 = 5
    var_20 = {var_17: var_2, var_18: var_19}
    var_21 = 10
    var_22 = {var_2: var_2, var_18: var_21}
    var_23 = [var_20, var_22]
    var_24 = {var_17: var_23}
    var_25 = module_0.from_json_schema(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert f'{type(var_25.all_of).__module__}.{type(var_25.all_of).__qualname__}' == 'builtins.list'
    assert len(var_25.all_of) == 2
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_26 = None
    var_27 = module_0.to_json_schema(var_25, var_26)

def test_case_50():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '$ref'
    var_2 = {var_1: var_1}
    with pytest.raises(AssertionError):
        module_0.ref_from_json_schema(var_2, var_0)

def test_case_51():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'minLength'
    var_8 = 5
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_18 = {var_4: var_5, var_7: var_8}
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
    var_24 = {var_4: var_10}
    var_25 = {var_1: var_23, var_3: var_24}
    var_26 = module_0.if_then_else_from_json_schema(var_25, var_0)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_26.title == ''
    assert var_26.description == ''
    assert var_26.allow_null is False
    assert var_26.read_only is False
    assert f'{type(var_26.if_clause).__module__}.{type(var_26.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_26.then_clause).__module__}.{type(var_26.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_26.else_clause).__module__}.{type(var_26.else_clause).__qualname__}' == 'typesystem.fields.Float'
    var_27 = var_26.if_clause
    var_28 = var_26.else_clause
    var_29 = 'default'
    var_30 = {var_4: var_5}
    var_31 = {var_4: var_5, var_7: var_8}
    var_32 = {var_4: var_10}
    var_33 = 'default_value'
    var_34 = {var_1: var_30, var_2: var_31, var_3: var_32, var_29: var_33}
    var_35 = module_0.if_then_else_from_json_schema(var_34, var_0)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_35.default == 'default_value'
    assert var_35.title == ''
    assert var_35.description == ''
    assert var_35.allow_null is False
    assert var_35.read_only is False
    assert f'{type(var_35.if_clause).__module__}.{type(var_35.if_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_35.then_clause).__module__}.{type(var_35.then_clause).__qualname__}' == 'typesystem.fields.String'
    assert f'{type(var_35.else_clause).__module__}.{type(var_35.else_clause).__qualname__}' == 'typesystem.fields.Float'

@pytest.mark.xfail(strict=True)
def test_case_52():
    var_0 = module_3.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = module_4.escape(var_1)
    assert var_4 == 'if'
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
    var_5 = var_0.clear()
    var_6 = {var_1: var_4, var_2: var_4, var_3: var_5}
    module_0.if_then_else_from_json_schema(var_6, var_0)

def test_case_53():
    var_0 = None
    var_1 = module_2.Object(properties=var_0, pattern_properties=var_0, required=var_0)
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
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_2 = module_0.to_json_schema(var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_3 = module_0.from_json_schema(var_2, var_0)
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

def test_case_54():
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
    var_14 = module_3.Definitions()
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
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_18 = False
    var_19 = module_3.Definitions()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_19) == 0
    var_20 = module_0.from_json_schema_type(var_17, var_16, var_18, var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.fields.Integer'
    assert var_20.default == 50
    assert var_20.title == ''
    assert var_20.description == ''
    assert var_20.allow_null is False
    assert var_20.read_only is False
    assert var_20.minimum is False
    assert var_20.maximum == 100
    assert var_20.exclusive_minimum is False
    assert var_20.exclusive_maximum == 100
    assert var_20.multiple_of == 2
    assert var_20.precision is None
    assert var_20.coerce_types is False
    var_21 = 'minLength'
    var_22 = 'maxLength'
    var_23 = 'format'
    var_24 = 'string'
    var_25 = 5
    var_26 = '^[a-zA-Z0-9]+$'
    var_27 = 'test'
    var_28 = {var_0: var_24, var_21: var_25, var_22: var_9, var_23: var_23, var_7: var_26, var_6: var_27}
    var_29 = False
    var_30 = module_0.from_json_schema_type(var_28, var_24, var_29, var_14)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.fields.String'
    assert var_30.default == 'test'
    assert var_30.title == ''
    assert var_30.description == ''
    assert var_30.allow_null is False
    assert var_30.read_only is False
    assert var_30.allow_blank is False
    assert var_30.trim_whitespace is True
    assert var_30.max_length == 100
    assert var_30.min_length == 5
    assert var_30.format == 'format'
    assert var_30.coerce_types is False
    assert var_30.pattern is None
    assert var_30.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_31 = 'boolean'
    var_32 = True
    var_33 = {var_0: var_31, var_6: var_32}
    var_34 = False
    var_35 = module_3.Definitions()
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_35) == 0
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    var_36 = module_0.from_json_schema_type(var_33, var_31, var_34, var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_36.default is True
    assert var_36.title == ''
    assert var_36.description == ''
    assert var_36.allow_null is False
    assert var_36.read_only is False
    assert var_36.coerce_types is False
    assert module_2.Boolean.errors == {'type': 'Must be a boolean.', 'null': 'May not be null.'}
    assert module_2.Boolean.coerce_values == {'true': True, 'false': False, 'on': True, 'off': False, '1': True, '0': False, '': False, 1: True, 0: False}
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_37 = 'items'
    var_38 = 'additionalItems'
    var_39 = 'minItems'
    var_40 = 'maxItems'
    var_41 = 'uniqueItems'
    var_42 = 'array'
    var_43 = {var_0: var_24}
    var_44 = False
    var_45 = 10
    var_46 = [var_27]
    var_47 = {var_0: var_42, var_37: var_43, var_38: var_44, var_39: var_32, var_40: var_45, var_41: var_32, var_6: var_46}
    var_48 = False
    var_49 = module_3.Definitions()
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
    assert var_50.additional_items is False
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
    var_62 = {var_0: var_24}
    var_63 = {var_0: var_16}
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = '^S_'
    var_66 = '^I_'
    var_67 = {var_0: var_24}
    var_68 = {var_65: var_67, var_66: var_47}
    var_69 = False
    var_70 = {var_0: var_24}
    var_71 = [var_60]
    var_72 = 30
    var_73 = {var_60: var_27, var_61: var_72}
    var_74 = {var_0: var_59, var_52: var_64, var_53: var_68, var_54: var_69, var_55: var_70, var_56: var_32, var_57: var_45, var_58: var_71, var_6: var_73}
    var_75 = False
    var_76 = module_3.Definitions()
    assert f'{type(var_76).__module__}.{type(var_76).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_76) == 0
    var_77 = module_0.from_json_schema_type(var_74, var_59, var_75, var_76)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'typesystem.fields.Object'
    assert var_77.default == {'name': 'test', 'age': 30}
    assert var_77.title == ''
    assert var_77.description == ''
    assert var_77.allow_null is False
    assert var_77.read_only is False
    assert f'{type(var_77.properties).__module__}.{type(var_77.properties).__qualname__}' == 'builtins.dict'
    assert len(var_77.properties) == 2
    assert f'{type(var_77.pattern_properties).__module__}.{type(var_77.pattern_properties).__qualname__}' == 'builtins.dict'
    assert len(var_77.pattern_properties) == 2
    assert var_77.additional_properties is False
    assert f'{type(var_77.property_names).__module__}.{type(var_77.property_names).__qualname__}' == 'typesystem.fields.String'
    assert var_77.min_properties is True
    assert var_77.max_properties == 10
    assert var_77.required == ['name']
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_78 = var_77.properties[var_60]
    var_79 = var_77.properties[var_61]
    var_80 = var_77.pattern_properties[var_65]
    with pytest.raises(AttributeError):
        var_81 = var_2.pattern_properties[var_66]

def test_case_55():
    var_0 = None
    var_1 = {}
    var_2 = module_2.Object(properties=var_0, additional_properties=var_0, max_properties=var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Object'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.properties == {}
    assert var_2.pattern_properties == {}
    assert var_2.additional_properties is None
    assert var_2.property_names is None
    assert var_2.min_properties is None
    assert var_2.max_properties is None
    assert var_2.required == []
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Object.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.', 'invalid_property': 'Invalid property name.', 'empty': 'Must not be empty.', 'max_properties': 'Must have no more than {max_properties} properties.', 'min_properties': 'Must have at least {min_properties} properties.'}
    var_3 = module_0.to_json_schema(var_2, var_0)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
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
    var_0 = True
    var_1 = None
    var_2 = module_2.String(allow_blank=var_0, max_length=var_0, coerce_types=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.default == ''
    assert var_2.allow_blank is True
    assert var_2.trim_whitespace is True
    assert var_2.max_length is True
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is None
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = module_0.to_json_schema(var_2, var_1)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
    var_4 = var_2.serialize(var_1)
    var_5 = module_1.NeverMatch()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}

def test_case_57():
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
    assert module_2.Boolean.coerce_null_values == {'', 'null', 'none'}
    var_1 = None
    var_2 = module_2.String(trim_whitespace=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is None
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_3 = False
    var_4 = module_2.Array(additional_items=var_0, unique_items=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Array'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.items is None
    assert f'{type(var_4.additional_items).__module__}.{type(var_4.additional_items).__qualname__}' == 'typesystem.fields.Boolean'
    assert var_4.min_items is None
    assert var_4.max_items is None
    assert var_4.unique_items is False
    assert module_2.Array.errors == {'type': 'Must be an array.', 'null': 'May not be null.', 'empty': 'Must not be empty.', 'exact_items': 'Must have {min_items} items.', 'min_items': 'Must have at least {min_items} items.', 'max_items': 'Must have no more than {max_items} items.', 'additional_items': 'May not contain additional items.', 'unique_items': 'Items must be unique.'}
    var_5 = module_0.to_json_schema(var_4)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert module_0.TYPE_CONSTRAINTS == {'minItems', 'contains', 'additionalProperties', 'pattern', 'type', 'propertyNames', 'required', 'boolean_schema', 'maxProperties', 'exclusiveMinimum', 'dependencies', 'properties', 'minProperties', 'maxItems', 'additionalItems', 'minimum', 'exclusiveMaximum', 'maximum', 'maxLength', 'minLength', 'patternProperties', 'uniqueItems', 'multipleOf', 'items'}
    assert f'{type(module_0.definitions).__module__}.{type(module_0.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(module_0.definitions) == 1
    assert f'{type(module_0.JSONSchema).__module__}.{type(module_0.JSONSchema).__qualname__}' == 'typesystem.fields.Union'
    assert module_0.JSONSchema.title == ''
    assert module_0.JSONSchema.description == ''
    assert module_0.JSONSchema.allow_null is False
    assert module_0.JSONSchema.read_only is False
    assert f'{type(module_0.JSONSchema.any_of).__module__}.{type(module_0.JSONSchema.any_of).__qualname__}' == 'builtins.list'
    assert len(module_0.JSONSchema.any_of) == 2
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
    var_7 = module_0.from_json_schema(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert module_1.NeverMatch.errors == {'never': 'This never validates.'}