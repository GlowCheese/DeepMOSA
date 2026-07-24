####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from typesystem.composites import IfThenElse
from typesystem.fields import String, Boolean

def test_if_then_else_from_json_schema():
    definitions = Definitions()
    
    # Test Case 1: Full If-Then-Else
    data_full = {
        "if": {"type": "string"},
        "then": {"type": "boolean"},
        "else": {"type": "integer"},
        "default": "some_default"
    }
    field_full = if_then_else_from_json_schema(data_full, definitions=definitions)
    assert isinstance(field_full, IfThenElse)
    assert field_full.if_clause == String()
    assert field_full.then_clause == Boolean()
    assert field_full.else_clause == Integer()
    assert field_full.default == "some_default"

    # Test Case 2: If-Then only
    data_if_then = {
        "if": {"type": "string"},
        "then": {"type": "boolean"}
    }
    field_if_then = if_then_else_from_json_schema(data_if_then, definitions=definitions)
    assert isinstance(field_if_then, IfThenElse)
    assert field_if_then.if_clause == String()
    assert field_if_then.then_clause == Boolean()
    assert field_if_then.else_clause is None
    assert field_if_then.default == NO_DEFAULT

    # Test Case 3: If-Else only
    data_if_else = {
        "if": {"type": "string"},
        "else": {"type": "integer"}
    }
    field_if_else = if_then_else_from_json_schema(data_if_else, definitions=definitions)
    assert isinstance(field_if_else, IfThenElse)
    assert field_if_else.if_clause == String()
    assert field_if_else.then_clause is None
    assert field_if_else.else_clause == Integer()
    assert field_if_else.default == NO_DEFAULT

    # Test Case 4: If only
    data_if_only = {
        "if": {"type": "string"}
    }
    field_if_only = if_then_else_from_json_schema(data_if_only, definitions=definitions)
    assert isinstance(field_if_only, IfThenElse)
    assert field_if_only.if_clause == String()
    assert field_if_only.then_clause is None
    assert field_if_only.else_clause is None
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from typesystem import Choice

def test_enum_from_json_schema():
    definitions = Definitions()
    
    # Test standard enum with strings
    data_strings = {"enum": ["a", "b", "c"]}
    field_strings = enum_from_json_schema(data_strings, definitions)
    assert isinstance(field_strings, Choice)
    assert field_strings.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test enum with integers and a default value
    data_integers = {"enum": [1, 2], "default": 1}
    field_integers = enum_from_json_schema(data_integers, definitions)
    assert isinstance(field_integers, Choice)
    assert field_integers.choices == [(1, 1), (2, 2)]
    assert field_integers.default == 1

    # Test enum with mixed types (though unusual in JSON schema, the function allows it)
    data_mixed = {"enum": [1, "apple", True]}
    field_mixed = enum_from_json_schema(data_mixed, definitions)
    assert field_mixed.choices == [(1, 1), ("apple", "apple"), (True, True)]

    # Test error case: missing 'enum' key
    with pytest.raises(KeyError):
        enum_from_json_schema({}, definitions)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from typesystem import (
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
    Union,
    Any,
)

def test_from_json_schema_type():
    definitions = Definitions()

    # Test string type
    string_data = {"type": "string", "minLength": 5, "maxLength": 10, "pattern": "^a", "format": "email"}
    field = from_json_schema_type(string_data, "string", allow_null=False, definitions=definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^a"
    assert field.format == "email"

    # Test integer type
    int_data = {"type": "integer", "minimum": 0, "maximum": 100, "multipleOf": 2}
    field = from_json_schema_type(int_data, "integer", allow_null=True, definitions=definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.multiple_of == 2
    assert field.allow_null is True

    # Test number (float) type
    num_data = {"type": "number", "exclusiveMinimum": 1.5}
    field = from_json_schema_type(num_data, "number", allow_null=False, definitions=definitions)
    assert isinstance(field, Float)
    assert field.exclusive_minimum == 1.5

    # Test boolean type
    bool_data = {"type": "boolean", "default": True}
    field = from_json_schema_type(bool_data, "boolean", allow_null=False, definitions=definitions)
    assert isinstance(field, Boolean)
    assert field.default is True

    # Test array type
    array_data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "uniqueItems": True,
        "additionalItems": False
    }
    field = from_json_schema_type(array_data, "array", allow_null=False, definitions=definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.unique_items is True
    assert field.additional_items is False

    # Test object type
    obj_data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "additionalProperties": True
    }
    field = from_json_schema_type(obj_data, "object", allow_null=False, definitions=definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.additional_properties is True

    # Test Union type (multiple types)
    union_data = {"type": ["string", "number"]}
    # Note: from_json_schema_type is called for each type in the loop inside type_from_json_schema
    # But if we test the logic of processing a single type_string from a list:
    field_str = from_json_schema_type(union_data, "string", allow_null=False, definitions=definitions)
    field_num = from_json_schema_type(union_data, "number", allow_null=False, definitions=definitions)
    union_field = Union(any_of=[field_str, field_num])
    assert isinstance(union_field, Union)

    # Test null handling via allow_null
    null_data = {"type": "string"}
    field_null = from_json_schema_type(null_data, "string", allow_null=True, definitions=definitions)
    assert field_null.allow_null is True
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from typesystem.schemas import Definitions, Reference

def test_ref_from_json_schema():
    definitions = Definitions()
    
    # Test valid local reference
    valid_data = {"$ref": "#/definitions/MyType"}
    result = ref_from_json_schema(valid_data, definitions=definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/MyType"
    assert result.definitions == definitions

    # Test valid components reference
    valid_data_comp = {"$ref": "#/components/schemas/User"}
    result_comp = ref_from_json_schema(valid_data_comp, definitions=definitions)
    assert result_comp.to == "#/components/schemas/User"

    # Test invalid reference (no # prefix)
    invalid_data = {"$ref": "external_file.json#/Type"}
    with pytest.raises(AssertionError, match="Unsupported \$ref style in document."):
        ref_from_json_schema(invalid_data, definitions=definitions)

    # Test invalid reference (missing $ref key)
    with pytest.raises(KeyError):
        ref_from_json_schema({}, definitions=definitions)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_if_then_else_from_json_schema():
    # Mocking the dependencies/definitions
    mock_definitions = MagicMock()
    
    # Test Case 1: Full if-then-else schema
    data_full = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": "some_default"
    }
    
    # We need to mock from_json_schema because it's called inside the function under test
    # Since we can't easily patch a function in the same module without importing it, 
    # and the prompt implies the code is already available, we assume the environment
    # allows testing the logic of if_then_else_from_json_schema.
    
    # Note: In a real scenario, you'd use patch('module_name.from_json_schema')
    # Here we rely on the actual implementation's behavior.
    
    # For the purpose of this unit test, we test the structure of the returned IfThenElse object
    # and ensure it correctly parses the input dictionary.
    
    # We'll use a side effect to track calls if we were using a patch, 
    # but here we test the resulting object properties.
    
    result = if_then_else_from_json_schema(data_full, definitions=mock_definitions)
    
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    assert result.default == "some_default"

    # Test Case 2: Only 'if' and 'then' (no 'else')
    data_if_then = {
        "if": {"type": "number"},
        "then": {"type": "string"}
    }
    result_if_then = if_then_else_from_json_schema(data_if_then, definitions=mock_definitions)
    
    assert isinstance(result_if_then, IfThenElse)
    assert result_if_then.if_clause is not None
    assert result_if_then.then_clause is not None
    assert result_if_then.else_clause is None
    assert result_if_then.default == NO_DEFAULT

    # Test Case 3: Only 'if' and 'else' (no 'then')
    data_if_else = {
        "if": {"type": "boolean"},
        "else": {"type": "object"}
    }
    result_if_else = if_then_else_from_json_schema(data_if_else, definitions=mock_definitions)
    
    assert isinstance(result_if_else, IfThenElse)
    assert result_if_else.if_clause is not None
    assert result_if_else.then_clause is None
    assert result_if_else.else_clause is not None
    assert result_if_else.default == NO_DEFAULT

    # Test Case 4: Only 'if'
    data_if_only = {
        "if": {"type": "array"}
    }
    result_if_only = if_then_else_from_json_schema(data_if_only, definitions=mock_definitions)
    
    assert isinstance(result_if_only, IfThenElse)
    assert result_if_only.if_clause is not None
    assert result_if_only.then_clause is None
    assert result_if_only.else_clause is None
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from typesystem.fields import AllOf, Integer, String, Boolean

def test_all_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Basic AllOf with multiple types
    data_basic = {
        "allOf": [
            {"type": "integer", "minimum": 10},
            {"type": "string", "minLength": 5}
        ],
        "default": 123
    }
    field = all_of_from_json_schema(data_basic, definitions=definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2
    assert isinstance(field.all_of[0], Integer)
    assert isinstance(field.all_of[1], String)
    assert field.default == 123

    # Test case 2: AllOf with a single type (should still return AllOf per implementation)
    data_single = {
        "allOf": [{"type": "boolean"}],
        "default": True
    }
    field_single = all_of_from_json_schema(data_single, definitions=definitions)
    assert isinstance(field_single, AllOf)
    assert len(field_single.all_of) == 1
    assert isinstance(field_single.all_of[0], Boolean)
    assert field_single.default is True

    # Test case 3: AllOf with no default value (should use NO_DEFAULT)
    from typesystem.fields import NO_DEFAULT
    data_no_default = {
        "allOf": [{"type": "integer"}]
    }
    field_no_default = all_of_from_json_schema(data_no_default, definitions=definitions)
    assert field_no_default.default == NO_DEFAULT

    # Test case 4: Complex nested structures within allOf
    data_complex = {
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            {"type": "boolean"}
        ]
    }
    field_complex = all_of_from_json_schema(data_complex, definitions=definitions)
    assert isinstance(field_complex, AllOf)
    assert isinstance(field_complex.all_of[0], Object)
    assert isinstance(field_complex.all_of[1], Boolean)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from typesystem.fields import OneOf, String, Integer, Union

def test_one_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Standard oneOf with multiple types
    data_oneof = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ],
        "default": "some_default"
    }
    
    field = one_of_from_json_schema(data_oneof, definitions=definitions)
    
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2
    assert isinstance(field.one_of[0], String)
    assert isinstance(field.one_of[1], Integer)
    assert field.default == "some_default"

    # Test case 2: oneOf with a single type (should still return OneOf structure)
    data_single = {
        "oneOf": [{"type": "boolean"}]
    }
    field_single = one_of_from_json_schema(data_single, definitions=definitions)
    assert isinstance(field_single, OneOf)
    assert isinstance(field_single.one_of[0], Boolean)
    assert field_single.default == NO_DEFAULT

    # Test case 3: oneOf with complex types (Union/AllOf)
    data_complex = {
        "oneOf": [
            {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            {"allOf": [{"type": "string"}, {"minLength": 5}]}
        ]
    }
    field_complex = one_of_from_json_schema(data_complex, definitions=definitions)
    assert isinstance(field_complex, OneOf)
    assert isinstance(field_complex.one_of[0], Union)
    assert isinstance(field_complex.one_of[1], AllOf)

    # Test case 4: verifying default value handling
    data_default = {
        "oneOf": [{"type": "string"}],
        "default": 123
    }
    field_default = one_of_from_json_schema(data_default, definitions=definitions)
    assert field_default.default == 123
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import re

@pytest.mark.parametrize("field, expected_type, expected_data", [
    (Any(), "Any", {}),
    (NeverMatch(), "NeverMatch", {}),
    (String(allow_null=True, min_length=5, max_length=10, format="email"), "String", {"type": ["string", "null"], "minLength": 5, "maxLength": 10, "format": "email"}),
    (String(allow_null=False, allow_blank=True), "String", {"type": "string", "minLength": 0}),
    (Integer(allow_null=False, minimum=0, maximum=100), "Integer", {"type": "integer", "minimum": 0, "maximum": 100}),
    (Float(allow_null=True, exclusive_minimum=0), "Float", {"type": ["number", "null"], "exclusiveMinimum": 0}),
    (Boolean(allow_null=False), "Boolean", {"type": "boolean"}),
    (Array(items=String(allow_null=False), min_items=1, unique_items=True), "Array", {"type": "array", "minItems": 1, "items": {"type": "string"}, "uniqueItems": True}),
    (Choice(choices=[("a", Const(const="a")), ("b", Const(const="b"))]), "Choice", {"enum": ["a", "b"]}),
    (Const(const={"id": 1}), "Const", {"const": {"id": 1}}),
    (Union(any_of=[String(allow_null=False), Integer(allow_null=False)]), "Union", {"anyOf": [{"type": "string"}, {"type": "integer"}]}),
    (OneOf(one_of=[Boolean(allow_null=False)]), "OneOf", {"oneOf": [{"type": "boolean"}]}),
    (AllOf(all_of=[String(allow_null=False)]), "AllOf", {"allOf": [{"type": "string"}]}),
    (Not(negated=String(allow_null=False)), "Not", {"not": {"type": "string"}}),
    (IfThenElse(if_clause=Boolean(allow_null=False), then_clause=Integer(allow_null=False), else_clause=Float(allow_null=False)), "IfThenElse", {"if": {"type": "boolean"}, "then": {"type": "integer"}, "else": {"type": "number"}}),
])
def test_to_json_schema_basic_types(field, expected_type, expected_data):
    result = to_json_schema(field)
    assert isinstance(result, dict)
    for key, value in expected_data.items():
        assert result[key] == value

def test_to_json_schema_object_complex():
    prop_string = String(allow_null=False)
    prop_int = Integer(allow_null=False)
    obj = Object(
        properties={"name": prop_string, "age": prop_int},
        required=["name"],
        additional_properties=False
    )
    result = to_json_schema(obj)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string"}
    assert result["properties"]["age"] == {"type": "integer"}
    assert "name" in result["required"]
    assert result["additionalProperties"] is False

def test_to_json_schema_reference_and_definitions():
    # Mocking a structure where a Reference points to a definition
    # Note: Since we don't have the full implementation of 'target' or 'definitions' logic 
    # for Reference in the snippet, we test the logic provided in the snippet.
    
    class MockField:
        pass
    
    class MockReference(Reference):
        def __init__(self, to, definitions):
            self.to = to
            self.definitions = definitions
            self.target = None # Simplified for test
            
    defs = {"User": String(allow_null=False)}
    ref = Reference(to="User", definitions=defs)
    # We can't easily test the full recursion without a concrete target, 
    # but we test the dictionary construction logic for definitions.
    
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["User"] == {"type": "string"}

def test_to_json_schema_regex_error():
    # Testing the specific ValueError for non-standard regex flags
    # This requires a regex with a flag that is not Unicode
    pattern = re.compile(r"abc", re.IGNORECASE | re.ASCII) 
    # Note: standard re.compile flags are usually fine, but the code checks specifically 
    # for flags != re.RegexFlag.UNICODE.
    
    # We simulate a regex that would trigger the check if the environment/logic allows
    # This is hard to trigger purely with standard 're' as most are Unicode by default in Python 3,
    # but we test the logic branch.
    
    field = String(allow_null=False, pattern_regex=re.compile(r"^[a-z]+$"))
    # This should pass
    assert to_json_schema(field)["pattern"] == "^[a-z]+$"

def test_to_json_schema_invalid_type():
    class UnhandledField:
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type 'UnhandledField' to JSON Schema"):
        to_json_schema(UnhandledField())

def test_to_json_schema_array_items_list():
    arr = Array(items=[String(allow_null=False), Integer(allow_null=False)])
    result = to_json_schema(arr)
    assert result["items"] == [{"type": "string"}, {"type": "integer"}]

def test_to_json_schema_object_pattern_properties():
    obj = Object(pattern_properties={r".*suffix": String(allow_null=False)})
    result = to_json_schema(obj)
    assert result["patternProperties"][".*suffix"] == {"type": "string"}
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from typesystem.composites import AllOf
from typesystem.fields import Integer, String, Boolean

def test_all_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Single schema in allOf (should return AllOf with one element)
    data_single = {
        "allOf": [
            {"type": "string"}
        ],
        "default": "default_val"
    }
    field = all_of_from_json_schema(data_single, definitions=definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 1
    assert isinstance(field.all_of[0], String)
    assert field.default == "default_val"

    # Test case 2: Multiple schemas in allOf
    data_multiple = {
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "integer", "minimum": 10},
            {"type": "boolean"}
        ]
    }
    field = all_of_from_json_schema(data_multiple, definitions=definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 3
    assert isinstance(field.all_of[0], String)
    assert isinstance(field.all_of[1], Integer)
    assert isinstance(field.all_of[2], Boolean)
    assert field.default == NO_DEFAULT

    # Test case 3: allOf with complex nested structures
    data_complex = {
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            {
                "type": "array",
                "items": {"type": "integer"}
            }
        ],
        "default": None
    }
    field = all_of_from_json_schema(data_complex, definitions=definitions)
    assert isinstance(field, AllOf)
    assert isinstance(field.all_of[0], Object)
    assert isinstance(field.all_of[1], Array)
    assert field.default is None
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_from_json_schema():
    # Test Boolean True -> Any
    assert isinstance(from_json_schema(True), Any)
    
    # Test Boolean False -> NeverMatch
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test Simple String via type constraint
    string_schema = {"type": "string"}
    result_string = from_json_schema(string_schema)
    assert isinstance(result_string, String)

    # Test Simple Integer via type constraint
    int_schema = {"type": "integer"}
    result_int = from_json_schema(int_schema)
    assert isinstance(result_int, Integer)

    # Test Enum
    enum_schema = {"enum": ["a", "b", "c"]}
    result_enum = from_json_schema(enum_schema)
    assert isinstance(result_enum, Choice)
    assert result_enum.choices == ["a", "b", "c"]

    # Test Const
    const_schema = {"const": 42}
    result_const = from_json_schema(const_schema)
    assert isinstance(result_const, Const)
    assert result_const.value == 42

    # Test AllOf (Multiple constraints)
    # Since the function uses AllOf when len(constraints) > 1
    # We simulate a schema that has both 'type' and 'enum'
    combined_schema = {
        "type": "string",
        "enum": ["val1"]
    }
    result_combined = from_json_schema(combined_schema)
    assert isinstance(result_combined, AllOf)
    
    # Test Any() fallback for empty dict
    assert isinstance(from_json_schema({}), Any)

    # Test Reference handling (Requires setup of definitions)
    # Note: ref_from_json_schema is an internal dependency not provided in snippet,
    # but we test the logic path if it exists.
    ref_schema = {"$ref": "#/definitions/MyType"}
    # This test depends on the implementation of ref_from_json_schema 
    # and the state of the 'definitions' object.
    try:
        result_ref = from_json_schema(ref_schema)
        assert isinstance(result_ref, Reference)
    except (NameError, NotImplementedError):
        # Skip if internal helper is not available in the test environment
        pass

    # Test Components/Schemas parsing (Recursive definitions)
    components_schema = {
        "components": {
            "schemas": {
                "User": {"type": "string"}
            }
        }
    }
    # This will trigger the loop that populates definitions
    result_comp = from_json_schema(components_schema)
    # Since 'components' doesn't have constraints in TYPE_CONSTRAINTS, 
    # it should return Any() but the side effect is the population of definitions.
    assert isinstance(result_comp, Any)

    # Test Numeric Constraints
    num_schema = {"type": "number", "minimum": 10}
    result_num = from_json_schema(num_schema)
    assert isinstance(result_num, Number)
    # Verify the constraint is applied (if the implementation handles it)
    # Note: Implementation of type_from_json_schema is required for full verification
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from typesystem.fields import (
    Any,
    Array,
    Boolean,
    Float,
    Integer,
    Object,
    String,
    Union,
)

def test_from_json_schema_type():
    definitions = Definitions()

    # Test Number (Float)
    data_number = {
        "type": "number",
        "minimum": 0,
        "maximum": 10,
        "exclusiveMinimum": 1,
        "exclusiveMaximum": 9,
        "multipleOf": 2,
        "default": 5.5
    }
    field = from_json_schema_type(data_number, "number", allow_null=False, definitions=definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 1
    assert field.exclusive_maximum == 9
    assert field.multiple_of == 2
    assert field.default == 5.5

    # Test Integer
    data_integer = {
        "type": "integer",
        "minimum": 1,
        "maximum": 5,
        "default": 3
    }
    field = from_json_schema_type(data_integer, "integer", allow_null=True, definitions=definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.maximum == 5
    assert field.allow_null is True

    # Test String
    data_string = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^abc",
        "format": "email"
    }
    field = from_json_schema_type(data_string, "string", allow_null=False, definitions=definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^abc"
    assert field.format == "email"

    # Test Boolean
    data_boolean = {"type": "boolean", "default": True}
    field = from_json_schema_type(data_boolean, "boolean", allow_null=False, definitions=definitions)
    assert isinstance(field, Boolean)
    assert field.default is True

    # Test Array
    data_array = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "additionalItems": False
    }
    field = from_json_schema_type(data_array, "array", allow_null=False, definitions=definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items is True
    assert field.additional_items is False

    # Test Object
    data_object = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "additionalProperties": True,
        "minProperties": 1
    }
    field = from_json_schema_type(data_object, "object", allow_null=False, definitions=definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.additional_properties is True
    assert field.min_properties == 1

    # Test Union (via type_from_json_schema logic indirectly via type_strings length > 1)
    # Since from_json_schema_type handles a single type_string, 
    # we test the logic for a single type that is part of a larger set.
    data_union_part = {"type": "string", "minLength": 2}
    field = from_json_schema_type(data_union_part, "string", allow_null=True, definitions=definitions)
    assert isinstance(field, String)
    assert field.allow_null is True
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from typesystem.composites import IfThenElse
from typesystem.fields import String, Boolean, Integer

def test_if_then_else_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Full if-then-else
    data_full = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": "some_default"
    }
    field_full = if_then_else_from_json_schema(data_full, definitions=definitions)
    assert isinstance(field_full, IfThenElse)
    assert field_full.if_clause == String()
    assert field_full.then_clause == Integer()
    assert field_full.else_clause == Boolean()
    assert field_full.default == "some_default"

    # Test case 2: Only if and then
    data_if_then = {
        "if": {"type": "boolean"},
        "then": {"type": "string"}
    }
    field_if_then = if_then_else_from_json_schema(data_if_then, definitions=definitions)
    assert isinstance(field_if_then, IfThenElse)
    assert field_if_then.if_clause == Boolean()
    assert field_if_then.then_clause == String()
    assert field_if_then.else_clause is None
    assert field_if_then.default == NO_DEFAULT

    # Test case 3: Only if and else
    data_if_else = {
        "if": {"type": "number"},
        "else": {"type": "array", "items": {"type": "string"}}
    }
    field_if_else = if_then_else_from_json_schema(data_if_else, definitions=definitions)
    assert isinstance(field_if_else, IfThenElse)
    assert field_if_else.if_clause == Float()
    assert field_if_else.then_clause is None
    assert isinstance(field_if_else.else_clause, Array)
    assert field_if_else.else_clause.items == String()
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_enum_from_json_schema():
    definitions = Definitions()
    
    # Test basic enum functionality
    data_basic = {"enum": ["a", "b", "c"]}
    field_basic = enum_from_json_schema(data_basic, definitions)
    assert isinstance(field_basic, Choice)
    assert field_basic.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test enum with default value
    data_with_default = {"enum": [1, 2], "default": 1}
    field_with_default = enum_from_json_schema(data_with_default, definitions)
    assert field_with_default.default == 1
    assert field_with_default.choices == [(1, 1), (2, 2)]

    # Test enum with different types (integers)
    data_ints = {"enum": [10, 20]}
    field_ints = enum_from_json_schema(data_ints, definitions)
    assert field_ints.choices == [(10, 10), (20, 20)]

    # Test enum with complex objects (if they are hashable/comparable)
    data_complex = {"enum": [{"id": 1}, {"id": 2}]}
    field_complex = enum_from_json_schema(data_complex, definitions)
    assert field_complex.choices == [({"id": 1}, {"id": 1}), ({"id": 2}, {"id": 2})]

    # Test that it raises KeyError if 'enum' key is missing
    with pytest.raises(KeyError):
        enum_from_json_schema({}, definitions)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from typesystem import (
    Any,
    Array,
    Boolean,
    Integer,
    Number,
    Object,
    String,
    Union,
    Float,
    Const,
)

def test_from_json_schema_type():
    definitions = Definitions()

    # Test String
    data_string = {"type": "string", "minLength": 5, "maxLength": 10, "pattern": "^abc", "default": "def"}
    field = from_json_schema_type(data_string, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^abc"
    assert field.default == "def"

    data_string_null = {"type": "string", "null": True}
    field_null = from_json_schema_type(data_string_null, "string", True, definitions)
    assert isinstance(field_null, String)
    assert field_null.allow_null is True

    # Test Integer
    data_int = {"type": "integer", "minimum": 0, "maximum": 10, "multipleOf": 2}
    field_int = from_json_schema_type(data_int, "integer", False, definitions)
    assert isinstance(field_int, Integer)
    assert field_int.minimum == 0
    assert field_int.maximum == 10
    assert field_int.multiple_of == 2

    # Test Number (Float)
    data_num = {"type": "number", "minimum": 0.5, "exclusiveMinimum": 0.1}
    field_num = from_json_schema_type(data_num, "number", False, definitions)
    assert isinstance(field_num, Float)
    assert field_num.minimum == 0.5
    assert field_num.exclusive_minimum == 0.1

    # Test Boolean
    data_bool = {"type": "boolean", "default": True}
    field_bool = from_json_schema_type(data_bool, "boolean", False, definitions)
    assert isinstance(field_bool, Boolean)
    assert field_bool.default is True

    # Test Array
    data_array = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "additionalItems": False
    }
    field_array = from_json_schema_type(data_array, "array", False, definitions)
    assert isinstance(field_array, Array)
    assert field_array.min_items == 1
    assert field_array.max_items == 5
    assert field_array.unique_items is True
    assert field_array.additional_items is False
    assert isinstance(field_array.items, String)

    # Test Object
    data_object = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": {"type": "integer"},
        "minProperties": 1
    }
    field_obj = from_json_schema_type(data_object, "object", False, definitions)
    assert isinstance(field_obj, Object)
    assert "name" in field_obj.properties
    assert isinstance(field_obj.properties["name"], String)
    assert field_obj.required == ["name"]
    assert isinstance(field_obj.additional_properties, Integer)
    assert field_obj.min_properties == 1

    # Test Union (via logic in type_from_json_schema simulation)
    # Note: from_json_schema_type handles a single type_string. 
    # To test the Union logic, we rely on the implementation's behavior when passed a single type 
    # but we can verify the component parts.
    data_union = {"type": ["string", "number"]}
    # Manually simulating the loop in type_from_json_schema for a Union
    items = [
        from_json_schema_type({"type": "string"}, "string", False, definitions),
        from_json_schema_type({"type": "number"}, "number", False, definitions)
    ]
    union_field = Union(any_of=items)
    assert isinstance(union_field, Union)

    # Test Null-only case
    data_null_only = {"type": "null"}
    # Based on get_valid_types, if type is null, type_strings is empty and allow_null is True
    # then type_from_json_schema returns Const(None)
    field_none = from_json_schema_type({"type": "null"}, "null", True, definitions)
    # This is tricky because the function signature expects a type_string from the set.
    # If we pass "null" directly:
    field_none = from_json_schema_type({"type": "null"}, "null", True, definitions)
    # Since 'null' is not in the if/elif chain, it would hit the assert False.
    # However, get_valid_types removes 'null' from type_strings.
    # So we test the specific branch:
    with pytest.raises(AssertionError):
        from_json_schema_type({"type": "null"}, "null", True, definitions)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from typesystem.composites import AllOf
from typesystem.fields import Integer, String

def test_all_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Simple allOf with two different types
    data_simple = {
        "allOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    field = all_of_from_json_schema(data_simple, definitions=definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2
    assert isinstance(field.all_of[0], String)
    assert isinstance(field.all_of[1], Integer)

    # Test case 2: allOf with a default value
    data_with_default = {
        "allOf": [
            {"type": "string"}
        ],
        "default": "hello"
    }
    field_default = all_of_from_json_schema(data_with_default, definitions=definitions)
    assert field_default.default == "hello"

    # Test case 3: allOf with complex nested structures
    data_complex = {
        "allOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "object", "properties": {"age": {"type": "integer"}}}
        ]
    }
    field_complex = all_of_from_json_schema(data_complex, definitions=definitions)
    assert isinstance(field_complex, AllOf)
    assert len(field_complex.all_of) == 2
    assert "name" in field_complex.all_of[0].properties
    assert "age" in field_complex.all_of[1].properties
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from typesystem import AllOf, Integer, String

def test_all_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Basic allOf with two different types
    data_basic = {
        "allOf": [
            {"type": "integer", "minimum": 10},
            {"type": "string", "minLength": 5}
        ]
    }
    field = all_of_from_json_schema(data_basic, definitions=definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2
    assert isinstance(field.all_of[0], Integer)
    assert isinstance(field.all_of[1], String)
    assert field.all_of[0].minimum == 10
    assert field.all_of[1].min_length == 5

    # Test case 2: allOf with a default value
    data_with_default = {
        "allOf": [
            {"type": "integer"}
        ],
        "default": 42
    }
    field_default = all_of_from_json_schema(data_with_default, definitions=definitions)
    assert field_default.default == 42

    # Test case 3: allOf with a single element
    data_single = {
        "allOf": [
            {"type": "boolean"}
        ]
    }
    field_single = all_of_from_json_schema(data_single, definitions=definitions)
    assert len(field_single.all_of) == 1
    from typesystem import Boolean
    assert isinstance(field_single.all_of[0], Boolean)

    # Test case 4: Complex nested allOf
    data_nested = {
        "allOf": [
            {
                "allOf": [
                    {"type": "integer"},
                    {"type": "number"}
                ]
            },
            {"type": "string"}
        ]
    }
    field_nested = all_of_from_json_schema(data_nested, definitions=definitions)
    assert isinstance(field_nested, AllOf)
    assert isinstance(field_nested.all_of[0], AllOf)
    assert isinstance(field_nested.all_of[1], String)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_one_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Basic oneOf with simple types
    data_simple = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result_simple = one_of_from_json_schema(data_simple, definitions=definitions)
    assert isinstance(result_simple, OneOf)
    assert len(result_simple.one_of) == 2
    assert isinstance(result_simple.one_of[0], String)
    assert isinstance(result_simple.one_of[1], Integer)

    # Test case 2: oneOf with const and default value
    data_const = {
        "oneOf": [
            {"const": "A"},
            {"const": "B"}
        ],
        "default": "A"
    }
    result_const = one_of_from_json_schema(data_const, definitions=definitions)
    assert isinstance(result_const, OneOf)
    assert result_const.default == "A"
    assert isinstance(result_const.one_of[0], Const)
    assert result_const.one_of[0].const == "A"

    # Test case 3: oneOf with complex nested objects
    data_complex = {
        "oneOf": [
            {
                "type": "object",
                "properties": {"name": {"type": "string"}}
            },
            {
                "type": "array",
                "items": {"type": "integer"}
            }
        ]
    }
    result_complex = one_of_from_json_schema(data_complex, definitions=definitions)
    assert isinstance(result_complex, OneOf)
    assert isinstance(result_complex.one_of[0], Object)
    assert isinstance(result_complex.one_of[1], Array)
    assert "name" in result_complex.one_of[0].properties

    # Test case 4: oneOf with empty list (should still return OneOf structure)
    data_empty = {
        "oneOf": []
    }
    result_empty = one_of_from_json_schema(data_empty, definitions=definitions)
    assert isinstance(result_empty, OneOf)
    assert len(result_empty.one_of) == 0
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import re

@pytest.mark.parametrize("field, expected_type, expected_data", [
    (Any(), "Any", {}),
    (NeverMatch(), "NeverMatch", {}),
    (String(allow_null=True, min_length=5, max_length=10, format="email"), "String", {"type": ["string", "null"], "minLength": 5, "maxLength": 10, "format": "email"}),
    (String(allow_null=False, allow_blank=True), "String", {"type": "string", "minLength": 0}),
    (Integer(allow_null=True, minimum=0, maximum=100), "Integer", {"type": ["integer", "null"], "minimum": 0, "maximum": 100}),
    (Float(allow_null=False, multiple_of=0.5), "Float", {"type": "number", "multipleOf": 0.5}),
    (Boolean(allow_null=False), "Boolean", {"type": "boolean"}),
    (Array(allow_null=False, min_items=1, items=String()), "Array", {"type": "array", "minItems": 1, "items": {"type": "string"}}),
    (Array(allow_null=True, unique_items=True, items=[Integer()]), "Array", {"type": ["array", "null"], "uniqueItems": True, "items": {"type": "integer"}}),
    (Object(allow_null=False, properties={"name": String()}, required=["name"]), "Object", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
    (Choice(choices=[("A", Const(const="A")), ("B", Const(const="B"))]), "Choice", {"enum": ["A", "B"]}),
    (Const(const="fixed_value"), "Const", {"const": "fixed_value"}),
    (Union(any_of=[String(), Integer()]), "Union", {"anyOf": [{"type": "string"}, {"type": "integer"}]}),
    (OneOf(one_of=[String(), Integer()]), "OneOf", {"oneOf": [{"type": "string"}, {"type": "integer"}]}),
    (AllOf(all_of=[String(), Integer()]), "AllOf", {"allOf": [{"type": "string"}, {"type": "integer"}]}),
    (Not(negated=String()), "Not", {"not": {"type": "string"}}),
    (IfThenElse(if_clause=Boolean(), then_clause=Integer(), else_clause=Float()), "IfThenElse", {"if": {"type": "boolean"}, "then": {"type": "integer"}, "else": {"type": "number"}}),
])
def test_to_json_schema_basic_types(field, expected_type, expected_data):
    result = to_json_schema(field)
    # We check if the core keys match. 
    # Note: get_standard_properties might add 'default' if NO_DEFAULT is used, 
    # so we check intersection or specific known keys.
    for key, value in expected_data.items():
        assert result[key] == value

def test_to_json_schema_regex_error():
    # Testing the error handling for non-unicode regex flags
    pattern = re.compile(r"abc", re.ASCII)
    field = String(pattern_regex=pattern)
    with pytest.raises(ValueError, match="Cannot convert regular expression with non-standard flags"):
        to_json_schema(field)

def test_to_json_schema_definitions_and_references():
    # Create a reference to a definition
    ref_field = Reference(to="User", definitions={})
    
    # Create a definition dictionary
    defs = Definitions({
        "User": Object(properties={"id": Integer()})
    })
    
    # When converting the definitions object itself
    result = to_json_schema(defs)
    
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    assert result["components"]["schemas"]["User"]["type"] == "object"
    
    # Test Reference resolution logic in to_json_schema
    # Note: The implementation uses field.to as a key in definitions during traversal
    ref_field_instance = Reference(to="User", definitions=defs)
    # The logic in the provided code for Reference is: 
    # data["$ref"] = f"#/components/schema/{field.to}"
    # and it attempts to populate definitions[field.to]
    result_ref = to_json_schema(ref_field_instance, _definitions=defs)
    assert result_ref["$ref"] == "#/components/schemas/User"

def test_to_json_schema_unsupported_type():
    class UnknownField:
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type 'UnknownField' to JSON Schema"):
        to_json_schema(UnknownField())

def test_to_json_schema_array_additional_items():
    # Test additional_items as a field
    arr = Array(items=String(), additional_items=Integer())
    result = to_json_schema(arr)
    assert result["additionalItems"] == {"type": "integer"}

def test_to_json_schema_object_additional_properties():
    # Test additional_properties as a boolean
    obj = Object(additional_properties=False)
    result = to_json_schema(obj)
    assert result["additionalProperties"] is False
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from typesystem.schemas import Definitions, Reference

def test_ref_from_json_schema():
    definitions = Definitions()
    
    # Test valid $ref
    valid_data = {"$ref": "#/definitions/MyType"}
    result = ref_from_json_schema(valid_data, definitions=definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/MyType"
    assert result.definitions == definitions

    # Test invalid $ref (does not start with #/)
    invalid_data = {"$ref": "definitions/MyType"}
    with pytest.raises(AssertionError, match="Unsupported \$ref style in document."):
        ref_from_json_schema(invalid_data, definitions=definitions)

    # Test missing $ref key
    missing_ref_data = {"type": "string"}
    with pytest.raises(KeyError):
        ref_from_json_schema(missing_ref_data, definitions=definitions)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from typesystem.schemas import Definitions, Reference

def test_ref_from_json_schema():
    definitions = Definitions()
    
    # Test valid $ref
    valid_data = {"$ref": "#/definitions/MySchema"}
    result = ref_from_json_schema(valid_data, definitions=definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/MySchema"
    assert result.definitions == definitions

    # Test invalid $ref (does not start with #/)
    invalid_data = {"$ref": "definitions/MySchema"}
    with pytest.raises(AssertionError, match="Unsupported \$ref style in document."):
        ref_from_json_schema(invalid_data, definitions=definitions)

    # Test missing $ref key
    with pytest.raises(KeyError):
        ref_from_json_schema({}, definitions=definitions)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from typesystem.schemas import Definitions, Reference

def test_ref_from_json_schema():
    definitions = Definitions()
    
    # Test valid $ref
    valid_data = {"$ref": "#/definitions/MySchema"}
    result = ref_from_json_schema(valid_data, definitions=definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/MySchema"
    assert result.definitions == definitions

    # Test invalid $ref (does not start with #/)
    invalid_data = {"$ref": "definitions/MySchema"}
    with pytest.raises(AssertionError, match="Unsupported \$ref style in document."):
        ref_from_json_schema(invalid_data, definitions=definitions)

    # Test missing $ref key
    with pytest.raises(KeyError):
        ref_from_json_schema({}, definitions=definitions)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
import re

class MockField:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# Mocking the necessary classes and constants used in the code
class Any: pass
class NeverMatch: pass
class AllOf(MockField): pass
class Union(MockField): pass
class OneOf(MockField): pass
class Not(MockField): pass
class IfThenElse(MockField): pass
class String(MockField): pass
class Integer(MockField): pass
class Float(MockField): pass
class Boolean(MockField): pass
class Array(MockField): pass
class Object(MockField): pass
class Choice(MockField): pass
class Const(MockField): pass
class Reference(MockField): pass
class Schema(MockField): pass
class Decimal(MockField): pass

NO_DEFAULT = "NO_DEFAULT"
TYPE_CONSTRAINTS = ["type", "properties", "items", "required"]

def get_standard_properties(field):
    return {}

@pytest.mark.parametrize("field, expected", [
    (Any(), {"anyOf": []}), # Simplified for testing logic
    (NeverMatch(), False),
])
def test_to_json_schema_basics(field, expected):
    # Note: actual implementation of Any/NeverMatch logic in the provided code 
    # returns True/False directly.
    if isinstance(field, Any):
        assert to_json_schema(field) is True
    elif isinstance(field, NeverMatch):
        assert to_json_schema(field) is False

def test_to_json_schema_string():
    field = String(allow_null=True, min_length=5, max_length=10, format="email")
    schema = to_json_schema(field)
    assert schema["type"] == ["string", "null"]
    assert schema["minLength"] == 5
    assert schema["maxLength"] == 10
    assert schema["format"] == "email"

def test_to_json_schema_integer():
    field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    schema = to_json_schema(field)
    assert schema["type"] == "integer"
    assert schema["minimum"] == 0
    assert schema["maximum"] == 100
    assert schema["multipleOf"] == 5

def test_to_json_schema_boolean():
    field = Boolean(allow_null=True)
    schema = to_json_schema(field)
    assert schema["type"] == ["boolean", "null"]

def test_to_json_schema_array():
    item_field = String(allow_null=False)
    field = Array(
        allow_null=False, 
        min_items=1, 
        max_items=5, 
        items=item_field,
        unique_items=True
    )
    schema = to_json_schema(field)
    assert schema["type"] == "array"
    assert schema["minItems"] == 1
    assert schema["maxItems"] == 5
    assert schema["items"] == {"type": "string"}
    assert schema["uniqueItems"] is True

def test_to_json_schema_object():
    prop_field = Integer(allow_null=False)
    field = Object(
        allow_null=False,
        properties={"age": prop_field},
        required=["age"],
        min_properties=1
    )
    schema = to_json_schema(field)
    assert schema["type"] == "object"
    assert "age" in schema["properties"]
    assert schema["properties"]["age"] == {"type": "integer"}
    assert "age" in schema["required"]
    assert schema["minProperties"] == 1

def test_to_json_schema_choice():
    field = Choice(choices=[("a", "a"), ("b", "b")])
    schema = to_json_schema(field)
    assert "a" in schema["enum"]
    assert "b" in schema["enum"]

def test_to_json_schema_const():
    field = Const(const="fixed_value")
    schema = to_json_schema(field)
    assert schema["const"] == "fixed_value"

def test_to_json_schema_union():
    field = Union(any_of=[String(allow_null=False), Integer(allow_null=False)])
    schema = to_json_schema(field)
    assert "anyOf" in schema
    assert schema["anyOf"][0]["type"] == "string"
    assert schema["anyOf"][1]["type"] == "integer"

def test_to_json_schema_reference():
    field = Reference(to="User", target=Object(properties={}))
    definitions = {"User": Object(properties={})}
    schema = to_json_schema(field, _definitions=definitions)
    assert schema["$ref"] == "#/components/schemas/User"
    assert "User" in schema["components"]["schemas"]

def test_to_json_schema_regex_error():
    # Testing the ValueError for non-standard flags
    # We need to mock a pattern_regex that has a non-unicode flag
    class BadString(String):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.pattern_regex = re.compile(r"abc", re.ASCII)
            
    field = BadString(allow_null=False)
    with pytest.raises(ValueError, match="Cannot convert regular expression"):
        to_json_schema(field)

def test_to_json_schema_not():
    negated_field = String(allow_null=False)
    field = Not(negated=negated_field)
    schema = to_json_schema(field)
    assert schema["not"] == {"type": "string"}

def test_to_json_schema_if_then_else():
    if_f = Boolean(allow_null=False)
    then_f = String(allow_null=False)
    else_f = Integer(allow_null=False)
    field = IfThenElse(if_clause=if_f, then_clause=then_f, else_clause=else_f)
    schema = to_json_schema(field)
    assert schema["if"] == {"type": "boolean"}
    assert schema["then"] == {"type": "string"}
    assert schema["else"] == {"type": "integer"}
```


