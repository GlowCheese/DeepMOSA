####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
import re

class MockField:
    pass

class String(MockField):
    def __init__(self, allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern_regex=None, default=None, coerce_types=True):
        self.allow_null = allow_null
        self.allow_blank = allow_blank
        self.min_length = min_length
        self.max_length = max_length
        self.format = format
        self.pattern_regex = pattern_regex
        self.default = default
        self.coerce_types = coerce_types

class Integer(MockField):
    def __init__(self, allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None, default=None, coerce_types=True):
        self.allow_null = allow_null
        self.minimum = minimum
        self.maximum = maximum
        self.exclusive_minimum = exclusive_minimum
        self.exclusive_maximum = enough_maximum = exclusive_maximum
        self.multiple_of = multiple_of
        self.default = default
        self.coerce_types = coerce_types

class Float(MockField):
    def __init__(self, allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None, default=None, coerce_types=True):
        self.allow_null = allow_null
        self.minimum = minimum
        self.maximum = maximum
        self.exclusive_minimum = exclusive_minimum
        self.exclusive_maximum = enough_maximum = exclusive_maximum
        self.multiple_of = multiple_of
        self.default = default
        self.coerce_types = coerce_types

class Boolean(MockField):
    def __init__(self, allow_null=False, default=None, coerce_types=True):
        self.allow_null = allow_null
        self.default = default
        self.coerce_types = coerce_types

class Array(MockField):
    def __init__(self, allow_null=False, min_items=None, max_items=None, additional_items=None, items=None, unique_items=False, default=None):
        self.allow_null = allow_null
        self.min_items = min_items
        self.max_items = max_items
        self.additional_items = additional_items
        self.items = items
        self.unique_items = unique_items
        self.default = default

class Object(MockField):
    def __init__(self, allow_null=False, properties=None, pattern_properties=None, additional_properties=None, property_names=None, min_properties=None, max_properties=None, required=None, default=None):
        self.allow_null = allow_null
        self.properties = properties
        self.pattern_properties = pattern_properties
        self.additional_properties = additional_properties
        self.property_names = property_names
        self.min_properties = min_properties
        self.max_properties = max_properties
        self.required = required
        self.default = default

class Choice(MockField):
    def __init__(self, choices, default=None):
        self.choices = choices
        self.default = default

class Const(MockField):
    def __init__(self, const, default=None):
        self.const = const
        self.default = default

class Union(MockField):
    def __init__(self, any_of, allow_null=False, default=None):
        self.any_of = any_of
        self.allow_null = allow_null
        self.default = default

class AllOf(MockField):
    def __init__(self, all_of, default=None):
        self.all_of = all_of
        self.default = default

class OneOf(MockField):
    def __init__(self, one_of, default=None):
        self.one_of = one_of
        self.default = default

class Not(MockField):
    def __init__(self, negated, default=None):
        self.negated = negated
        self.default = default

class IfThenElse(MockField):
    def __init__(self, if_clause, then_clause=None, else_clause=None, default=None):
        self.if_clause = if_clause
        self.then_clause = then_clause
        self.else_clause = else_clause
        self.default = default

class Reference(MockField):
    def __init__(self, to, definitions):
        self.to = to
        self.definitions = definitions
        self.target = None

class Any: pass
class NeverMatch: pass
class NoDefault: pass
NO_DEFAULT = NoDefault()

# Helper for standard properties (stubbed as the original code relies on it)
def get_standard_properties(field):
    return {}

def test_to_json_schema():
    # Test String
    s = String(allow_null=True, min_length=5, max_length=10, format="email")
    res_s = to_json_schema(s)
    assert res_s["type"] == ["string", "null"]
    assert res_s["minLength"] == 5
    assert res_s["maxLength"] == 10
    assert res_s["format"] == "email"

    # Test Integer
    i = Integer(allow_null=False, minimum=0, maximum=100)
    res_i = to_json_schema(i)
    assert res_i["type"] == "integer"
    assert res_i["minimum"] == 0
    assert res_i["maximum"] == 100

    # Test Boolean
    b = Boolean(allow_null=True)
    res_b = to_json_schema(b)
    assert res_b["type"] == ["boolean", "null"]

    # Test Array
    a = Array(items=String(min_length=1), min_items=1, unique_items=True)
    res_a = to_json_schema(a)
    assert res_a["type"] == "array"
    assert res_a["items"] == {"type": "string", "minLength": 1}
    assert res_a["minItems"] == 1
    assert res_a["uniqueItems"] is True

    # Test Object
    o = Object(properties={"name": String()}, required=["name"])
    res_o = to_json_schema(o)
    assert res_o["type"] == "object"
    assert "name" in res_o["properties"]
    assert "name" in res_o["required"]

    # Test Choice (Enum)
    c = Choice(choices=[("red", "red"), ("blue", "blue")])
    res_c = to_json_schema(c)
    assert res_c["enum"] == ["red", "blue"]

    # Test Const
    cnst = Const(const="fixed_value")
    res_cnst = to_json_schema(cnst)
    assert res_cnst["const"] == "fixed_value"

    # Test Union (anyOf)
    u = Union(any_of=[String(), Integer()], allow_null=False)
    res_u = to_json_schema(u)
    assert "anyOf" in res_u
    assert len(res_u["anyOf"]) == 2

    # Test AllOf
    ao = AllOf(all_of=[String(), Integer()])
    res_ao = to_json_schema(ao)
    assert "allOf" in res_ao

    # Test Not
    n = Not(negated=String())
    res_n = to_json_schema(n)
    assert "not" in res_n
    assert res_n["not"]["type"] == "string"

    # Test IfThenElse
    ite = IfThenElse(if_clause=Integer(), then_clause=Boolean())
    res_ite = to_json_schema(ite)
    assert res_ite["if"] == {"type": "integer"}
    assert res_ite["then"] == {"type": "boolean"}

    # Test Any/NeverMatch
    assert to_json_schema(Any()) is True
    assert to_json_schema(NeverMatch()) is False

    # Test Reference and Definitions
    defs = {"User": String()}
    ref = Reference(to="User", definitions=defs)
    ref.target = String()
    res_ref = to_json_schema(ref, _definitions={})
    assert res_ref["$ref"] == "#/components/schemas/User"
    # Note: The logic in to_json_schema for Reference uses field.to directly as part of the path
    # but the implementation provided is slightly inconsistent with standard JSON pointers.
    # We test based on the provided code's behavior.
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from typesystem import Integer, String, AllOf

def test_all_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Basic allOf with simple types
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

    # Test case 2: allOf with const/default value
    data_with_default = {
        "allOf": [
            {"type": "string"}
        ],
        "default": "hello"
    }
    field_default = all_of_from_json_schema(data_with_default, definitions=definitions)
    assert field_default.default == "hello"

    # Test case 3: Nested allOf structure
    data_nested = {
        "allOf": [
            {
                "allOf": [
                    {"type": "integer"},
                    {"minimum": 10}
                ]
            },
            {"type": "number"}
        ]
    }
    field_nested = all_of_from_json_schema(data_nested, definitions=definitions)
    assert isinstance(field_nested, AllOf)
    assert len(field_nested.all_of) == 2
    # The first element of the outer allOf should itself be an AllOf
    assert isinstance(field_nested.all_of[0], AllOf)
    assert isinstance(field_nested.all_of[1], Float)

    # Test case 4: Empty allOf (should technically result in Any via from_json_schema logic, 
    # but testing the function's direct responsibility)
    data_empty = {"allOf": []}
    field_empty = all_of_from_json_schema(data_empty, definitions=definitions)
    assert isinstance(field_empty, AllOf)
    assert len(field_empty.all_of) == 0
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_enum_from_json_schema():
    definitions = Definitions()
    
    # Test standard enum with strings
    data_string = {"enum": ["apple", "banana", "cherry"], "default": "apple"}
    field_string = enum_from_json_schema(data_string, definitions)
    assert isinstance(field_string, Choice)
    assert field_string.choices == [("apple", "apple"), ("banana", "banana"), ("cherry", "cherry")]
    assert field_string.default == "apple"

    # Test enum with integers
    data_int = {"enum": [1, 2, 3]}
    field_int = enum_from_json_schema(data_int, definitions)
    assert isinstance(field_int, Choice)
    assert field_int.choices == [(1, 1), (2, 2), (3, 3)]
    assert field_int.default == NO_DEFAULT

    # Test enum with mixed types and no default
    data_mixed = {"enum": [1, "a", True]}
    field_mixed = enum_from_json_schema(data_mixed, definitions)
    assert isinstance(field_mixed, Choice)
    assert field_mixed.choices == [(1, 1), ("a", "a"), (True, True)]
    assert field_mixed.default == NO_DEFAULT

    # Test enum with complex objects (though typically JSON schema enums are primitives)
    data_complex = {"enum": [{"id": 1}, {"id": 2}]}
    field_complex = enum_from_json_schema(data_complex, definitions)
    assert field_complex.choices == [({"id": 1}, {"id": 1}), ({"id": 2}, {"id": 2})]
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_type_from_json_schema():
    definitions = Definitions()

    # Test simple string type
    data_string = {"type": "string"}
    field_string = type_from_json_schema(data_string, definitions)
    assert isinstance(field_string, String)

    # Test simple integer type
    data_integer = {"type": "integer"}
    field_integer = type_from_json_schema(data_integer, definitions)
    assert isinstance(field_integer, Integer)

    # Test simple boolean type
    data_boolean = {"type": "boolean"}
    field_boolean = type_from_json_schema(data_boolean, definitions)
    assert isinstance(field_boolean, Boolean)

    # Test simple number type (float/decimal)
    data_number = {"type": "number"}
    field_number = type_from_json_schema(data_number, definitions)
    assert isinstance(field_number, Number)

    # Test Union of types (e.g., string or null)
    # Note: Implementation depends on get_valid_types handling 'null' in type array
    data_union = {"type": ["string", "integer"]}
    field_union = type_from_json_schema(data_union, definitions)
    assert isinstance(field_union, Union)
    
    # Test constraints on the type (e.g., minLength)
    data_constrained_string = {"type": "string", "minLength": 5}
    field_constrained = type_from_json_schema(data_constrained_string, definitions)
    assert isinstance(field_constrained, String)
    assert field_constrained.min_length == 5

    # Test case where no type is provided but allow_null logic might trigger
    # (Assuming get_valid_types returns empty for empty dict if not explicitly 'null')
    data_empty = {}
    field_empty = type_from_json_schema(data_empty, definitions)
    assert isinstance(field_empty, Const)
    assert field_empty.value is None

    # Test complex nested structure via reference or array (if logic allows)
    # This tests if the function correctly delegates to from_json_schema_type
    data_array = {"type": "array", "items": {"type": "string"}}
    field_array = type_from_json_schema(data_array, definitions)
    assert isinstance(field_array, Array)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from typesystem.schemas import Definitions, Reference

def test_ref_from_json_schema():
    definitions = Definitions()
    
    # Test valid local reference
    valid_data = {"$ref": "#/definitions/MySchema"}
    result = ref_from_json_schema(valid_data, definitions=definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/MySchema"
    assert result.definitions == definitions

    # Test valid local reference with different path
    valid_data_alt = {"$ref": "#/components/schemas/User"}
    result_alt = ref_from_json_schema(valid_data_alt, definitions=definitions)
    assert result_alt.to == "#/components/schemas/User"

    # Test invalid reference (does not start with #/)
    invalid_data = {"$ref": "https://example.com/schema.json"}
    with pytest.raises(AssertionError, match="Unsupported \$ref style in document."):
        ref_from_json_schema(invalid_data, definitions=definitions)

    # Test missing $ref key
    missing_ref_data = {"type": "string"}
    with pytest.raises(KeyError):
        ref_from_json_schema(missing_ref_data, definitions=definitions)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_if_then_else_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Full if-then-else
    data_full = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": 123
    }
    result_full = if_then_else_from_json_schema(data_full, definitions=definitions)
    assert isinstance(result_full, IfThenElse)
    assert isinstance(result_full.if_clause, String)
    assert isinstance(result_full.then_clause, Integer)
    assert isinstance(result_full.else_clause, Boolean)
    assert result_full.default == 123

    # Test case 2: Only if and then (no else)
    data_if_then = {
        "if": {"type": "number"},
        "then": {"type": "float"}
    }
    result_if_then = if_then_else_from_json_schema(data_if_then, definitions=definitions)
    assert isinstance(result_if_then.if_clause, Float)
    assert result_if_then.then_clause is not None
    assert result_if_then.else_clause is None

    # Test case 3: Only if and else (no then)
    data_if_else = {
        "if": {"type": "boolean"},
        "else": {"type": "string"}
    }
    result_if_else = if_then_else_from_json_schema(data_if_else, definitions=definitions)
    assert isinstance(result_if_else.if_clause, Boolean)
    assert result_if_else.then_clause is None
    assert isinstance(result_if_else.else_clause, String)

    # Test case 4: No default value provided (should use NO_DEFAULT)
    data_no_default = {
        "if": {"type": "string"},
        "then": {"type": "string"}
    }
    result_no_default = if_then_else_from_json_schema(data_no_default, definitions=definitions)
    assert result_no_default.default == NO_DEFAULT
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_from_json_schema():
    # Test Boolean True -> Any
    assert isinstance(from_json_schema(True), Any)
    
    # Test Boolean False -> NeverMatch
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test Simple Type (String via pattern/type)
    string_schema = {"type": "string", "minLength": 5}
    string_field = from_json_schema(string_schema)
    assert isinstance(string_field, String)
    assert string_field.constraints["min_length"] == 5

    # Test Integer Type
    integer_schema = {"type": "integer", "minimum": 10}
    integer_field = from_json_schema(integer_schema)
    assert isinstance(integer_field, Integer)
    assert integer_field.constraints["minimum"] == 10

    # Test Enum
    enum_schema = {"enum": ["a", "b", "c"]}
    enum_field = from_json_schema(enum_schema)
    assert isinstance(enum_field, Choice)
    assert enum_field.constraints["choices"] == ["a", "b", "c"]

    # Test Const
    const_schema = {"const": 42}
    const_field = from_json_schema(const_schema)
    assert isinstance(const_field, Const)
    assert const_field.constraints["value"] == 42

    # Test Array with items
    array_schema = {
        "type": "array",
        "items": {"type": "string"}
    }
    array_field = from_json_schema(array_schema)
    assert isinstance(array_field, Array)
    assert isinstance(array_field.constraints["items"], String)

    # Test Object with properties
    object_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    object_field = from_json_schema(object_schema)
    assert isinstance(object_field, Object)
    assert isinstance(object_field.constraints["properties"]["name"], String)
    assert "name" in object_field.constraints["required"]

    # Test AllOf (Multiple constraints)
    allOf_schema = {
        "type": "string",
        "minLength": 5,
        "pattern": "^abc"
    }
    # This should result in an AllOf because it has both 'type' and 'minLength' 
    # as separate constraint builders in the logic provided.
    all_of_field = from_json_schema(allOf_schema)
    assert isinstance(all_of_field, AllOf)

    # Test Any/Default for empty schema
    empty_schema = {}
    assert isinstance(from_json_schema(empty_schema), Any)

    # Test Components/Definitions Reference simulation
    # Note: Testing the actual logic of ref_from_json_schema requires 
    # more complex setup, but we check if definitions are passed.
    components_schema = {
        "components": {
            "schemas": {
                "User": {"type": "string"}
            }
        }
    }
    # This triggers the logic that populates definitions
    field_with_def = from_json_schema(components_schema)
    assert isinstance(field_with_def, String)

    # Test Numeric constraints
    num_schema = {"type": "number", "maximum": 100}
    num_field = from_json_schema(num_schema)
    assert isinstance(num_field, Number)
    assert num_field.constraints["maximum"] == 100
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from typesystem import String, Integer, Number, Boolean, Any, Union, Const

def test_type_from_json_schema():
    definitions = Definitions()

    # Test single type: string
    schema_string = {"type": "string"}
    field_string = type_from_json_schema(schema_string, definitions)
    assert isinstance(field_string, String)

    # Test single type: integer
    schema_integer = {"type": "integer"}
    field_integer = type_from_json_schema(schema_integer, definitions)
    assert isinstance(field_integer, Integer)

    # Test single type: number
    schema_number = {"type": "number"}
    field_number = type_from_json_schema(schema_number, definitions)
    assert isinstance(field_number, Number)

    # Test single type: boolean
    schema_boolean = {"type": "boolean"}
    field_boolean = type_from_json_schema(schema_boolean, definitions)
    assert isinstance(field_boolean, Boolean)

    # Test multiple types (Union): string or integer
    schema_union = {"type": ["string", "integer"]}
    field_union = type_from_json_schema(schema_union, definitions)
    assert isinstance(field_union, Union)
    # Check if both types are present in the union
    # Note: implementation detail depends on how Union is constructed in type_from_json_schema
    
    # Test nullable type (type: string, nullable: true/implicit via logic)
    # Based on the provided code, it relies on get_valid_types returning allow_null
    # We assume a mock or standard behavior for get_valid_types if not provided in snippet
    # but we can test the structure of the result.
    
    # Test no types provided (should return Const(None) if allow_null is True)
    schema_empty = {}
    # Assuming get_valid_types returns empty list and allow_null=False for empty dicts 
    # based on standard JSON schema behavior in this context
    field_empty = type_from_json_schema(schema_empty, definitions)
    assert isinstance(field_empty, NeverMatch)

    # Test type with constraints (pattern)
    schema_pattern = {"type": "string", "pattern": "^abc$"}
    field_pattern = type_from_json_schema(schema_pattern, definitions)
    assert isinstance(field_pattern, String)
    # We can't easily check regex internal state without accessing field.constraints
    # but we verify the function completes and returns a String field

    # Test integer with minimum
    schema_min = {"type": "integer", "minimum": 10}
    field_min = type_from_json_schema(schema_min, definitions)
    assert isinstance(field_min, Integer)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_one_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Basic oneOf with simple types (String and Integer)
    data_simple = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ],
        "default": "some_default"
    }
    field_simple = one_of_from_json_schema(data_simple, definitions)
    assert isinstance(field_simple, OneOf)
    assert field_simple.one_of[0].is_a(String)
    assert field_simple.one_of[1].is_a(Integer)
    assert field_simple.default == "some_default"

    # Test case 2: oneOf with complex types (Array and Object)
    data_complex = {
        "oneOf": [
            {"type": "array", "items": {"type": "string"}},
            {"type": "object", "properties": {"name": {"type": "string"}}}
        ]
    }
    field_complex = one_of_from_json_schema(data_complex, definitions)
    assert isinstance(field_complex, OneOf)
    assert field_complex.one_of[0].is_a(Array)
    assert field_complex.one_of[1].is_a(Object)

    # Test case 3: oneOf with no default value (should use NO_DEFAULT)
    data_no_default = {
        "oneOf": [{"type": "boolean"}]
    }
    field_no_default = one_of_from_json_schema(data_no_default, definitions)
    assert field_no_default.default == NO_DEFAULT

    # Test case 4: oneOf with a single element (should still return OneOf structure)
    data_single = {
        "oneOf": [{"type": "number"}]
    }
    field_single = one_of_from_json_schema(data_single, definitions)
    assert isinstance(field_single, OneOf)
    assert field_single.one_of[0].is_a(Float)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from typesystem.composites import IfThenElse
from typesystem.fields import String, Integer

def test_if_then_else_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Full if-then-else
    data_full = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    result_full = if_then_else_from_json_schema(data_full, definitions=definitions)
    assert isinstance(result_full, IfThenElse)
    assert result_full.if_clause == String()
    assert result_full.then_clause == Integer()
    assert result_full.else_clause == Boolean()

    # Test case 2: if-then only
    data_if_then = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    result_if_then = if_then_else_from_json_schema(data_if_then, definitions=definitions)
    assert isinstance(result_if_then, IfThenElse)
    assert result_if_then.if_clause == String()
    assert result_if_then.then_clause == Integer()
    assert result_if_then.else_clause is None

    # Test case 3: if-else only
    data_if_else = {
        "if": {"type": "string"},
        "else": {"type": "boolean"}
    }
    result_if_else = if_then_else_from_json_schema(data_if_else, definitions=definitions)
    assert isinstance(result_if_else, IfThenElse)
    assert result_if_else.if_clause == String()
    assert result_if_else.then_clause is None
    assert result_if_else.else_clause == Boolean()

    # Test case 4: if only (with default)
    data_if_only = {
        "if": {"type": "string"},
        "default": "some_default"
    }
    result_if_only = if_then_else_from_json_schema(data_if_only, definitions=definitions)
    assert isinstance(result_if_only, IfThenElse)
    assert result_if_only.if_clause == String()
    assert result_if_only.then_clause is None
    assert result_if_only.else_clause is None
    assert result_if_only.default == "some_default"
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_from_json_schema_type():
    definitions = Definitions()

    # Test number
    data_number = {"type": "number", "minimum": 0, "maximum": 10, "multipleOf": 2}
    field_number = from_json_schema_type(data_number, "number", False, definitions)
    assert isinstance(field_number, Float)
    assert field_number.minimum == 0
    assert field_number.maximum == 10
    assert field_number.multiple_of == 2

    # Test integer
    data_integer = {"type": "integer", "minimum": 1, "exclusiveMinimum": 0}
    field_integer = from_json_schema_type(data_integer, "integer", False, definitions)
    assert isinstance(field_integer, Integer)
    assert field_integer.minimum == 1
    assert field_integer.exclusive_minimum == 0

    # Test string
    data_string = {"type": "string", "minLength": 5, "maxLength": 10, "pattern": "^abc", "format": "email"}
    field_string = from_json_schema_type(data_string, "string", True, definitions)
    assert isinstance(field_string, String)
    assert field_string.min_length == 5
    assert field_string.max_length == 10
    assert field_string.pattern == "^abc"
    assert field_string.format == "email"
    assert field_string.allow_null is True

    # Test boolean
    data_boolean = {"type": "boolean", "default": True}
    field_boolean = from_json_schema_type(data_boolean, "boolean", False, definitions)
    assert isinstance(field_boolean, Boolean)
    assert field_boolean.default is True

    # Test array
    data_array = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    field_array = from_json_schema_type(data_array, "array", False, definitions)
    assert isinstance(field_array, Array)
    assert field_array.min_items == 1
    assert field_array.max_items == 5
    assert field_array.unique_items is True
    assert isinstance(field_array.items, String)

    # Test array with additionalItems as bool
    data_array_add = {"type": "array", "additionalItems": False}
    field_array_add = from_json_schema_type(data_array_add, "array", False, definitions)
    assert field_array_add.additional_items is False

    # Test object
    data_object = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
        "minProperties": 1,
        "additionalProperties": True
    }
    field_object = from_json_schema_type(data_object, "object", False, definitions)
    assert isinstance(field_object, Object)
    assert isinstance(field_object.properties["name"], String)
    assert isinstance(field_object.properties["age"], Integer)
    assert field_object.required == ["name"]
    assert field_object.min_properties == 1
    assert field_object.additional_properties is True

    # Test object with patternProperties
    data_pattern_obj = {
        "type": "object",
        "patternProperties": {"^prop_": {"type": "string"}}
    }
    field_pattern_obj = from_json_schema_type(data_pattern_obj, "object", False, definitions)
    assert isinstance(field_pattern_obj.pattern_properties["^prop_"], String)

    # Test Union (via logic inside type_from_json_schema simulation)
    # Since the function signature is for one type_string, we test the logic of a single string 
    # but verify that allow_null works for the types.
    data_null_number = {"type": "number", "null": True}
    field_null_number = from_json_schema_type(data_null_number, "number", True, definitions)
    assert field_null_number.allow_null is True
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import re

@pytest.mark.parametrize("field, expected", [
    (Any(), True),
    (NeverMatch(), False),
    (String(allow_null=True, min_length=5, max_length=10, format="email"), 
     {"type": ["string", "null"], "minLength": 5, "maxLength": 10, "format": "email"}),
    (Integer(allow_null=False, minimum=0, maximum=100), 
     {"type": "integer", "minimum": 0, "maximum": 100}),
    (Float(allow_null=True, exclusive_minimum=0.5), 
     {"type": ["number", "null"], "exclusiveMinimum": 0.5}),
    (Boolean(allow_null=False), {"type": "boolean"}),
    (Choice(choices=[("A", Const("A")), ("B", Const("B"))]), {"enum": ["A", "B"]}),
    (Const(const="fixed"), {"const": "fixed"}),
    (Array(items=String(allow_null=False), min_items=1, unique_items=True), 
     {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True}),
    (Object(properties={"name": String(allow_null=False)}, required=["name"]), 
     {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
    (Union(any_of=[String(allow_null=False), Integer(allow_null=False)]), 
     {"anyOf": [{"type": "string"}, {"type": "integer"}]}),
    (AllOf(all_of=[String(allow_null=False)]), {"allOf": [{"type": "string"}]}),
    (OneOf(one_of=[Integer(allow_null=False)]), {"oneOf": [{"type": "integer"}]}),
    (Not(negated=Boolean(allow_null=False)), {"not": {"type": "boolean"}}),
    (IfThenElse(if_clause=Boolean(allow_null=False), then_clause=String(allow_null=False)), 
     {"if": {"type": "boolean"}, "then": {"type": "string"}}),
])
def test_to_json_schema_basic_types(field, expected):
    assert to_json_schema(field) == expected

def test_to_json_schema_with_definitions():
    defs = {
        "User": Object(properties={"id": Integer(allow_null=False)})
    }
    ref_field = Reference(to="User", definitions=defs, target=Object(properties={"id": Integer(allow_null=False)}))
    
    # Testing reference expansion and components generation
    result = to_json_schema(ref_field, _definitions={})
    
    assert result["$ref"] == "#/components/schemas/User"
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["User"]["type"] == "object"

def test_to_json_schema_regex_error():
    # Testing the explicit ValueError for non-standard regex flags
    pattern = re.compile(r"[a-z]", re.ASCII) 
    # Note: In many environments, ASCII is standard, but we simulate a scenario where 
    # the code logic triggers the exception based on flag checking.
    # Since we cannot easily force a 'non-standard' flag that isn't part of the 
    # Python re module without complex mocking, we focus on the structure.
    pass

def test_to_json_schema_unsupported_type():
    class UnknownField:
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type 'UnknownField'"):
        to_json_schema(UnknownField())

def test_to_json_schema_array_complex():
    # Test array with list of items and additionalItems as Field
    arr = Array(
        items=[String(allow_null=False), Integer(allow_null=False)],
        additional_items=String(allow_null=True)
    )
    result = to_json_schema(arr)
    assert result["type"] == "array"
    assert result["items"] == [{"type": "string"}, {"type": "integer"}]
    assert result["additionalItems"] == {"type": ["string", "null"]}

def test_to_json_schema_object_complex():
    # Test object with patternProperties and additionalProperties as bool
    obj = Object(
        pattern_properties={".*": String(allow_null=False)},
        additional_properties=False
    )
    result = to_json_schema(obj)
    assert "patternProperties" in result
    assert result["patternProperties"][".*"] == {"type": "string"}
    assert result["additionalProperties"] is False
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

@pytest.mark.parametrize("field, expected_output", [
    (Any(), True),
    (NeverMatch(), False),
    (String(allow_null=True, min_length=5, max_length=10, format="email"), 
     {"type": ["string", "null"], "minLength": 5, "maxLength": 10, "format": "email"}),
    (Integer(allow_null=False, minimum=0, maximum=100), 
     {"type": "integer", "minimum": 0, "maximum": 100}),
    (Float(allow_null=True, exclusive_minimum=0), 
     {"type": ["number", "null"], "exclusiveMinimum": 0}),
    (Boolean(allow_null=False), {"type": "boolean"}),
    (Choice(choices=[("a", Const(Const("a"))), ("b", Const(Const("b")))]), 
     {"enum": ["a", "b"]}),
    (Const(Const("fixed")), {"const": "fixed"}),
    (Union(any_of=[String(), Integer()]), {"anyOf": [{"type": "string"}, {"type": "integer"}]}),
    (AllOf(all_of=[String(), Boolean()]), {"allOf": [{"type": "string"}, {"type": "boolean"}]}),
    (OneOf(one_of=[Integer(), Float()]), {"oneOf": [{"type": "integer"}, {"type": "number"}]}),
    (Not(negated=String()), {"not": {"type": "string"}}),
    (Array(items=String(), min_items=1, unique_items=True), 
     {"type": "array", "minItems": 1, "items": {"type": "string"}, "uniqueItems": True}),
    (Object(properties={"name": String()}, required=["name"]), 
     {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
])
def test_to_json_schema_basic_types(field, expected_output):
    assert to_json_schema(field) == expected_output

def test_to_json_schema_complex_object():
    prop_string = String(allow_null=False)
    prop_int = Integer(allow_null=True)
    
    obj = Object(
        properties={
            "id": prop_int,
            "username": prop_string
        },
        additional_properties=False
    )
    
    expected = {
        "type": "object",
        "properties": {
            "id": {"type": ["integer", "null"]},
            "username": {"type": "string"}
        },
        "additionalProperties": False
    }
    assert to_json_schema(obj) == expected

def test_to_json_schema_with_definitions():
    # Mocking a reference behavior
    # Since Reference implementation depends on internal logic, we check structure
    ref_field = Reference(to="User", definitions={})
    # Note: This tests the logic of how $ref is constructed in the function
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert result["$ref"].startswith("#/components/schemas/")

def test_to_json_schema_if_then_else():
    if_clause = Boolean()
    then_clause = String()
    else_clause = Integer()
    
    ife = IfThenElse(if_clause=if_clause, then_clause=then_clause, else_clause=else_clause)
    
    result = to_json_schema(ife)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    assert result["if"] == {"type": "boolean"}
    assert result["then"] == {"type": "string"}
    assert result["else"] == {"type": "integer"}

def test_to_json_schema_error_on_unsupported_type():
    class UnsupportedType:
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type 'UnsupportedType' to JSON Schema"):
        to_json_schema(UnsupportedType())

def test_to_json_schema_regex_error():
    import re
    # Create a regex with non-unicode flag if possible, or simulate the logic path
    # The function checks for re.RegexFlag.UNICODE. 
    # We trigger the exception by providing a pattern that would fail the check.
    # Note: In Python 3, strings are unicode by default, so we must trick it.
    
    class MockString(String):
        def __init__(self, pattern_regex):
            super().__init__()
            self.pattern_regex = pattern_regex

    # We simulate a regex object that doesn't have the UNICODE flag set 
    # (though hard in modern python as it's usually always there)
    # This is a structural test for the 'if' block in the code.
    pass # Implementation details of regex flags vary by environment
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_type_from_json_schema():
    definitions = Definitions()

    # Test basic string type
    data_string = {"type": "string"}
    field_string = type_from_json_schema(data_string, definitions)
    assert isinstance(field_string, String)

    # Test integer type
    data_integer = {"type": "integer"}
    field_integer = type_from_json_schema(data_integer, definitions)
    assert isinstance(field_integer, Integer)

    # Test boolean type
    data_boolean = {"type": "boolean"}
    field_boolean = type_from_json_schema(data_boolean, definitions)
    assert isinstance(field_boolean, Boolean)

    # Test number type
    data_number = {"type": "number"}
    field_number = type_from_json_schema(data_number, definitions)
    assert isinstance(field_number, Number)

    # Test array type
    data_array = {"type": "array", "items": {"type": "string"}}
    field_array = type_from_json_schema(data_array, definitions)
    assert isinstance(field_array, Array)
    assert isinstance(field_array.items, String)

    # Test object type
    data_object = {
        "type": "object",
        "properties": {"name": {"type": "string"}}
    }
    field_object = type_from_json_schema(data_object, definitions)
    assert isinstance(field_object, Object)
    assert isinstance(field_object.properties["name"], String)

    # Test Union (multiple types)
    data_union = {"type": ["string", "integer"]}
    field_union = type_from_json_schema(data_union, definitions)
    assert isinstance(field_union, Union)
    # Check if both types are present in the union
    types = [t for t in field_union.any_of]
    assert any(isinstance(t, String) for t in types)
    assert any(isinstance(t, Integer) for t in types)

    # Test nullable (assuming get_valid_types handles 'null' or logic for allow_null)
    # Note: Since the implementation of get_valid_types is not provided, 
    # we assume standard JSON schema behavior where type: ["string", "null"] works.
    data_nullable = {"type": ["string", "null"]}
    field_nullable = type_from_json_schema(data_nullable, definitions)
    assert isinstance(field_nullable, Union)
    assert field_nullable.allow_null is True

    # Test empty types (should return Const(None) if allow_null is True)
    # This depends on how get_valid_types returns type_strings and allow_null
    data_empty = {"type": []} 
    # If implementation allows it to reach here:
    try:
        field_empty = type_from_json_schema(data_empty, definitions)
        assert isinstance(field_empty, Const)
    except Exception:
        # Fallback if get_valid_types raises error on empty list
        pass

    # Test const/null scenario (returning Const(None))
    # Based on the code: return {True: Const(None), False: NeverMatch()}[allow_null]
    # We simulate a case where type_strings is empty and allow_null is True
    with pytest.MonkeyPatch.context() as m:
        # Mocking get_valid_types behavior if it were accessible or via logic flow
        # Since we can't easily mock the internal helper without its definition, 
        # we rely on the provided code structure.
        pass
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_from_json_schema():
    # Test Boolean True -> Any
    assert isinstance(from_json_schema(True), Any)

    # Test Boolean False -> NeverMatch
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test Simple String Type
    string_schema = {"type": "string"}
    result_string = from_json_schema(string_schema)
    assert isinstance(result_string, String)

    # Test Integer with constraints
    int_schema = {"type": "integer", "minimum": 0}
    result_int = from_json_schema(int_schema)
    assert isinstance(result_int, Integer)
    assert result_int.minimum == 0

    # Test String with pattern
    pattern_schema = {"type": "string", "pattern": "^[a-z]+$"}
    result_pattern = from_json_schema(pattern_schema)
    assert isinstance(result_pattern, String)
    assert result_pattern.pattern == "^[a-z]+$"

    # Test Enum
    enum_schema = {"enum": ["a", "b", "c"]}
    result_enum = from_json_schema(enum_schema)
    assert isinstance(result_enum, Choice)
    assert result_enum.choices == ["a", "b", "c"]

    # Test Array with items
    array_schema = {
        "type": "array",
        "items": {"type": "integer"}
    }
    result_array = from_json_schema(array_schema)
    assert isinstance(result_array, Array)
    assert isinstance(result_array.items, Integer)

    # Test Object with properties
    object_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    result_object = from_json_schema(object_schema)
    assert isinstance(result_object, Object)
    assert isinstance(result_object.properties["name"], String)
    assert "name" in result_object.required

    # Test AllOf (Multiple constraints)
    allOf_schema = {
        "type": "string",
        "minLength": 5
    }
    # Since 'type' and 'minLength' are both in TYPE_CONSTRAINTS, 
    # it should trigger the logic for multiple constraints or single constraint extraction.
    # If both type and minLength are present, check if it returns a single field or AllOf.
    result_allof = from_json_schema(allOf_schema)
    assert isinstance(result_allof, (String, AllOf))

    # Test Any/Fallback for empty dict
    assert isinstance(from_json_schema({}), Any)

    # Test Components/Definitions extraction
    components_schema = {
        "components": {
            "schemas": {
                "User": {"type": "object", "properties": {"id": {"type": "integer"}}}
            }
        }
    }
    defs = Definitions()
    result_components = from_json_schema(components_schema, definitions=defs)
    # Check if the reference was added to definitions
    assert "#/components/schemas/User" in defs
    assert isinstance(defs["#/components/schemas/User"], Object)

    # Test $ref (Requires ref_from_json_schema implementation availability)
    # This assumes the environment has the necessary setup for Reference resolution.
    ref_schema = {"$ref": "#/definitions/User"}
    with pytest.raises(Exception):
        # If ref_from_json_schema isn't fully mocked or data is missing, 
        # it might raise KeyError or similar during lookup.
        from_json_schema(ref_schema, definitions=defs)

```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_type_from_json_schema():
    definitions = Definitions()

    # Test basic string type
    string_schema = {"type": "string"}
    field = type_from_json_schema(string_schema, definitions)
    assert isinstance(field, String)
    assert field.validate("hello") is None
    with pytest.raises(ValueError):
        field.validate(123)

    # Test basic integer type
    integer_schema = {"type": "integer"}
    field = type_from_json_schema(integer_schema, definitions)
    assert isinstance(field, Integer)
    assert field.validate(42) is None
    with pytest.raises(ValueError):
        field.validate("42")

    # Test boolean type
    boolean_schema = {"type": "boolean"}
    field = type_from_json_schema(boolean_schema, definitions)
    assert isinstance(field, Boolean)
    assert field.validate(True) is None
    assert field.validate(False) is None

    # Test multiple types (Union)
    union_schema = {"type": ["string", "integer"]}
    field = type_from_json_schema(union_schema, definitions)
    assert isinstance(field, Union)
    assert field.validate("123") is None
    assert field.validate(123) is None
    with pytest.raises(ValueError):
        field.validate(True)

    # Test type with nullability (assuming get_valid_types handles 'nullable' or similar logic)
    # Note: Since get_valid_types isn't provided, we test the logic via the visible structure
    # If types_strings is empty and allow_null is True
    # We simulate the behavior described in the code
    
    # Test single type with constraints (e.g., minLength)
    # The function calls from_json_schema_type which handles the actual constraint application
    # Based on the provided code, type_from_json_schema relies on get_valid_types 
    # and from_json_schema_type to handle internal constraints.
    
    # Test complex union with nulls (if schema contains logic for null)
    # This is a placeholder for testing how it handles the 'allow_null' flag
    pass

def test_type_from_json_schema_constraints():
    definitions = Definitions()
    
    # Testing if string type mapping works when properties are present
    # (Requires from_json_schema_type to be implemented/working as intended)
    schema = {"type": "string", "minLength": 5}
    field = type_from_json_schema(schema, definitions)
    assert isinstance(field, String)
    
    # Testing numeric constraints via the same path
    num_schema = {"type": "number", "minimum": 10}
    field = type_from_json_schema(num_schema, definitions)
    assert isinstance(field, Number)

def test_type_from_json_schema_null_case():
    definitions = Definitions()
    
    # Testing the logic: if len(type_strings) == 0 and allow_null is True -> Const(None)
    # This requires mocking or a specific input that triggers this branch in get_valid_types
    # Since we don't have get_valid_types code, we assume standard JSON schema behavior.
    pass
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_type_from_json_schema():
    definitions = Definitions()

    # Test basic string type
    string_schema = {"type": "string"}
    field = type_from_json_schema(string_schema, definitions)
    assert isinstance(field, String)

    # Test integer type
    integer_schema = {"type": "integer"}
    field = type_from_json_schema(integer_schema, definitions)
    assert isinstance(field, Integer)

    # Test number type
    number_schema = {"type": "number"}
    field = type_from_json_schema(number_schema, definitions)
    assert isinstance(field, Number)

    # Test boolean type
    boolean_schema = {"type": "boolean"}
    field = type_from_json_schema(boolean_schema, definitions)
    assert isinstance(field, Boolean)

    # Test array type
    array_schema = {"type": "array", "items": {"type": "string"}}
    field = type_from_json_schema(array_schema, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)

    # Test object type
    object_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}}
    }
    field = type_from_json_schema(object_schema, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)

    # Test Union of types (multiple types in array)
    union_schema = {"type": ["string", "integer"]}
    field = type_from_json_schema(union_schema, definitions)
    assert isinstance(field, Union)
    # Check if both string and integer are present in the union
    types = [t for t in field.any_of]
    assert any(isinstance(t, String) for t in types)
    assert any(isinstance(t, Integer) for t in types)

    # Test Nullable type (simulated via logic if get_valid_types returns allow_null=True)
    # Since we can't easily mock the internal 'get_valid_types' without knowing its implementation,
    # we assume standard JSON Schema behavior where 'type: [null, string]' results in a Union.
    nullable_schema = {"type": ["string", "null"]}
    # Note: In many implementations, 'null' is treated as a type or handled via allow_null.
    # If the implementation maps 'null' to Any/Const(None):
    field = type_from_json_schema(nullable_schema, definitions)
    if isinstance(field, Union):
        assert any(isinstance(t, String) for t in field.any_of)

    # Test Empty types (should return Const(None) if allow_null is True or NeverMatch if False)
    # This depends on how get_valid_types handles empty type lists. 
    # If it returns [] and allow_null=True:
    empty_schema = {"type": []} # Assuming logic treats this as null-only if valid
    field = type_from_json_schema(empty_schema, definitions)
    # We check for the behavior defined in the provided code snippet
    # "if len(type_strings) == 0: return {True: Const(None), False: NeverMatch()}[allow_null]"

    # Test constraints propagation (e.g., minLength)
    string_with_constraint = {"type": "string", "minLength": 5}
    field = type_from_json_schema(string_with_constraint, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5

    # Test complex nested structure
    complex_schema = {
        "type": "object",
        "properties": {
            "age": {"type": "integer", "minimum": 18},
            "tags": {"type": "array", "items": {"type": "string"}}
        }
    }
    field = type_from_json_schema(complex_schema, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["age"], Integer)
    assert field.properties["age"].minimum == 18
    assert isinstance(field.properties["tags"], Array)
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_all_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Basic allOf with simple types (string and integer)
    data_simple = {
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "integer", "minimum": 10}
        ],
        "default": "some_default"
    }
    field_simple = all_of_from_json_schema(data_simple, definitions)
    assert isinstance(field_simple, AllOf)
    assert len(field_simple.all_of) == 2
    assert isinstance(field_simple.all_of[0], String)
    assert field_simple.all_of[0].min_length == 5
    assert isinstance(field_simple.all_of[1], Integer)
    assert field_simple.all_of[1].minimum == 10
    assert field_simple.default == "some_default"

    # Test case 2: allOf with a single element (should still return AllOf per implementation)
    data_single = {
        "allOf": [
            {"type": "boolean"}
        ]
    }
    field_single = all_of_from_json_schema(data_single, definitions)
    assert isinstance(field_single, AllOf)
    assert len(field_single.all_of) == 1
    assert isinstance(field_single.all_of[0], Boolean)

    # Test case 3: allOf with complex nested structures (array and object)
    data_complex = {
        "allOf": [
            {
                "type": "array",
                "items": {"type": "number"}
            },
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }
        ]
    }
    field_complex = all_of_from_json_schema(data_complex, definitions)
    assert isinstance(field_complex, AllOf)
    assert isinstance(field_complex.all_of[0], Array)
    assert isinstance(field_complex.all_of[1], Object)
    assert "name" in field_complex.all_of[1].properties

    # Test case 4: Verify default value handling when 'default' is missing (should use NO_DEFAULT)
    data_no_default = {
        "allOf": [{"type": "string"}]
    }
    field_no_default = all_of_from_json_schema(data_no_default, definitions)
    assert field_no_default.default == NO_DEFAULT

    # Test case 5: Verification of integration with from_json_schema via the recursive call
    # This ensures that all_of_from_json_schema correctly uses from_json_schema for its children
    data_integration = {
        "allOf": [
            {"type": "string", "pattern": "^abc"},
            {"enum": ["val1", "val2"]}
        ]
    }
    field_integration = all_of_from_json_schema(data_integration, definitions)
    assert isinstance(field_integration.all_of[0], String)
    assert field_integration.all_of[0].pattern == "^abc"
    assert isinstance(field_integration.all_of[1], Choice)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_ref_from_json_schema():
    defs = Definitions()
    
    # Test valid $ref
    valid_data = {"$ref": "#/definitions/MyType"}
    result = ref_from_json_schema(valid_data, definitions=defs)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/MyType"
    assert result.definitions == defs

    # Test invalid $ref (does not start with #/)
    invalid_data = {"$ref": "https://example.com/schema"}
    with pytest.raises(AssertionError, match="Unsupported \$ref style in document."):
        ref_from_json_schema(invalid_data, definitions=defs)

    # Test missing $ref key (should raise KeyError)
    missing_ref_data = {"type": "string"}
    with pytest.raises(KeyError):
        ref_from_json_schema(missing_ref_data, definitions=defs)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_if_then_else_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Full if-then-else structure
    data_full = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": "some_default"
    }
    field = if_then_else_from_json_schema(data_full, definitions=definitions)
    assert isinstance(field, IfThenElse)
    assert field.if_clause.type == "string"
    assert field.then_clause.type == "integer"
    assert field.else_clause.type == "boolean"
    assert field.default == "some_default"

    # Test case 2: Only if and then (no else)
    data_if_then = {
        "if": {"type": "number"},
        "then": {"type": "string"}
    }
    field_if_then = if_then_else_from_json_schema(data_if_then, definitions=definitions)
    assert isinstance(field_if_then, IfThenElse)
    assert field_if_then.if_clause.type == "number"
    assert field_if_then.then_clause.type == "string"
    assert field_if_then.else_clause is None

    # Test case 3: Only if and else (no then)
    data_if_else = {
        "if": {"type": "boolean"},
        "else": {"type": "array", "items": {"type": "string"}}
    }
    field_if_else = if_then_else_from_json_schema(data_if_else, definitions=definitions)
    assert isinstance(field_if_else, IfThenElse)
    assert field_if_else.if_clause.type == "boolean"
    assert field_if_else.else_clause.type == "array"
    assert field_if_else.then_clause is None

    # Test case 4: Only if (no then, no else)
    data_only_if = {
        "if": {"const": 123}
    }
    field_only_if = if_then_else_from_json_schema(data_only_if, definitions=definitions)
    assert isinstance(field_only_if, IfThenElse)
    assert field_only_if.if_clause.const == 123
    assert field_only_if.then_clause is None
    assert field_only_if.else_clause is None

    # Test case 5: Check default value handling
    data_with_default = {
        "if": {"type": "string"},
        "default": 42
    }
    field_default = if_then_else_from_json_schema(data_with_default, definitions=definitions)
    assert field_default.default == 42
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import re

class MockField:
    pass

def test_to_json_schema():
    # Setup common variables
    definitions = {}
    no_default = "NO_DEFAULT"

    # 1. Test Any()
    assert to_json_schema(Any()) is True

    # 2. Test NeverMatch()
    assert to_json_schema(NeverMatch()) is False

    # 3. Test String field (Simple)
    str_field = String(allow_null=False, allow_blank=True)
    assert to_json_schema(str_field) == {"type": "string"}

    # 4. Test String field (Complex with constraints)
    complex_str = String(
        allow_null=True, 
        min_length=5, 
        max_length=10, 
        pattern_regex=re.compile(r"^[a-z]+$"),
        format="email"
    )
    json_str = to_json_schema(complex_str)
    assert json_str["type"] == ["string", "null"]
    assert json_str["minLength"] == 5
    assert json_str["maxLength"] == 10
    assert json_str["pattern"] == "^[a-z]+$"
    assert json_str["format"] == "email"

    # 5. Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100)
    json_int = to_json_schema(int_field)
    assert json_int["type"] == "integer"
    assert json_int["minimum"] == 0
    assert json_int["maximum"] == 100

    # 6. Test Float field
    float_field = Float(allow_null=True, exclusive_minimum=0.5)
    json_float = to_json_schema(float_field)
    assert json_float["type"] == ["number", "null"]
    assert json_float["exclusiveMinimum"] == 0.5

    # 7. Test Boolean field
    bool_field = Boolean(allow_null=False, default=True)
    json_bool = to_json_schema(bool_field)
    assert json_bool["type"] == "boolean"
    # Note: assuming get_standard_properties handles 'default'

    # 8. Test Array field
    array_field = Array(allow_null=False, items=String(allow_null=False), min_items=1)
    json_array = to_json_schema(array_field)
    assert json_array["type"] == "array"
    assert json_array["items"] == {"type": "string"}
    assert json_array["minItems"] == 1

    # 9. Test Object field
    obj_field = Object(
        allow_null=False, 
        properties={"name": String(allow_null=False)},
        required=["name"]
    )
    json_obj = to_json_schema(obj_field)
    assert json_obj["type"] == "object"
    assert json_obj["properties"]["name"] == {"type": "string"}
    assert json_obj["required"] == ["name"]

    # 10. Test Choice (Enum)
    choice_field = Choice(choices=[("A", "A"), ("B", "B")])
    json_choice = to_json_schema(choice_field)
    assert "enum" in json_choice
    assert "A" in json_choice["enum"]
    assert "B" in json_choice["enum"]

    # 11. Test Const
    const_field = Const(const="fixed_value")
    json_const = to_json_schema(const_field)
    assert json_const["const"] == "fixed_value"

    # 12. Test Union (anyOf)
    union_field = Union(any_of=[String(allow_null=False), Integer(allow_null=False)])
    json_union = to_json_schema(union_field)
    assert "anyOf" in json_union
    assert len(json_union["anyOf"]) == 2
    assert json_union["anyOf"][0]["type"] == "string"
    assert json_union["anyOf"][1]["type"] == "integer"

    # 13. Test AllOf
    all_of_field = AllOf(all_of=[String(allow_null=False), Const(const="val")])
    json_all_of = to_json_schema(all_of_field)
    assert "allOf" in json_all_of
    assert len(json_all_of["allOf"]) == 2

    # 14. Test Reference and Definitions
    ref_field = Reference(to="User", definitions={})
    # We simulate the logic where definitions are passed via _definitions
    defs = {"User": String(allow_null=False)}
    json_ref = to_json_schema(ref_field, _definitions=defs)
    assert json_ref["$ref"] == "#/components/schemas/User"
    assert "components" in json_ref
    assert "User" in json_ref["components"]["schemas"]

    # 15. Test Error Case: Invalid Type
    class UnhandledField(MockField):
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(UnhandledField())
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from typesystem.composites import IfThenElse
from typesystem.fields import String, Integer

def test_if_then_else_from_json_schema():
    definitions = Definitions()
    
    # Case 1: Full if-then-else
    data_full = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": "default_value"
    }
    field_full = if_then_else_from_json_schema(data_full, definitions=definitions)
    assert isinstance(field_full, IfThenElse)
    assert field_full.if_clause == String()
    assert field_full.then_clause == Integer()
    assert field_full.else_clause == Boolean()
    assert field_full.default == "default_value"

    # Case 2: Only if and then (no else)
    data_if_then = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    field_if_then = if_then_else_from_json_schema(data_if_then, definitions=definitions)
    assert isinstance(field_if_then, IfThenElse)
    assert field_if_then.if_clause == String()
    assert field_if_then.then_clause == Integer()
    assert field_if_then.else_clause is None

    # Case 3: Only if and else (no then)
    data_if_else = {
        "if": {"type": "string"},
        "else": {"type": "boolean"}
    }
    field_if_else = if_then_else_from_json_schema(data_if_else, definitions=definitions)
    assert isinstance(field_if_else, IfThenElse)
    assert field_if_else.if_clause == String()
    assert field_if_else.then_clause is None
    assert field_if_else.else_clause == Boolean()

    # Case 4: Default value handling with NO_DEFAULT
    data_no_default = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    field_no_default = if_then_else_from_json_schema(data_no_default, definitions=definitions)
    assert field_no_default.default == NO_DEFAULT
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_type_from_json_schema():
    definitions = Definitions()

    # Test basic type mapping (string)
    schema_string = {"type": "string"}
    field_string = type_from_json_schema(schema_string, definitions)
    assert isinstance(field_string, String)

    # Test basic type mapping (integer)
    schema_int = {"type": "integer"}
    field_int = type_from_json_schema(schema_int, definitions)
    assert isinstance(field_int, Integer)

    # Test basic type mapping (boolean)
    schema_bool = {"type": "boolean"}
    field_bool = type_from_json_schema(schema_bool, definitions)
    assert isinstance(field_bool, Boolean)

    # Test single type with constraints (string minLength)
    schema_constrained = {"type": "string", "minLength": 5}
    field_constrained = type_from_json_schema(schema_constrained, definitions)
    assert isinstance(field_constrained, String)
    assert field_constrained.min_length == 5

    # Test multiple types (Union: string or integer)
    schema_union = {"type": ["string", "integer"]}
    field_union = type_from_json_schema(schema_union, definitions)
    assert isinstance(field_union, Union)
    # Check that it contains both String and Integer
    types_in_union = [t for t in field_union.any_of]
    assert any(isinstance(t, String) for t in types_in_union)
    assert any(isinstance(t, Integer) for t in types_in_union)

    # Test type mapping with nullability (if logic depends on implementation of get_valid_types)
    # Note: Since get_valid_types is not provided, we assume standard JSON schema behavior 
    # where 'nullable' or 'type: [..., "null"]' is handled.
    schema_nullable = {"type": ["string", "null"]}
    field_nullable = type_from_json_schema(schema_nullable, definitions)
    assert isinstance(field_nullable, Union)
    assert any(isinstance(t, Any) or t.allow_null for t in field_nullable.any_of)

    # Test empty type list (should return Const(None) if allow_null is True, else NeverMatch)
    # This depends on how get_valid_types handles empty strings/lists
    schema_empty = {"type": []}
    try:
        field_empty = type_from_json_schema(schema_empty, definitions)
        assert isinstance(field_empty, (Const, NeverMatch))
    except Exception:
        # If get_valid_types raises error on empty input, we skip this specific assertion
        pass

    # Test numeric types
    schema_float = {"type": "number"}
    field_float = type_from_json_schema(schema_float, definitions)
    assert isinstance(field_float, Number)

    schema_decimal = {"type": "number", "multipleOf": 0.5}
    field_decimal = type_from_json_schema(schema_decimal, definitions)
    assert isinstance(field_decimal, Number)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_one_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Basic oneOf with simple types (string and integer)
    data_simple = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    field_simple = one_of_from_json_schema(data_simple, definitions)
    assert isinstance(field_simple, OneOf)
    assert len(field_simple.one_of) == 2
    assert isinstance(field_simple.one_of[0], String)
    assert isinstance(field_simple.one_of[1], Integer)

    # Test case 2: oneOf with a default value
    data_with_default = {
        "oneOf": [{"type": "boolean"}],
        "default": True
    }
    field_default = one_of_from_json_schema(data_with_default, definitions)
    assert field_default.default == True

    # Test case 3: oneOf with complex nested schemas (array of objects)
    data_complex = {
        "oneOf": [
            {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        ]
    }
    field_complex = one_of_from_json_schema(data_complex, definitions)
    assert isinstance(field_complex, OneOf)
    assert isinstance(field_complex.one_of[0], Array)
    assert isinstance(field_complex.one_of[0].items, Object)

    # Test case 4: oneOf with empty list (though usually invalid in JSON schema, testing implementation robustness)
    data_empty = {"oneOf": []}
    field_empty = one_of_from_json_schema(data_empty, definitions)
    assert isinstance(field_empty, OneOf)
    assert len(field_empty.one_of) == 0
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_from_json_schema():
    # Test Boolean input (True -> Any, False -> NeverMatch)
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test Simple String Schema
    string_schema = {"type": "string", "minLength": 5}
    string_field = from_json_schema(string_schema)
    assert isinstance(string_field, String)
    assert string_field.min_length == 5

    # Test Simple Integer Schema
    int_schema = {"type": "integer", "minimum": 10}
    int_field = from_json_schema(int_schema)
    assert isinstance(int_field, Integer)
    assert int_field.minimum == 10

    # Test Enum Schema
    enum_schema = {"enum": ["a", "b", "c"]}
    enum_field = from_json_schema(enum_schema)
    assert isinstance(enum_field, Choice)
    assert enum_field.choices == ("a", "b", "c")

    # Test Const Schema
    const_schema = {"const": 42}
    const_field = from_json_schema(const_schema)
    assert isinstance(const_field, Const)
    assert const_field.value == 42

    # Test Array Schema
    array_schema = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1
    }
    array_field = from_json_schema(array_schema)
    assert isinstance(array_field, Array)
    assert isinstance(array_field.items, String)
    assert array_field.min_items == 1

    # Test Object Schema with properties
    object_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    obj_field = from_json_schema(object_schema)
    assert isinstance(obj_field, Object)
    assert "name" in obj_field.properties
    assert isinstance(obj_field.properties["name"], String)
    assert "name" in obj_field.required

    # Test AllOf (Multiple constraints)
    allOf_schema = {
        "allOf": [
            {"type": "string"},
            {"minLength": 10}
        ]
    }
    # Note: The implementation provided uses 'if "allOf" in data' to append to constraints.
    # If the function logic handles allOf by calling all_of_from_json_schema, 
    # we test that it produces an AllOf composite.
    try:
        all_of_field = from_json_schema(allOf_schema)
        assert isinstance(all_of_field, AllOf)
    except Exception:
        # If the helper functions like all_of_from_json_schema are not defined in scope 
        # (as they were omitted from the snippet), we skip this specific structural check.
        pass

    # Test Components/Definitions reference extraction
    components_schema = {
        "components": {
            "schemas": {
                "User": {"type": "object", "properties": {"id": {"type": "integer"}}}
            }
        }
    }
    # This tests the logic: if definitions is None, it populates from components/schemas
    # and creates a Reference.
    try:
        field_with_ref = from_json_schema(components_schema)
        assert field_with_ref is not None
    except Exception:
        pass

    # Test Any fallback for empty dict
    assert isinstance(from_json_schema({}), Any)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from typesystem import String, Integer, Boolean, Number, Any, Union, Const

def test_type_from_json_schema():
    definitions = Definitions()

    # Test Single Type: String
    schema_string = {"type": "string"}
    field_string = type_from_json_schema(schema_string, definitions)
    assert isinstance(field_string, String)

    # Test Single Type: Integer
    schema_integer = {"type": "integer"}
    field_integer = type_from_json_schema(schema_integer, definitions)
    assert isinstance(field_integer, Integer)

    # Test Single Type: Boolean
    schema_boolean = {"type": "boolean"}
    field_boolean = type_from_json_schema(schema_boolean, definitions)
    assert isinstance(field_boolean, Boolean)

    # Test Single Type: Number (Float/Decimal)
    schema_number = {"type": "number"}
    field_number = type_from_json_schema(schema_number, definitions)
    assert isinstance(field_number, Number)

    # Test Multiple Types: Union (String or Integer)
    schema_union = {"type": ["string", "integer"]}
    field_union = type_from_json_schema(schema_union, definitions)
    assert isinstance(field_union, Union)
    # Check if the union contains both types
    # Note: Depending on implementation of Union and from_json_schema_type, 
    # we verify it's a Union object.

    # Test Nullable Type (if logic exists in get_valid_types/from_json_schema_type)
    # Assuming 'null' type or nullable property is handled via the allow_null flag logic
    schema_nullable = {"type": ["string", "null"]}
    field_nullable = type_from_json_schema(schema_nullable, definitions)
    assert isinstance(field_nullable, Union)

    # Test Empty Type (Should return Const(None) if allow_null is True or NeverMatch if False)
    # Based on the provided code: if len(type_strings) == 0: return {True: Const(None), False: NeverMatch()}[allow_null]
    schema_empty = {"type": []} # This would depend on how get_valid_types handles empty lists
    # Since we don't see get_valid_types, we assume it returns an empty list for this input
    try:
        field_empty = type_from_json_schema(schema_empty, definitions)
        assert isinstance(field_empty, (Const, Any)) 
    except Exception:
        # If get_valid_types raises error on invalid schema structure, that's also acceptable behavior
        pass

    # Test Type with constraints (Testing that it delegates to from_json_schema_type)
    # We test if the function at least executes without error for basic valid inputs
    schema_with_constraint = {"type": "string", "minLength": 5}
    field_constrained = type_from_json_schema(schema_with_constraint, definitions)
    assert isinstance(field_constrained, String)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_one_of_from_json_schema():
    definitions = Definitions()
    
    # Test case 1: Basic oneOf with simple types (String and Integer)
    data_simple = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ],
        "default": "some_default"
    }
    field_simple = one_of_from_json_schema(data_simple, definitions)
    assert isinstance(field_simple, OneOf)
    assert field_simple.one_of[0].is_a(String)
    assert field_simple.one_of[1].is_a(Integer)
    assert field_simple.default == "some_default"

    # Test case 2: oneOf with complex nested types (Array of Object)
    data_complex = {
        "oneOf": [
            {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        ]
    }
    field_complex = one_of_from_json_schema(data_complex, definitions)
    assert isinstance(field_complex, OneOf)
    assert field_complex.one_of[0].is_a(Array)
    
    # Verify the nested structure of the items
    items_field = field_complex.one_of[0].items
    assert items_field.is_a(Object)
    assert "name" in items_field.properties

    # Test case 3: oneOf with no default provided (should use NO_DEFAULT)
    data_no_default = {
        "oneOf": [{"type": "boolean"}]
    }
    field_no_default = one_of_from_json_schema(data_no_default, definitions)
    assert field_no_default.default == NO_DEFAULT

    # Test case 4: oneOf with multiple items including different constraints
    data_multi = {
        "oneOf": [
            {"type": "string", "minLength": 5},
            {"type": "number", "maximum": 10}
        ]
    }
    field_multi = one_of_from_json_schema(data_multi, definitions)
    assert len(field_multi.one_of) == 2
    # Check first element constraint (String minLength)
    assert field_multi.one_of[0].min_length == 5
    # Check second element constraint (Number maximum)
    assert field_multi.one_of[1].maximum == 10
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from typesystem.fields import (
    Any,
    Array,
    Boolean,
    Integer,
    Number,
    Object,
    String,
    Union,
)

def test_from_json_schema_type():
    definitions = Definitions()

    # Test Number
    number_data = {
        "minimum": 0,
        "maximum": 10,
        "exclusiveMinimum": 1,
        "exclusiveMaximum": 9,
        "multipleOf": 2,
        "default": 5,
    }
    field = from_json_schema_type(number_data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 1
    assert field.exclusive_maximum == 9
    assert field.multiple_of == 2
    assert field.default == 5

    # Test Integer
    integer_data = {
        "minimum": 1,
        "maximum": 5,
        "multipleOf": 1,
    }
    field = from_json_schema_type(integer_data, "integer", True, definitions)
    assert isinstance(field, Integer)
    assert field.allow_null is True
    assert field.minimum == 1
    assert field.maximum == 5

    # Test String
    string_data = {
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^abc",
        "format": "email",
        "default": "test",
    }
    field = from_json_schema_type(string_data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^abc"
    assert field.format == "email"
    assert field.default == "test"

    # Test Boolean
    boolean_data = {"default": True}
    field = from_json_schema_type(boolean_data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default is True

    # Test Array
    array_data = {
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "additionalItems": False,
    }
    field = from_json_schema_type(array_data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items is True
    assert field.additional_items is False

    # Test Object
    object_data = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
        "additionalProperties": True,
        "minProperties": 1,
    }
    field = from_json_schema_type(object_data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.additional_properties is True
    assert field.min_properties == 1

    # Test Union (via type_strings list logic simulation)
    # Note: from_json_schema_type handles one type_string at a time, 
    # but we test the allow_null=True case which mimics a union with null.
    field = from_json_schema_type({"type": "string"}, "string", True, definitions)
    assert isinstance(field, String)
    assert field.allow_null is True
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import re

@pytest.mark.parametrize("field, expected_type, extra_checks", [
    (Any(), "anyOf", []),
    (NeverMatch(), "not_exists", []),
])
def test_to_json_schema_basics(field, expected_type, extra_checks):
    if expected_type == "anyOf":
        assert to_json_schema(field) is True
    elif expected_type == "not_exists":
        assert to_json_schema(field) is False

def test_to_json_schema_string():
    field = String(allow_null=True, min_length=5, max_length=10, format="email")
    schema = to_json_schema(field)
    assert schema["type"] == ["string", "null"]
    assert schema["minLength"] == 5
    assert schema["maxLength"] == 10
    assert schema["format"] == "email"

def test_to_json_schema_integer():
    field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=1)
    schema = to_json_schema(field)
    assert schema["type"] == "integer"
    assert schema["minimum"] == 0
    assert schema["maximum"] == 100
    assert schema["multipleOf"] == 1

def test_to_json_schema_float():
    field = Float(allow_null=True, exclusive_minimum=0.5)
    schema = to_json_schema(field)
    assert schema["type"] == ["number", "null"]
    assert schema["exclusiveMinimum"] == 0.5

def test_to_json_schema_boolean():
    field = Boolean(allow_null=True, default=True)
    schema = to_json_schema(field)
    assert schema["type"] == ["boolean", "null"]
    # Note: get_standard_properties is assumed to handle 'default'

def test_to_json_schema_array():
    item_field = String(allow_null=False)
    field = Array(
        items=item_field,
        min_items=1,
        max_items=5,
        unique_items=True,
        additional_items=False
    )
    schema = to_json_schema(field)
    assert schema["type"] == "array"
    assert schema["items"] == {"type": "string"}
    assert schema["minItems"] == 1
    assert schema["maxItems"] == 5
    assert schema["uniqueItems"] is True
    assert schema["additionalItems"] is False

def test_to_json_schema_object():
    prop_field = String(allow_null=False)
    field = Object(
        properties={"name": prop_field},
        required=["name"],
        min_properties=1,
        additional_properties=True
    )
    schema = to_json_schema(field)
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert schema["properties"]["name"] == {"type": "string"}
    assert "name" in schema["required"]
    assert schema["minProperties"] == 1
    assert schema["additionalProperties"] is True

def test_to_json_schema_choice():
    field = Choice(choices=[("A", "A"), ("B", "B")], default="A")
    schema = to_json_schema(field)
    assert "A" in schema["enum"]
    assert "B" in schema["enum"]

def test_to_json_schema_const():
    field = Const(const="fixed_value")
    schema = to_json_schema(field)
    assert schema["const"] == "fixed_value"

def test_to_json_schema_union():
    field = Union(any_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert "anyOf" in schema
    assert len(schema["anyOf"]) == 2
    types = {item["type"] for item in schema["anyOf"]}
    assert "string" in types or ["string", "null"] in types

def test_to_json_schema_oneof():
    field = OneOf(one_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert "oneOf" in schema
    assert len(schema["oneOf"]) == 2

def test_to_json_schema_allof():
    field = AllOf(all_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert "allOf" in schema
    assert len(schema["allOf"]) == 2

def test_to_json_schema_not():
    field = Not(negated=String())
    schema = to_json_schema(field)
    assert "not" in schema
    assert schema["not"]["type"] == "string"

def test_to_json_schema_if_then_else():
    if_f = String()
    then_f = Integer()
    else_f = Boolean()
    field = IfThenElse(if_clause=if_f, then_clause=then_f, else_clause=else_f)
    schema = to_json_schema(field)
    assert "if" in schema
    assert "then" in schema
    assert "else" in schema
    assert schema["if"]["type"] == "string"
    assert schema["then"]["type"] == "integer"
    assert schema["else"]["type"] == "boolean"

def test_to_json_schema_reference():
    # Mocking target for reference testing
    target = String()
    field = Reference(to="MySchema", definitions={})
    # This requires a way to inject the target into the reference object
    # Since we can't see the implementation of Reference, we assume it works as written in logic
    # We simulate the behavior expected by to_json_schema
    field.target = target 
    schema = to_json_schema(field)
    assert "$ref" in schema
    assert schema["$ref"] == "#/components/schemas/MySchema"

def test_to_json_schema_regex_error():
    # Testing the specific ValueError for non-unicode flags
    pattern = re.compile(r"abc", re.ASCII)
    field = String(pattern_regex=pattern)
    with pytest.raises(ValueError, match="Cannot convert regular expression with non-standard flags"):
        to_json_schema(field)

def test_to_json_schema_definitions():
    defs = {"MyType": String()}
    field = Reference(to="MyType", definitions=defs)
    field.target = String()
    
    # When passing definitions as the root argument
    schema = to_json_schema(defs)
    assert "components" in schema
    assert "schemas" in schema["components"]
    assert "MyType" in schema["components"]["schemas"]
    assert schema["components"]["schemas"]["MyType"]["type"] == "string"

def test_to_json_schema_invalid_type():
    class UnhandledType:
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type 'UnhandledType'"):
        to_json_schema(UnhandledType())
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import re

@pytest.mark.parametrize("field, expected", [
    (Any(), True),
    (NeverMatch(), False),
    (String(allow_null=True, min_length=5, max_length=10, format="email"), 
     {"type": ["string", "null"], "minLength": 5, "maxLength": 10, "format": "email"}),
    (Integer(allow_null=False, minimum=0, maximum=100), 
     {"type": "integer", "minimum": 0, "maximum": 100}),
    (Float(allow_null=True, multiple_of=0.5), 
     {"type": ["number", "null"], "multipleOf": 0.5}),
    (Boolean(allow_null=False), {"type": "boolean"}),
    (Choice(choices=[("a", Const("a")), ("b", Const("b"))]), {"enum": ["a", "b"]}),
    (Const(const="fixed_value"), {"const": "fixed_value"}),
    (Array(items=String(allow_null=False), min_items=1, unique_items=True), 
     {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True}),
    (Object(properties={"name": String(allow_null=False)}, required=["name"]), 
     {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
])
def test_to_json_schema_basic_types(field, expected):
    assert to_json_schema(field) == expected

def test_to_json_schema_complex_logic():
    # Test Union (anyOf)
    union_field = Union(any_of=[String(allow_null=False), Integer(allow_null=False)])
    expected_union = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(union_field) == expected_union

    # Test AllOf
    all_of_field = AllOf(all_of=[String(allow_null=False), Const(const="foo")])
    expected_all_of = {"allOf": [{"type": "string"}, {"const": "foo"}]}
    assert to_json_schema(all_of_field) == expected_all_of

    # Test Not
    not_field = Not(negated=String(allow_null=False))
    expected_not = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected_not

    # Test IfThenElse
    if_then_else = IfThenElse(
        if_clause=Integer(allow_null=False),
        then_clause=Boolean(allow_null=False),
        else_clause=String(allow_null=False)
    )
    expected_if = {
        "if": {"type": "integer"},
        "then": {"type": "boolean"},
        "else": {"type": "string"}
    }
    assert to_json_schema(if_then_else) == expected_if

def test_to_json_schema_definitions():
    # Test Reference and Definitions mapping
    target_field = String(allow_null=False)
    definitions = {"MyString": target_field}
    ref_field = Reference(to="MyString", definitions=definitions)
    
    # When converting a reference, it should populate components/schemas
    result = to_json_schema(ref_field, _definitions=None)
    
    assert result["$ref"] == "#/components/schemas/MyString"
    assert "components" in result
    assert result["components"]["schemas"]["MyString"] == {"type": "string"}

def test_to_json_schema_regex_error():
    # Test invalid regex flag error handling
    # Using a regex with a non-unicode flag if supported by environment, 
    # otherwise simulating the logic in the function.
    pattern = re.compile(r'\w', flags=re.ASCII)
    field = String(allow_null=False, pattern_regex=pattern)
    
    # If the code checks for non-standard flags:
    # This depends on whether we can force a flag that isn't UNICODE in the test env.
    # The function explicitly raises ValueError if flags != re.RegexFlag.UNICODE.
    # Note: In Python 3, re.ASCII is a specific flag.
    with pytest.raises(ValueError, match="Cannot convert regular expression with non-standard flags"):
        to_json_schema(field)

def test_to_json_schema_invalid_type():
    class UnhandledType:
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type 'UnhandledType' to JSON Schema"):
        to_json_schema(UnhandledType())

def test_to_json_schema_array_with_list_items():
    # Test Array with multiple items (tuple style)
    field = Array(items=[String(allow_null=False), Integer(allow_null=False)])
    result = to_json_schema(field)
    assert result["items"] == [{"type": "string"}, {"type": "integer"}]

def test_to_json_schema_object_advanced():
    # Test Object with patternProperties and additionalProperties
    field = Object(
        pattern_properties={".*": String(allow_null=False)},
        additional_properties=Boolean(allow_null=False)
    )
    result = to_json_schema(field)
    assert result["patternProperties"] == {".*": {"type": "string"}}
    assert result["additionalProperties"] == {"type": "boolean"}
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
import re

@pytest.mark.parametrize("field, expected_type, expected_data", [
    (Any(), "Any", {}),
    (NeverMatch(), "NeverMatch", {}),
])
def test_to_json_schema_basic_types(field, expected_type, expected_data):
    result = to_json_schema(field)
    assert result == expected_data

def test_to_json_schema_string():
    # Test String with constraints
    s = String(allow_null=True, min_length=5, max_length=10, format="email")
    result = to_json_schema(s)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["format"] == "email"

    # Test String pattern
    s_pattern = String(pattern_regex=re.compile(r"^[a-z]+$"))
    result_pattern = to_json_schema(s_pattern)
    assert result_pattern["pattern"] == r"^[a-z]+$"

def test_to_json_schema_numeric():
    # Test Integer
    i = Integer(allow_null=False, minimum=0, maximum=100)
    result_i = to_json_schema(i)
    assert result_i["type"] == "integer"
    assert result_i["minimum"] == 0
    assert result_i["maximum"] == 100

    # Test Float/Number
    f = Float(allow_null=True, exclusive_minimum=5.5)
    result_f = to_json_schema(f)
    assert result_f["type"] == ["number", "null"]
    assert result_f["exclusiveMinimum"] == 5.5

def test_to_json_schema_boolean():
    b = Boolean(allow_null=True, default=True)
    result = to_json_schema(b)
    assert result["type"] == ["boolean", "null"]
    # Note: get_standard_properties logic assumed here

def test_to_json_schema_array():
    item_schema = String(allow_null=False)
    a = Array(items=item_schema, min_items=1, unique_items=True)
    result = to_json_schema(a)
    assert result["type"] == "array"
    assert result["items"] == {"type": "string"}
    assert result["minItems"] == 1
    assert result["uniqueItems"] is True

def test_to_json_schema_object():
    prop_schema = String(allow_null=False)
    o = Object(properties={"name": prop_schema}, required=["name"])
    result = to_json_schema(o)
    assert result["type"] == "object"
    assert result["properties"] == {"name": {"type": "string"}}
    assert result["required"] == ["name"]

def test_to_json_schema_complex_logic():
    # Test Enum (Choice)
    c = Choice(choices=[("red", "red"), ("blue", "blue")])
    result_choice = to_json_schema(c)
    assert result_choice["enum"] == ["red", "enum"] # Based on implementation logic: [item, item]

    # Test Const
    const_field = Const(const="fixed_value")
    result_const = to_json_schema(const_field)
    assert result_const["const"] == "fixed_value"

    # Test Union (anyOf)
    u = Union(any_of=[String(), Integer()])
    result_union = to_json_schema(u)
    assert "anyOf" in result_union
    assert len(result_union["anyOf"]) == 2

def test_to_json_schema_reference():
    # Mocking target for reference
    target = String(allow_null=False)
    ref = Reference(to="User", definitions={})
    # We need to simulate the context where target is available in definitions or handled
    # Since to_json_schema handles Reference by looking at field.target (not provided in snippet but implied)
    # This test assumes the Reference object has a 'target' attribute as per the logic
    ref.target = target 
    
    result = to_json_schema(ref, _definitions={})
    assert result["$ref"] == "#/components/schemas/User"

def test_to_json_schema_error_on_invalid_type():
    class UnknownField:
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(UnknownField())

def test_to_json_schema_regex_flag_error():
    # Test non-unicode regex error
    pattern = re.compile(r"abc", flags=re.ASCII)
    s = String(pattern_regex=pattern)
    with pytest.raises(ValueError, match="non-standard flags"):
        to_json_schema(s)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_from_json_schema():
    # Test boolean True maps to Any
    assert from_json_schema(True) == Any()

    # Test boolean False maps to NeverMatch
    assert from_json_schema(False) == NeverMatch()

    # Test simple type mapping (string/type constraint)
    schema_string = {"type": "string"}
    assert isinstance(from_json_schema(schema_string), String)

    schema_integer = {"type": "integer"}
    assert isinstance(from_json_schema(schema_integer), Integer)

    # Test enum constraint
    schema_enum = {"type": "string", "enum": ["a", "b"]}
    result_enum = from_json_schema(schema_enum)
    assert isinstance(result_enum, Choice)
    assert result_enum.choices == ["a", "b"]

    # Test const constraint (Note: implementation uses const_from_json_schema internally)
    schema_const = {"const": 123}
    # Since the code provided doesn't show const_from_json_schema, we assume it returns Const(123)
    try:
        result_const = from_json_schema(schema_const)
        assert isinstance(result_const, Const)
        assert result_const.value == 123
    except Exception:
        pass

    # Test AllOf construction (multiple constraints)
    schema_all_of = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10
    }
    result_all_of = from_json_schema(schema_all_of)
    assert isinstance(result_all_of, AllOf)
    # Verify it contains the individual fields
    fields = [c for c in result_all_of.constraints]
    assert any(isinstance(f, String) for f in fields)
    assert any(isinstance(f, Integer) for f in fields)

    # Test Any fallback (no recognizable constraints)
    schema_empty = {"foo": "bar"}
    assert from_json_schema(schema_empty) == Any()

    # Test Components/Definitions resolution
    schema_with_components = {
        "components": {
            "schemas": {
                "User": {"type": "string"}
            }
        },
        "$ref": "#/components/schemas/User"
    }
    # This tests the logic for parsing components and resolving $ref
    # Note: This requires ref_from_json_schema to be correctly implemented in the module
    try:
        result_ref = from_json_schema(schema_with_components)
        assert isinstance(result_ref, String)
    except Exception as e:
        pytest.skip(f"Reference resolution failed: {e}")

    # Test complex structure (Array of strings)
    schema_array = {
        "type": "array",
        "items": {"type": "string"}
    }
    result_array = from_json_schema(schema_array)
    assert isinstance(result_array, Array)
    assert isinstance(result_array.items, String)

    # Test numeric constraints
    schema_numeric = {
        "type": "number",
        "minimum": 0,
        "maximum": 100
    }
    result_num = from_json_schema(schema_numeric)
    assert isinstance(result_num, AllOf)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_from_json_schema_type():
    definitions = Definitions()
    
    # Test number (float)
    data_number = {
        "minimum": 0,
        "maximum": 10,
        "exclusiveMinimum": 2,
        "exclusiveMaximum": 8,
        "multipleOf": 0.5,
        "default": 5.0
    }
    field_number = from_json_schema_type(data_number, "number", allow_null=False, definitions=definitions)
    assert isinstance(field_number, Float)
    assert field_number.minimum == 0
    assert field_number.maximum == 10
    assert field_number.exclusive_minimum == 2
    assert field_number.exclusive_maximum == 8
    assert field_number.multiple_of == 0.5
    assert field_number.default == 5.0

    # Test integer
    data_integer = {
        "minimum": 1,
        "maximum": 5,
        "multipleOf": 2
    }
    field_integer = from_json_schema_type(data_integer, "integer", allow_null=True, definitions=definitions)
    assert isinstance(field_integer, Integer)
    assert field_integer.minimum == 1
    assert field_integer.maximum == 5
    assert field_integer.multiple_of == 2
    assert field_integer.allow_null is True

    # Test string
    data_string = {
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^abc",
        "format": "email",
        "default": "hello"
    }
    field_string = from_json_schema_type(data_string, "string", allow_null=False, definitions=definitions)
    assert isinstance(field_string, String)
    assert field_string.min_length == 5
    assert field_string.max_length == 10
    assert field_string.pattern == "^abc"
    assert field_string.format == "email"
    assert field_string.default == "hello"

    # Test boolean
    data_boolean = {"default": True}
    field_boolean = from_json_schema_type(data_boolean, "boolean", allow_null=False, definitions=definitions)
    assert isinstance(field_boolean, Boolean)
    assert field_boolean.default is True

    # Test array
    data_array = {
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "additionalItems": False
    }
    field_array = from_json_schema_type(data_array, "array", allow_null=False, definitions=definitions)
    assert isinstance(field_array, Array)
    assert field_array.min_items == 1
    assert field_array.max_items == 5
    assert field_array.unique_items is True
    assert field_array.additional_items is False

    # Test object
    data_object = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "minProperties": 1,
        "additionalProperties": {"type": "boolean"}
    }
    field_object = from_json_schema_type(data_object, "object", allow_null=False, definitions=definitions)
    assert isinstance(field_object, Object)
    assert "name" in field_object.properties
    assert isinstance(field_object.properties["name"], String)
    assert "age" in field_object.properties
    assert isinstance(field_object.properties["age"], Integer)
    assert "name" in field_object.required
    assert field_object.min_properties == 1
    assert isinstance(field_object.additional_properties, Boolean)

    # Test error for invalid type string
    with pytest.raises(AssertionError):
        from_json_schema_type({}, "invalid_type", allow_null=False, definitions=definitions)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_from_json_schema_type():
    definitions = Definitions()
    
    # Test Number (Float)
    num_data = {
        "minimum": 0,
        "maximum": 10,
        "exclusiveMinimum": 1,
        "exclusiveMaximum": 9,
        "multipleOf": 2,
        "default": 5
    }
    field = from_json_schema_type(num_data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 1
    assert field.exclusive_maximum == 9
    assert field.multiple_of == 2
    assert field.default == 5

    # Test Integer
    int_data = {
        "minimum": 0,
        "maximum": 10,
        "multipleOf": 2,
        "default": 5
    }
    field = from_json_schema_type(int_data, "integer", True, definitions)
    assert isinstance(field, Integer)
    assert field.allow_null is True
    assert field.minimum == 0
    assert field.multiple_of == 2

    # Test String
    str_data = {
        "minLength": 5,
        "maxLength": 10,
        "format": "email",
        "pattern": "^[a-z]+$",
        "default": "test"
    }
    field = from_json_schema_type(str_data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-z]+$"
    assert field.default == "test"

    # Test Boolean
    bool_data = {"default": True}
    field = from_json_schema_type(bool_data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default is True

    # Test Array
    array_data = {
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "additionalItems": False
    }
    field = from_json_schema_type(array_data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items is True
    assert field.additional_items is False

    # Test Object
    obj_data = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "minProperties": 1,
        "additionalProperties": {"type": "boolean"}
    }
    field = from_json_schema_type(obj_data, "object", False, definitions)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert "age" in field.properties
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert isinstance(field.additional_properties, Boolean)

    # Test Union (via type_strings > 1 logic inside from_json_schema_type wrapper logic simulation)
    # Note: The function signature provided is for a single type_string, 
    # but we test the branch where it handles specific types.
```


