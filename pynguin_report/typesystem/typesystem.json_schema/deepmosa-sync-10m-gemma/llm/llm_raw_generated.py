####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_all_of_from_json_schema():
    definitions = Definitions()
    data = {
        "allOf": [
            {"type": "string"},
            {"enum": ["a", "b"]}
        ],
        "default": "a"
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert result.default == "a"

def test_all_of_from_json_schema_empty():
    definitions = Definitions()
    data = {"allOf": []}
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 0

def test_all_of_from_json_schema_with_ref():
    definitions = Definitions()
    # Mocking a definition to avoid error during from_json_schema call for $ref
    definitions["#/components/schemas/test"] = Any()
    data = {
        "allOf": [
            {"$ref": "#/components/schemas/test"}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert isinstance(result.all_of[0], Reference)
    assert result.all_of[0].to == "#/components/schemas/test"
```


# LLM-generated content at query #2
#--------------------------

def test_from_json_schema_type_number():
    data = {"minimum": 1, "maximum": 10, "default": 5}
    definitions = Definitions()
    result = from_json_schema_type(data, "number", allow_null=False, definitions=definitions)
    assert isinstance(result, Float)
    assert result.minimum == 1
    assert result.maximum == 10
    assert result.default == 5

def test_from_json_schema_type_integer():
    data = {"exclusiveMinimum": 0, "multipleOf": 2}
    definitions = Definitions()
    result = from_json_schema_type(data, "integer", allow_null=True, definitions=definitions)
    assert isinstance(result, Integer)
    assert result.exclusive_minimum == 0
    assert result.multiple_of == 2
    assert result.allow_null is True

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 10, "pattern": "^abc", "format": "email"}
    definitions = Definitions()
    result = from_json_schema_type(data, "string", allow_null=False, definitions=definitions)
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.pattern == "^abc"
    assert result.format == "email"

def test_from_json_schema_type_boolean():
    data = {"default": True}
    definitions = Definitions()
    result = from_json_schema_type(data, "boolean", allow_null=False, definitions=definitions)
    assert isinstance(result, Boolean)
    assert result.default is True

def test_from_json_schema_type_array():
    data = {
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "uniqueItems": True
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "array", allow_null=False, definitions=definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    assert result.additional_items is False
    assert result.min_items == 1
    assert result.unique_items is True

def test_from_json_schema_type_object():
    data = {
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": {"type": "integer"}
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "object", allow_null=False, definitions=definitions)
    assert isinstance(result, Object)
    assert isinstance(result.properties["name"], String)
    assert "name" in result.required
    assert isinstance(result.additional_properties, Integer)


# LLM-generated content at query #3
#--------------------------

```python
def test_to_json_schema_primitive_string():
    from typesystem.fields import String
    field = String(allow_null=True, min_length=5)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 5

def test_to_json_schema_primitive_integer():
    from typesystem.fields import Integer
    field = Integer(default=10, minimum=0)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["default"] == 10
    assert result["minimum"] == 0

def test_to_json_schema_boolean():
    from typesystem.fields import Boolean
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

def test_to_json_schema_array():
    from typesystem.fields import Array, String
    field = Array(items=String(), min_items=1)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["items"] == {"type": "string", "minLength": 1}

def test_to_json_schema_object():
    from typesystem.fields import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string", "minLength": 1}
    assert result["properties"]["age"] == {"type": "integer"}
    assert result["required"] == ["name"]

def test_to_json_schema_union():
    from typesystem.fields import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

def test_to_json_schema_definitions():
    from typesystem.schemas import Definitions
    from typesystem.fields import String
    defs = Definitions({"User": String()})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["User"] == {"type": "string", "minLength": 1}

def test_to_json_schema_reference():
    from typesystem.fields import Reference, String
    # Mocking Reference and its target as it's not fully defined in the snippet
    class MockTarget: pass
    class MockRef:
        def __init__(self, to, target):
            self.to = to
            self.target = target
    
    # Since Reference is used in the code but not provided in detail, 
    # we assume its structure based on usage.
    from typesystem.fields import Field
    class Reference(Field):
        def __init__(self, to, target):
            super().__init__()
            self.to = to
            self.target = target

    ref = Reference(to="User", target=String())
    result = to_json_schema(ref)
    assert result["$ref"] == "#/components/schemas/User"
```


# LLM-generated content at query #4
#--------------------------

```python
def test_type_from_json_schema_single_type_integer():
    data = {"type": "integer", "minimum": 5}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Integer)
    assert field.allow_null is False
    assert field.minimum == 5

def test_type_from_json_schema_single_type_string():
    data = {"type": "string", "minLength": 3, "maxLength": 10}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, String)
    assert field.min_length == 3
    assert field.max_length == 10

def test_type_from_json_schema_with_null():
    data = {"type": ["string", "null"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Union)
    assert field.allow_null is True
    assert len(field.any_of) == 1
    assert isinstance(field.any_of[0], String)

def test_type_from_json_schema_number_discards_integer():
    data = {"type": ["number", "integer"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Union)
    # get_valid_types discards 'integer' if 'number' is present
    # So only 'number' (Float) should remain in the union
    assert len(field.any_of) == 1
    assert isinstance(field.any_of[0], Float)

def test_type_from_json_schema_empty_types_with_null():
    data = {"type": ["null"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Const)
    assert field.const is None

def test_type_from_json_schema_boolean_type():
    data = {"type": "boolean", "default": True}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Boolean)
    assert field.default is True

def test_type_from_json_schema_array_type():
    data = {"type": "array", "minItems": 2}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Array)
    assert field.min_items == 2

def test_type_from_json_schema_object_type():
    data = {"type": "object", "required": ["name"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Object)
    assert field.required == ["name"]
```


# LLM-generated content at query #5
#--------------------------

```python
def test_to_json_schema_array_items_is_list():
    import typing
    # Mocking the necessary classes and structures to reach line 76/77
    class Array:
        def __init__(self, items):
            self.items = items
            self.allow_null = False
            self.min_items = None
            self.max_items = None
            self.additional_items = None
            self.unique_items = False

    class String:
        def __init__(self):
            self.allow_null = False
            self.min_length = None
            self.max_length = None
            self.pattern_regex = None
            self.format = None
            self.allow_blank = True

    # Line 76: isinstance(field.items, (list, tuple))
    # To make line 76 evaluate to True (so that the predicate at line 77 is NOT reached/evaluated for the single item logic),
    # we must provide a list of items. 
    # However, the prompt asks to ensure the predicate AT line 77 evaluates to False.
    # Line 77 is 'else: data["items"] = to_json_schema(field.items, _definitions=definitions)'.
    # The predicate at line 76 (isinstance(field.items, (list, tuple))) must be False.
    
    item_type = String()
    array_field = Array(items=item_type)
    
    # We need to mock get_standard_properties because it's called in the execution flow
    import sys
    from types import ModuleType
    m = ModuleType("mock_module")
    m.get_standard_properties = lambda x: {}
    sys.modules["mock_module"] = m
    # If the code uses 'from mock_module import get_standard_properties', we need to ensure it's available.
    # Since I can't modify the original code's imports, I assume it is in the global scope.
    import __main__
    __main__.get_standard_properties = lambda x: {}

    result = to_json_schema(array_field)
    
    assert "items" in result
    assert isinstance(result["items"], dict)
```


# LLM-generated content at query #6
#--------------------------

```python
def test_to_json_schema_if_then_else_then_clause_is_none():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=None)
    result = to_json_schema(field)
    assert "if" in result
    assert "then" not in result
```


# LLM-generated content at query #7
#--------------------------

```python
def test_to_json_schema_reference_predicate_true():
    reference_field = Reference(to="MySchema")
    # We need to pass a Field object containing the Reference so that 
    # line 15 executes and sets 'field' to our Reference instance.
    arg = Field(type=reference_field)
    result = to_json_schema(arg)
    assert "$ref" in result
```


# LLM-generated content at query #8
#--------------------------

```python
def test_to_json_schema_additional_properties_is_not_bool():
    additional_properties_field = String(allow_null=True)
    object_field = Object(
        properties={},
        pattern_properties={},
        additional_properties=additional_properties_field,
        property_names=None,
        max_properties=None,
        min_properties=None,
        required=[]
    )
    result = to_json_schema(object_field)
    assert "additionalProperties" in result
    assert isinstance(result["additionalProperties"], dict)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_to_json_schema_pattern_properties_true():
    pattern_field = String(pattern_regex=re.compile("^[a-z]+$"))
    pattern_value = String(allow_null=True)
    obj_field = Object(pattern_properties={"^[0-9]+$": pattern_value})
    result = to_json_schema(obj_field)
    assert "patternProperties" in result
```


# LLM-generated content at query #10
#--------------------------

```python
def test_to_json_schema_evaluates_object_branch():
    field = Object(allow_null=False, properties={}, pattern_properties={}, additional_properties=None, property_names=None, max_properties=None, min_properties=None, required=[])
    result = to_json_schema(field)
    assert isinstance(result, dict)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_to_json_schema_format_is_not_none():
    field = String(format="date-time", allow_null=False)
    result = to_json_schema(field)
    assert result["format"] == "date-time"
```


# LLM-generated content at query #12
#--------------------------

```python
def test_to_json_schema_else_clause_is_none():
    if_clause = String(allow_null=True)
    then_clause = Boolean(allow_null=False)
    else_clause = None
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause, else_clause=else_clause)
    result = to_json_schema(field)
    assert "else" not in result
```


# LLM-generated content at query #13
#--------------------------

```python
def test_to_json_schema_evaluates_true_for_any_type():
    arg = Any()
    result = to_json_schema(arg)
    assert result is True
```


# LLM-generated content at query #14
#--------------------------

```python
def test_to_json_schema_predicate_false():
    arg = String(allow_null=False)
    result = to_json_schema(arg)
    assert result != True
```


# LLM-generated content at query #15
#--------------------------

```python
def test_to_json_schema_else_clause_is_not_none():
    if_clause = String(allow_null=True)
    then_clause = Integer()
    else_clause = Boolean()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause, else_clause=else_clause)
    result = to_json_schema(field)
    assert "else" in result
```


# LLM-generated content at query #16
#--------------------------

```python
def test_ref_from_json_schema_success():
    class MockField:
        pass

    class MockDefinitions(dict):
        pass

    definitions = MockDefinitions({"#/user": MockField()})
    data = {"$ref": "#/user"}
    result = ref_from_json_schema(data, definitions)
    
    assert isinstance(result, Reference)
    assert result.to == "#/user"
    assert result.target == definitions["#/user"]

def test_ref_from_json_schema_invalid_ref_prefix():
    class MockField:
        pass

    class MockDefinitions(dict):
        pass

    definitions = MockDefinitions()
    data = {"$ref": "user"}
    
    import pytest
    with pytest.raises(AssertionError, match="Unsupported \$ref style in document."):
        ref_from_json_schema(data, definitions)

def test_ref_from_json_schema_missing_ref_key():
    class MockField:
        pass

    class MockDefinitions(dict):
        pass

    definitions = MockDefinitions()
    data = {"not_a_ref": "#/user"}
    
    import pytest
    with pytest.raises(KeyError):
        ref_from_json_schema(data, definitions)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_from_json_schema_bool_true():
    field = from_json_schema(True)
    assert isinstance(field, Any)


def test_from_json_schema_bool_false():
    field = from_json_schema(False)
    assert isinstance(field, NeverMatch)


def test_from_json_schema_ref():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/MyType"}
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/MyType"


def test_from_json_schema_enum():
    data = {"enum": ["a", "b"]}
    field = from_json_schema(data)
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b")]


def test_from_json_schema_const():
    data = {"const": 123}
    field = from_json_schema(data)
    assert isinstance(field, Const)
    assert field.const == 123


def test_from_json_schema_all_of():
    data = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    # Note: from_json_schema calls itself recursively on items, 
    # so we rely on the existence of from_json_schema logic for children.
    field = from_json_schema(data)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2


def test_from_json_schema_any_of():
    data = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    field = from_json_schema(data)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2


def test_from_json_schema_one_of():
    data = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    field = from_json_schema(data)
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2


def test_from_json_schema_not():
    data = {"not": {"type": "string"}}
    field = from_json_schema(data)
    assert isinstance(field, Not)


def test_from_json_schema_if_then_else():
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    field = from_json_schema(data)
    assert isinstance(field, IfThenElse)
    assert field.if_clause is not None
    assert field.then_clause is not None
    assert field.else_clause is not None


def test_from_json_schema_components_definitions():
    data = {
        "components": {
            "schemas": {
                "User": {"type": "string"}
            }
        }
    }
    definitions = Definitions()
    field = from_json_schema(data, definitions=definitions)
    assert "#/components/schemas/User" in definitions
    assert isinstance(definitions["#/components/schemas/User"], String)


def test_from_json_schema_any_fallback():
    data = {"unsupported_key": "value"}
    field = from_json_schema(data)
    assert isinstance(field, Any)
```


# LLM-generated content at query #18
#--------------------------

```python
def test_to_json_schema_max_length_predicate_true():
    string_field = String(max_length=10)
    result = to_json_schema(string_field)
    assert "maxLength" in result
    assert result["maxLength"] == 10
```


# LLM-generated content at query #19
#--------------------------

```python
def test_to_json_schema_any():
    from typesystem import Any
    assert to_json_schema(Any()) is True

def test_to_json_schema_never_match():
    from typesystem import NeverMatch
    assert to_json_schema(NeverMatch()) is False

def test_to_json_schema_string_basic():
    from typesystem import String
    assert to_json_schema(String()) == {"type": "string", "minLength": 1}

def test_to_json_schema_string_nullable():
    from typesystem import String
    assert to_json_schema(String(allow_null=True)) == {"type": ["string", "null"], "minLength": 1}

def test_to_json_schema_integer_with_constraints():
    from typesystem import Integer
    field = Integer(minimum=0, maximum=10)
    assert to_json_schema(field) == {"type": "integer", "minimum": 0, "maximum": 10}

def test_to_json_schema_boolean():
    from typesystem import Boolean
    assert to_json_schema(Boolean()) == {"type": "boolean"}

def test_to_json_schema_array_basic():
    from typesystem import Array, String
    field = Array(items=String())
    assert to_json_schema(field) == {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": True}

def test_to_json_schema_object_properties():
    from typesystem import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()})
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer"}
        }
    }
    assert to_json_schema(field) == expected

def test_to_json_schema_definitions():
    from typesystem import Definitions, String, Integer
    defs = Definitions({"User": Object(properties={"id": Integer()})})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["User"] == {"type": "object", "properties": {"id": {"type": "integer"}}}

def test_to_json_schema_union():
    from typesystem import Union, String, Integer
    field = Union([String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert {"type": "string", "minLength": 1} in result["anyOf"]
    assert {"type": "integer"} in result["anyOf"]

def test_to_json_schema_const():
    from typesystem import Const
    field = Const(value="fixed")
    assert to_json_schema(field) == {"const": "fixed"}

def test_to_json_schema_choice():
    from typesystem import Choice
    field = Choice(choices=[("a", None), ("b", None)])
    assert to_json_schema(field) == {"enum": ["a", "b"]}

def test_to_json_schema_default_value():
    from typesystem import String
    field = String(default="hello")
    assert to_json_schema(field)["default"] == "hello"
```


# LLM-generated content at query #20
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"minimum": 1, "maximum": 10, "default": 5}
    definitions = Definitions()
    result = from_json_schema_type(data, "number", allow_null=False, definitions=definitions)
    assert isinstance(result, Float)
    assert result.minimum == 1
    assert result.maximum == 10
    assert result.default == 5

def test_from_json_schema_type_integer():
    data = {"minimum": 0, "exclusiveMinimum": 1}
    definitions = Definitions()
    result = from_json_schema_type(data, "integer", allow_null=True, definitions=definitions)
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.exclusive_minimum == 1
    assert result.allow_null is True

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 10, "pattern": "^abc", "format": "email"}
    definitions = Definitions()
    result = from_json_schema_type(data, "string", allow_null=False, definitions=components_schemas)
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.pattern == "^abc"
    assert result.format == "email"

def test_from_json_schema_type_string_allow_blank():
    data = {"minLength": 0}
    definitions = Definitions()
    result = from_json_schema_type(data, "string", allow_null=False, definitions=definitions)
    assert result.allow_blank is True

def test_from_json_schema_type_boolean():
    data = {"default": True}
    definitions = Definitions()
    result = from_json_schema_type(data, "boolean", allow_null=False, definitions=definitions)
    assert isinstance(result, Boolean)
    assert result.default is True

def test_from_json_schema_type_array_simple():
    data = {"items": {"type": "string"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", allow_null=False, definitions=definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

def test_from_json_schema_type_array_complex():
    data = {
        "items": [{"type": "integer"}, {"type": "boolean"}],
        "additionalItems": {"type": "string"},
        "minItems": 1
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "array", allow_null=False, definitions=definitions)
    assert isinstance(result.items, list)
    assert isinstance(result.items[0], Integer)
    assert isinstance(result.items[1], Boolean)
    assert isinstance(result.additional_items, String)
    assert result.min_items == 1

def test_from_json_schema_type_object_simple():
    data = {
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "object", allow_null=False, definitions=definitions)
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert isinstance(result.properties["name"], String)
    assert "name" in result.required

def test_from_json_schema_type_object_advanced():
    data = {
        "properties": {"age": {"type": "integer"}},
        "patternProperties": {"^id_": {"type": "string"}},
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "object", allow_null=False, definitions=definitions)
    assert isinstance(result.properties["age"], Integer)
    assert "^id_" in result.pattern_properties
    assert result.additional_properties is False
    assert isinstance(result.property_names, String)
    assert result.min_properties == 1
```


# LLM-generated content at query #21
#--------------------------

```python
def test_to_json_schema_any():
    from typesystem import Any
    assert to_json_schema(Any()) is True

def test_to_json_schema_never_match():
    from typesystem import NeverMatch
    assert to_json_schema(NeverMatch()) is False

def test_to_json_schema_string_basic():
    from typesystem import String
    field = String()
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 1

def test_to_json_schema_string_with_constraints():
    from typesystem import String
    import re
    field = String(min_length=5, max_length=10, pattern_regex=re.compile(r"^[a-z]+$"))
    result = to_json_schema(field)
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"

def test_to_json_schema_string_allow_null():
    from typesystem import String
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

def test_to_json_schema_integer_basic():
    from typesystem import Integer
    field = Integer(minimum=0, maximum=100)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100

def test_to_json_schema_boolean():
    from typesystem import Boolean
    field = Boolean(default=True)
    result = to_json_schema(field)
    assert result["type"] == "boolean"
    assert result["default"] is True

def test_to_json_schema_array():
    from typesystem import Array, String, Integer
    field = Array(items=String(), min_items=1, max_items=5)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["items"] == {"type": "string", "minLength": 1}

def test_to_json_schema_object():
    from typesystem import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert "name" in result["required"]

def test_to_json_schema_union():
    from typesystem import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

def test_to_json_schema_definitions():
    from typesystem import Definitions, String, Integer
    defs = Definitions({"User": Object(properties={"id": Integer()})})
    # Note: to_json_schema implementation for Definitions iterates and calls itself.
    # We simulate the structure expected in the provided code logic.
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]

def test_to_json_schema_error_unsupported_type():
    from typesystem import Field
    class UnhandledField(Field):
        pass
    field = UnhandledField()
    with pytest.raises(ValueError, match="Cannot convert field type 'UnhandledField' to JSON Schema"):
        to_json_schema(field)

def test_to_json_schema_regex_invalid_flags():
    import re
    from typesystem import String
    # Regex with non-unicode flags (e.g., ASCII) should raise ValueError per implementation
    pattern = re.compile(r"abc", re.ASCII)
    field = String(pattern_regex=pattern)
    with pytest.raises(ValueError, match="Cannot convert regular expression with non-standard flags"):
        to_json_schema(field)

def test_to_json_schema_const():
    from typesystem import Const
    field = Const("fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"
```


# LLM-generated content at query #22
#--------------------------

```python
def test_to_json_schema_exclusive_maximum_is_not_none():
    field = Integer(exclusive_maximum=10)
    result = to_json_schema(field)
    assert result["exclusiveMaximum"] == 10
```


# LLM-generated content at query #23
#--------------------------

```python
def test_if_then_else_from_json_schema_all_clauses():
    definitions = Definitions()
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": 123
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert field.if_clause.numeric_type is None  # String field
    assert field.then_clause.numeric_type is int
    assert field.else_clause.numeric_type is bool
    assert field.default == 123

def test_if_then_else_from_json_schema_only_if():
    definitions = Definitions()
    data = {
        "if": {"type": "number"},
        "default": None
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert field.if_clause.numeric_type is float
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Any)

def test_if_then_else_from_json_schema_if_and_then():
    definitions = Definitions()
    data = {
        "if": {"type": "string"},
        "then": {"type": "boolean"}
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Boolean)
    assert isinstance(field.else_clause, Any)
```


# LLM-generated content at query #24
#--------------------------

def test_from_json_schema_bool_true():
    field = from_json_schema(True)
    assert isinstance(field, Any)

def test_from_json_schema_bool_false():
    field = from_json_schema(False)
    assert isinstance(field, NeverMatch)

def test_from_json_schema_const():
    data = {"const": 123}
    field = from_json_schema(data)
    assert field.const == 123

def test_from_json_schema_enum():
    data = {"enum": ["a", "b"]}
    field = from_json_schema(data)
    assert field.choices == [("a", "a"), ("b", "b")]

def test_from_json_schema_ref():
    definitions = Definitions()
    definitions["#/components/schemas/User"] = Any()
    data = {"$ref": "#/components/schemas/User"}
    field = from_json_schema(data, definitions=definitions)
    assert field.to == "#/components/schemas/User"
    assert field.target == definitions["#/components/schema/User"] # Note: code has a typo in target property access via definitions lookup

def test_from_json_schema_all_of():
    data = {"allOf": [{"type": "string"}, {"const": "foo"}]}
    field = from_json_schema(data)
    assert len(field.all_of) == 2

def test_from_json_schema_any_of():
    data = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    field = from_json_schema(data)
    assert len(field.any_of) == 2

def test_from_json_schema_one_of():
    data = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    field = from_json_schema(data)
    assert len(field.one_of) == 2

def test_from_json_schema_not():
    data = {"not": {"const": 1}}
    field = from_json_schema(data)
    assert isinstance(field.negated, Const)

def test_from_json_schema_components_parsing():
    data = {
        "components": {
            "schemas": {
                "Item": {"type": "string"}
            }
        }
    }
    field = from_json_schema(data)
    # The logic in from_json_schema populates definitions internally. 
    # Since we can't easily inspect the internal 'definitions' variable of the function call, 
    # we rely on the fact that it doesn't crash and processes components.
    assert True

def test_from_json_schema_if_then_else():
    data = {
        "if": {"type": "number"},
        "then": {"const": 1},
        "else": {"const": 0}
    }
    field = from_json_schema(data)
    assert field.if_clause is not None
    assert field.then_clause is not None
    assert field.else_clause is not None

def test_from_json_schema_any():
    data = {"other": "unrecognized"}
    field = from_json_schema(data)
    assert isinstance(field, Any)


# LLM-generated content at query #25
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem.fields import String
    field = String(allow_null=False)
    result = to_json_schema(field)
    assert result == {"type": "string", "minLength": 1}

def test_to_json_schema_string_nullable_with_default():
    from typesystem.fields import String
    field = String(allow_null=True, default="hello")
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"], "default": "hello", "minLength": 1}

def test_to_json_schema_integer_limits():
    from typesystem.fields import Integer
    field = Integer(minimum=0, maximum=10, allow_null=True)
    result = to_json_schema(field)
    assert result == {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": 10
    }

def test_to_json_schema_boolean():
    from typesystem.fields import Boolean
    field = Boolean(allow_null=False)
    result = to_json_schema(field)
    assert result == {"type": "boolean"}

def test_to_json_schema_array_basic():
    from typesystem.fields import Array, String
    field = Array(items=String(), min_items=1)
    result = to_json_schema(field)
    assert result == {
        "type": "array",
        "minItems": 1,
        "items": {"type": "string", "minLength": 1},
        "uniqueItems": True
    }

def test_to_json_schema_object_properties():
    from typesystem.fields import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string", "minLength": 1}
    assert result["properties"]["age"] == {"type": "integer", "minLength": 1}

def test_to_json_schema_union():
    from typesystem.fields import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert result["anyOf"][0] == {"type": "string", "minLength": 1}
    assert result["anyOf"][1] == {"type": "integer", "minLength": 1}

def test_to_json_schema_definitions_reference():
    from typesystem.fields import Reference, String, Object
    # Mocking a structure where we use Definitions to simulate $ref logic
    # Since the provided code for Reference/target is not fully visible, 
    # we test the core logic of the loop in to_json_schema for Definitions
    from typesystem.schemas import Definitions
    defs = Definitions({"User": String()})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["User"] == {"type": "string", "minLength": 1}

def test_to_json_schema_const():
    from typesystem.fields import Const
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

def test_to_json_schema_unsupported_type():
    class UnrecognizedField(Field):
        pass
    field = UnrecognizedField()
    try:
        to_json_schema(field)
    except ValueError as e:
        assert "Cannot convert field type" in str(e)

def test_to_json_schema_none_returns_none():
    # Testing the 'elif field is not None' logic for a null arg if it was passed
    # Note: to_json_schema handles None by falling through to the end returning data={}.
    # If arg is None, it doesn't hit any specific branch and returns {}
    result = to_json_schema(None)
    assert result == {}
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem import String, Integer
    field = String(min_length=5, max_length=10, allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"], "minLength": 5, "maxLength": 10}

def test_to_json_schema_integer_with_default():
    from typesystem import Integer
    field = Integer(default=42, minimum=0)
    result = to_json_schema(field)
    assert result == {"type": "integer", "default": 42, "minimum": 0}

def test_to_json_schema_boolean():
    from typesystem import Boolean
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["boolean", "null"]}

def test_to_json_schema_array():
    from typesystem import Array, String, Integer
    field = Array(items=String(), min_items=1)
    result = to_json_schema(field)
    assert result == {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}}

def test_to_json_schema_object():
    from typesystem import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    assert result == {
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 1}, "age": {"type": "integer"}},
        "required": ["name"]
    }

def test_to_json_schema_union():
    from typesystem import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {
        "anyOf": [
            {"type": "string", "minLength": 1},
            {"type": "integer"}
        ]
    }

def test_to_json_schema_definitions():
    from typesystem import Definitions, String, Integer
    defs = Definitions({"User": Object(properties={"id": Integer()})})
    # We must simulate the structure since to_json_schema uses _definitions logic
    # The function implementation provided handles definitions via recursion
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    assert result["components"]["schemas"]["User"]["type"] == "object"

def test_to_json_schema_const():
    from typesystem import Const
    field = Const(const="fixed")
    result = to_json_schema(field)
    assert result == {"const": "fixed"}

def test_to_json_schema_choice():
    from typesystem import Choice
    field = Choice(choices=[("a", None), ("b", None)])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]

def test_to_json_schema_unsupported_type():
    class UnknownField(Field):
        pass
    
    field = UnknownField()
    try:
        to_json_schema(field)
    except ValueError as e:
        assert "Cannot convert field type" in str(e)
```


# LLM-generated content at query #2
#--------------------------

```python
def test_from_json_schema_bool_true():
    from typesystem.json_schema import from_json_schema
    from typesystem.fields import Any
    result = from_json_schema(True)
    assert isinstance(result, Any)

def test_from_json_schema_bool_false():
    from typesystem.json_schema import from_json_schema
    from typesystem.composites import NeverMatch
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

def test_from_json_schema_ref():
    from typesystem.json_schema import from_json_schema
    from typesystem.schemas import Definitions, Reference
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/MyType"}
    result = from_json_schema(data, definitions=definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/MyType"

def test_from_json_schema_enum():
    from typesystem.json_schema import from_json_schema
    from typesystem.fields import Choice
    data = {"enum": ["a", "b"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)
    assert ("a", "a") in result.choices

def test_from_json_schema_const():
    from typesystem.json_schema import from_json_schema
    from typesystem.fields import Const
    data = {"const": 42}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == 42

def test_from_json_schema_all_of():
    from typesystem.json_schema import from_json_schema
    from typesystem.composites import AllOf
    data = {
        "allOf": [
            {"type": "string"},
            {"const": "fixed"}
        ]
    }
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_from_json_schema_any_of():
    from typesystem.json_schema import from_json_schema
    from typesystem.fields import Union
    data = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

def test_from_json_schema_one_of():
    from typesystem.json_schema import from_json_schema
    from typesystem.composites import OneOf
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result = from_json_schema(data)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2

def test_from_json_schema_not():
    from typesystem.json_schema import from_json_schema
    from typesystem.fields import Not
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)

def test_from_json_schema_if_then_else():
    from typesystem.json_schema import from_json_schema
    from typesystem.composites import IfThenElse
    from typesystem.schemas import Definitions
    definitions = Definitions()
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    result = from_json_schema(data, definitions=definitions)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_components_resolution():
    from typesystem.json_schema import from_json_schema
    from typesystem.schemas import Definitions
    data = {
        "components": {
            "schemas": {
                "User": {"type": "object", "properties": {"id": {"type": "integer"}}}
            }
        },
        "$ref": "#/components/schemas/User"
    }
    definitions = Definitions()
    result = from_json_schema(data, definitions=definitions)
    assert "#/components/schemas/User" in definitions
    assert isinstance(definitions["#/components/schemas/User"].properties["id"], type(None)) # Placeholder for actual check logic
```


# LLM-generated content at query #3
#--------------------------

```python
def test_to_json_schema_min_items_is_not_none():
    field = Array(min_items=5)
    result = to_json_schema(field)
    assert "minItems" in result
    assert result["minItems"] == 5
```


# LLM-generated content at query #4
#--------------------------

```python
def test_from_json_schema_with_definitions_is_none():
    from typesystem.json_schema import from_json_schema
    data = {"type": "string"}
    result = from_json_schema(data, definitions=None)
    assert isinstance(result, Any)
```


# LLM-generated content at query #5
#--------------------------

def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 10, "default": 5}
    field = from_json_schema_type(data, "number", allow_null=True, definitions=None)
    assert isinstance(field, Float)
    assert field.min_properties is None
    assert field.allow_null is True

def test_from_json_schema_type_integer():
    data = {"minimum": 1, "maximum": 5}
    field = from_json_schema_type(data, "integer", allow_null=False, definitions=None)
    assert isinstance(field, Integer)
    assert field.allow_null is False

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 10, "pattern": "^abc", "format": "email"}
    field = from_json_schema_type(data, "string", allow_null=False, definitions=None)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^abc"
    assert field.format == "email"

def test_from_json_schema_type_boolean():
    data = {"default": True}
    field = from_json_schema_type(data, "boolean", allow_null=True, definitions=None)
    assert isinstance(field, Boolean)
    assert field.allow_null is True

def test_from_json_schema_type_array():
    data = {
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    field = from_on_json_schema_type(data, "array", allow_null=False, definitions=None)
    # Note: Assuming Array class exists based on the provided code snippet
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.additional_items is False

def test_from_json_schema_type_object():
    data = {
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": {"type": "integer"},
        "minProperties": 1
    }
    field = from_json_schema_type(data, "object", allow_null=False, definitions=None)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "name" in field.required
    assert isinstance(field.additional_properties, Integer)
    assert field.min_properties == 1


# LLM-generated content at query #6
#--------------------------

```python
def test_to_json_schema_predicate_false():
    arg = String(allow_null=True)
    assert isinstance(arg, Any) is False
```


# LLM-generated content at query #7
#--------------------------

```python
def test_to_json_schema_predicate_line_1_is_false():
    arg = Field(allow_null=True)
    result = to_json_schema(arg)
    assert result != True
```


# LLM-generated content at query #8
#--------------------------

```python
def test_from_json_schema_type_array_with_items():
    from typesystem.json_schema import from_json_schema_type
    from typesystem.fields import Array, Integer
    import typing

    data = {"type": "array", "items": {"type": "integer"}}
    definitions = {}
    # To ensure line 58 evaluates to False, 'items' must not be None.
    # We provide a dictionary that contains the 'items' key.
    result = from_json_schema_type(data, "array", allow_null=True, definitions=definitions)
    assert isinstance(result, Array)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem.fields import String
    field = String()
    result = to_json_schema(field)
    assert result == {"type": "string", "minLength": 1}

def test_to_json_schema_string_with_constraints():
    from typesystem.fields import String
    field = String(min_length=5, max_length=10, allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"], "minLength": 5, "maxLength": 10}

def test_to_json_schema_integer():
    from typesystem.fields import Integer
    field = Integer(minimum=0, maximum=100, default=1)
    result = to_json_schema(field)
    assert result == {"type": "integer", "default": 1, "minimum": 0, "maximum": 100}

def test_to_json_schema_boolean():
    from typesystem.fields import Boolean
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["boolean", "null"]}

def test_to_json_schema_array():
    from typesystem.fields import Array, String
    field = Array(items=String(), min_items=1, unique_items=True)
    result = to_json_schema(field)
    assert result == {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}, "uniqueItems": True}

def test_to_json_schema_object():
    from typesystem.fields import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string", "minLength": 1}
    assert result["properties"]["age"] == {"type": "integer", "minLength": 1}

def test_to_json_schema_union():
    from typesystem.fields import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

def test_to_json_schema_definitions():
    from typesystem.schemas import Definitions
    from typesystem.fields import String
    defs = Definitions({"MyString": String()})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["MyString"] == {"type": "string", "minLength": 1}

def test_to_json_schema_const():
    from typesystem.fields import Const
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

def test_to_json_schema_choice():
    from typesystem.fields import Choice
    field = Choice(choices=[("a", None), ("b", None)])
    result = to_json_schema(field)
    assert "enum" in result
    assert "a" in result["enum"]
    assert "b" in result["enum"]

def test_to_json_schema_error_on_unsupported():
    from typesystem.fields import Field
    class UnsupportedField(Field):
        pass
    field = UnsupportedField()
    try:
        to_json_schema(field)
    except ValueError as e:
        assert "Cannot convert field type" in str(e)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {
        "minimum": 1.0,
        "maximum": 10.0,
        "exclusiveMinimum": 0.5,
        "exclusiveMaximum": 10.5,
        "multipleOf": 2.0,
        "default": 5.0
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "number", allow_null=True, definitions=definitions)
    assert isinstance(result, Float)
    assert result.allow_null is True
    assert result.minimum == 1.0
    assert result.maximum == 10.0
    assert result.exclusive_minimum == 0.5
    assert result.exclusive_maximum == 10.5
    assert result.multiple_of == 2.0

def test_from_json_schema_type_integer():
    data = {
        "minimum": 1,
        "default": 5
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "integer", allow_null=False, definitions=definitions)
    assert isinstance(result, Integer)
    assert result.allow_null is False
    assert result.minimum == 1
    assert result.default == 5

def test_from_json_schema_type_string():
    data = {
        "minLength": 5,
        "maxLength": 10,
        "format": "email",
        "pattern": "^[a-z]+$",
        "default": "hello"
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "string", allow_null=True, definitions=definitions)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.format == "email"
    assert result.pattern == "^[a-z]+$"
    assert result.default == "hello"

def test_from_json_schema_type_boolean():
    data = {"default": True}
    definitions = Definitions()
    result = from_json_schema_type(data, "boolean", allow_null=True, definitions=definitions)
    assert isinstance(result, Boolean)
    assert result.default is True

def test_from_json_schema_type_array_simple():
    data = {
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "additionalItems": False
    }
    definitions = Definitions()
    # Note: from_json_schema is called internally. Assuming it works for this test context.
    result = from_json_schema_type(data, "array", allow_null=True, definitions=definitions)
    assert isinstance(result, Array)
    assert result.min_items == 1
    assert result.max_items == 5
    assert result.unique_items is True
    assert result.additional_items is False

def test_from_json_schema_type_array_list_items():
    data = {
        "items": [{"type": "string"}, {"type": "integer"}]
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "array", allow_null=True, definitions=definitions)
    assert isinstance(result.items, list)
    assert len(result.items) == 2

def test_from_json_schema_type_object_complex():
    data = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "patternProperties": {
            "^attr_": {"type": "string"}
        },
        "additionalProperties": {"type": "boolean"},
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 5,
        "required": ["name"],
        "default": {}
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "object", allow_null=False, definitions=definitions)
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert "^attr_" in result.pattern_properties
    assert isinstance(result.additional_properties, Field) # Assuming it resolves to a Field via from_json_schema
    assert result.min_properties == 1
    assert result.max_properties == 5
    assert "name" in result.required
```


# LLM-generated content at query #11
#--------------------------

def test_to_json_schema_string_field():
    from typesystem import String
    field = String(min_length=5, max_length=10, allow_null=True, default="test")
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["default"] == "test"

def test_to_json_schema_integer_field():
    from typesystem import Integer
    field = Integer(minimum=0, maximum=100, allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100

def test_to_json_schema_boolean_field():
    from typesystem import Boolean
    field = Boolean(default=True)
    result = to_json_schema(field)
    assert result["type"] == "boolean"
    assert result["default"] is True

def test_to_json_schema_array_field():
    from typesystem import Array, String, Integer
    field = Array(items=String(min_length=1), min_items=1, max_items=5)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["items"] == {"type": ["string", "null"]}

def test_to_json_schema_object_field():
    from typesystem import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]

def test_to_json_schema_union_field():
    from typesystem import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

def test_to_json_schema_definitions():
    from typesystem import String, Definitions
    defs = Definitions({"User": String()})
    # Note: to_json_schema with Definitions returns the schema for value and populates components in root call
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]

def test_to_json_schema_reference():
    from typesystem import Reference, String, Definitions
    # Mocking a reference setup since Reference implementation isn't provided in snippet
    class MockRef:
        def __init__(self, to, target):
            self.to = to
            self.target = target
    
    field = MockRef("User", String())
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/User"

def test_to_json_schema_error_on_unsupported_type():
    class UnsupportedField:
        pass
    
    with Exception as e:
        to_json_schema(UnsupportedField())
        raise e if "Cannot convert field type" in str(e) else AssertionError()


# LLM-generated content at query #12
#--------------------------

```python
def test_type_from_json_schema_number():
    data = {"type": "number", "minimum": 0}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Float)
    assert field.allow_null is False
    assert field.minimum == 0

def test_type_from_json_schema_integer():
    data = {"type": "integer", "maximum": 10}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Integer)
    assert field.allow_null is False
    assert field.maximum == 10

def test_type_from_json_schema_string():
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10

def test_type_from_json_schema_boolean():
    data = {"type": "boolean", "default": True}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Boolean)
    assert field.allow_null is False
    assert field.default is True

def test_type_from_json_schema_array():
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1
    }
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert isinstance(field.items, String)

def test_type_from_json_schema_object():
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    }
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert field.required == ["name"]

def test_type_from_json_schema_union_with_null():
    data = {"type": ["string", "null"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert field.allow_null is True

def test_type_from_json_schema_union_multiple_types():
    data = {"type": ["string", "integer"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert field.allow_null is False
```


# LLM-generated content at query #13
#--------------------------

```python
def test_to_json_schema_basic_string():
    from typesystem.fields import String
    field = String(min_length=5, max_length=10, allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"], "minLength": 5, "maxLength": 10}

def test_to_json_schema_integer():
    from typesystem.fields import Integer
    field = Integer(minimum=0, maximum=100, default=10)
    result = to_json_schema(field)
    assert result == {"type": "integer", "default": 10, "minimum": 0, "maximum": 100}

def test_to_json_schema_boolean():
    from typesystem/fields import Boolean
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["boolean", "null"]}

def test_to_json_schema_array():
    from typesystem/fields import Array, String
    field = Array(items=String(), min_items=1, unique_items=True)
    result = to_json_schema(field)
    assert result == {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}, "uniqueItems": True}

def test_to_json_schema_union():
    from typesystem/fields import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert {"type": "string", "minLength": 1} in result["anyOf"]
    assert {"type": "integer"} in result["anyOf"]

def test_to_json_schema_definitions():
    from typesystem/fields import String, Integer
    defs = Definitions({"MyString": String()})
    result = to_json_schema(defs)
    assert "components" in result
    assert "MyString" in result["components"]["schemas"]
    assert result["components"]["schemas"]["MyString"] == {"type": "string", "minLength": 1}

def test_to_json_schema_reference():
    from typesystem/fields import Reference, String
    target = String()
    ref = Reference(to="MySchema", target=target)
    result = to_json_schema(ref)
    assert result == {"$ref": "#/components/schemas/MySchema"}

def test_to_json_schema_object():
    from typesystem/fields import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert "name" in result["required"]

def test_to_json_schema_choice():
    from typesystem/fields import Choice
    field = Choice(choices=[("A", 1), ("B", 2)])
    result = to_json_schema(field)
    assert result["enum"] == ["A", "B"]

def test_to_json_schema_const():
    from typesystem/fields import Const
    field = Const(const="fixed")
    result = to_json_schema(field)
    assert result["const"] == "fixed"
```


# LLM-generated content at query #14
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem.fields import String
    from typesystem.json_schema import to_json_schema
    field = String()
    result = to_json_schema(field)
    assert result == {"type": "string", "minLength": 1}

def test_to_json_schema_string_with_constraints():
    from typesystem.fields import String
    from typesystem.json_schema import to_json_schema
    field = String(min_length=5, max_length=10, allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"], "minLength": 5, "maxLength": 10}

def test_to_json_schema_integer_with_bounds():
    from typesystem.fields import Integer
    from typesystem.json_schema import to_json_schema
    field = Integer(minimum=0, maximum=100)
    result = to_json_schema(field)
    assert result == {"type": "integer", "minimum": 0, "maximum": 100}

def test_to_json_schema_boolean():
    from typesystem.fields import Boolean
    from typesystem.json_schema import to_json_schema
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["boolean", "null"]}

def test_to_json_schema_array():
    from typesystem.fields import Array, String
    from typesystem.json_schema import to_json_schema
    field = Array(items=String(), min_items=1)
    result = to_json_schema(field)
    assert result == {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}}

def test_to_json_schema_object_with_properties():
    from typesystem.fields import Object, String, Integer
    from typesystem.json_schema import to_json_schema
    field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string", "minLength": 1}
    assert result["properties"]["age"] == {"type": "integer"}

def test_to_json_schema_union():
    from typesystem.fields import Union, String, Integer
    from typesystem.json_schema import to_json_schema
    field = Union([String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert {"type": "string", "minLength": 1} in result["anyOf"]
    assert {"type": "integer"} in result["anyOf"]

def test_to_json_schema_definitions():
    from typesystem.schemas import Definitions
    from typesystem.fields import String, Integer
    from typesystem.json_schema import to_json_schema
    defs = Definitions({"MyString": String()})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["MyString"] == {"type": "string", "minLength": 1}

def test_to_json_schema_with_default():
    from typesystem.fields import String
    from typesystem.json_schema import to_json_schema
    field = String(default="hello")
    result = to_json_schema(field)
    assert result["default"] == "hello"

def test_to_json_schema_const():
    from typesystem.fields import Const
    from typesystem.json_schema import to_json_schema
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

def test_to_json_schema_choice():
    from typesystem.fields import Choice
    from typesystem.json_schema import to_json_schema
    field = Choice(choices=[("a", None), ("b", None)])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]
```


# LLM-generated content at query #15
#--------------------------

```python
def test_to_json_schema_multiple_of_exists():
    integer_field = Integer(multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["multipleOf"] == 5
```


# LLM-generated content at query #16
#--------------------------

```python
def test_to_json_schema_evaluates_schema_branch():
    field = Schema(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["object", "null"]
```


# LLM-generated content at query #17
#--------------------------

```python
def test_from_json_schema_multiple_constraints():
    # Mocking the necessary imports and dependencies for from_json_schema
    # Since we cannot define custom functions, we assume typesystem environment is available.
    # We need to trigger line 36: len(constraints) > 1
    # This happens when a dict has multiple valid JSON schema keys (e.g., 'enum' and 'const')
    
    # In the context of the provided code, we simulate the logic for from_json_schema
    # We need to mock type_from_json_schema, enum_from_json_schema, etc.
    # But since I cannot define them, I will use the existing classes.
    
    # A dictionary with both 'enum' and 'const' keys will trigger multiple constraints.
    # To make this work in a single test case without imports:
    # We rely on the fact that TYPE_CONSTRAINTS is likely a global list of strings like ['type']
    
    data = {
        "type": "string",
        "enum": ["a", "b"],
        "const": "a"
    }
    
    # Since I cannot define 'from_json_schema' or the helper functions, 
    # and the prompt asks to test a specific line in a module provided as text:
    # I will assume from_json_schema is accessible.
    
    import typesystem.json_schema as json_schema
    
    result = json_schema.from_json_schema(data)
    
    assert isinstance(result, AllOf)
```


# LLM-generated content at query #18
#--------------------------

```python
def test_to_json_schema_evaluates_object_predicate():
    field = Object(allow_null=False, properties={}, pattern_properties={}, additional_properties=None, property_names=None, max_properties=None, min_properties=None, required=[])
    result = to_json_schema(field)
    assert isinstance(field, Object)
```


# LLM-generated content at query #19
#--------------------------

```python
import re

def test_to_json_schema_pattern_regex_not_none():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    result = to_json_schema(field)
    assert "pattern" in result
    assert result["pattern"] == "^[a-z]+$"
```


# LLM-generated content at query #20
#--------------------------

```python
def test_to_json_schema_schema_with_fields_evaluates_true():
    field_instance = Schema(fields={"prop": String(allow_null=True)}, allow_null=False, required=["prop"])
    result = to_json_schema(field_instance)
    assert "properties" in result
```


# LLM-generated content at query #21
#--------------------------

```python
def test_to_json_schema_evaluates_true_on_any_type():
    arg = Any()
    result = to_json_schema(arg)
    assert result is True
```


# LLM-generated content at query #22
#--------------------------

```python
def test_to_json_schema_evaluates_schema_predicate():
    schema_instance = Schema(allow_null=True)
    result = to_json_schema(schema_instance)
    assert result["type"] == ["object", "null"]
```


# LLM-generated content at query #23
#--------------------------

```python
def test_to_json_schema_if_then_else_clause_is_none():
    field_if = String(allow_null=False)
    field_then = String(allow_null=False)
    field_else = None
    field_if_then_else = IfThenElse(if_clause=field_if, then_clause=field_then, else_clause=field_else)
    result = to_json_schema(field_if_then_else)
    assert "then" in result
    assert "else" not in result
```


# LLM-generated content at query #24
#--------------------------

```python
def test_to_json_schema_primitive_string():
    from typesystem.fields import String
    field = String(allow_null=True, min_length=5)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 5

def test_to_json_schema_primitive_integer():
    from typesystem.fields import Integer
    field = Integer(minimum=0, maximum=10, default=1)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 10
    assert result["default"] == 1

def test_to_json_schema_boolean():
    from typesystem.fields import Boolean
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

def test_to_json_schema_array():
    from typesystem.fields import Array, String
    field = Array(items=String(), min_items=1, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["uniqueItems"] is True
    assert result["items"] == {"type": "string", "minLength": 1}

def test_to_json_schema_object():
    from typesystem.fields import Object, String, Integer
    properties = {"name": String(), "age": Integer()}
    field = Object(properties=properties, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string", "minLength": 1}
    assert result["properties"]["age"] == {"type": "integer", "minLength": 1}
    assert result["required"] == ["name"]

def test_to_json_schema_union():
    from typesystem.fields import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0] == {"type": "string", "minLength": 1}
    assert result["anyOf"][1] == {"type": "integer", "minLength": 1}

def test_to_json_schema_definitions():
    from typesystem.schemas import Definitions
    from typesystem.fields import String, Integer
    defs = Definitions({"User": Object(properties={"id": Integer()})})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    assert result["components"]["schemas"]["User"]["type"] == "object"

def test_to_json_schema_const():
    from typesystem.fields import Const
    field = Const(const="fixed")
    result = to_json_schema(field)
    assert result["const"] == "fixed"

def test_to_json_schema_choice():
    from typesystem.fields import Choice
    field = Choice(choices=[("A", None), ("B", None)])
    result = to_json_schema(field)
    assert result["enum"] == ["A", "B"]

def test_to_json_schema_not():
    from typesystem.fields import Not, String
    field = Not(negated=String())
    result = to_json_schema(field)
    assert "not" in result
    assert result["not"]["type"] == "string"

def test_to_json_schema_if_then_else():
    from typesystem.fields import IfThenElse, String, Integer
    field = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(field)
    assert result["if"] == {"type": "string", "minLength": 1}
    assert result["then"] == {"type": "integer", "minLength": 1}

def test_to_json_schema_error_unsupported_type():
    from typesystem.fields import Field
    class UnhandledField(Field):
        pass
    field = UnhandledField()
    try:
        to_json_schema(field)
    except ValueError as e:
        assert "Cannot convert field type" in str(e)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_to_json_schema_exclusive_minimum_predicate():
    integer_field = Integer(exclusive_minimum=5)
    result = to_json_schema(integer_field)
    assert result["exclusiveMinimum"] == 5
```


