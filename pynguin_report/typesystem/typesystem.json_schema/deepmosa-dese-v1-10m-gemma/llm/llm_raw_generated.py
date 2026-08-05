####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_from_json_schema_bool_true():
    from typesystem.json_schema import from_json_schema
    from typesystem.fields import Any
    field = from_json_schema(True)
    assert isinstance(field, Any)

def test_from_json_schema_bool_false():
    from typesystem.json_schema import from_json_schema
    from typesystem.composites import NeverMatch
    field = from_json_schema(False)
    assert isinstance(field, NeverMatch)

def test_from_json_schema_ref():
    from typesystem.json_schema import from_json_schema, Definitions
    from typesystem.schemas import Reference
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/MyType"}
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/MyType"

def test_from_json_schema_enum():
    from typesystem.json_schema import from_json_schema, Definitions
    from typesystem.fields import Choice
    data = {"enum": ["a", "b"]}
    field = from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", ""]) # Note: actual implementation detail in provided code for tuple conversion

def test_from_json_schema_const():
    from typesystem.json_schema import from_json_schema, Definitions
    from typesystem.fields import Const
    data = {"const": 123}
    field = from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Const)
    assert field.const == 123

def test_from_json_schema_all_of():
    from typesystem.json_schema import from_json_schema, Definitions
    from typesystem.composites import AllOf
    data = {
        "allOf": [{"type": "string"}, {"const": "foo"}]
    }
    field = from_json_schema(data, definitions=Definitions())
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2

def test_from_json_schema_components_definitions():
    from typesystem.json_schema import from_json_schema, Definitions
    from typesystem.schemas import Reference
    data = {
        "components": {
            "schemas": {
                "User": {"type": "string"}
            }
        },
        "$ref": "#/components/schemas/User"
    }
    field = from_json_schema(data)
    assert isinstance(field, Reference)
    assert "#/components/schemas/User" in field.definitions
```


# LLM-generated content at query #2
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 10, "default": 5}
    definitions = Definitions()
    field = from_json_schema_type(data, "number", allow_null=True, definitions=definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.default == 5
    assert field.allow_null is True

def test_from_json_schema_type_integer():
    data = {"minimum": 1, "exclusiveMinimum": 0}
    definitions = Definitions()
    field = from_json_schema_type(data, "integer", allow_null=False, definitions=definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.exclusive_minimum == 0
    assert field.allow_null is False

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 10, "pattern": "^abc", "format": "email"}
    definitions = Definitions()
    field = from_json_schema_type(data, "string", allow_null=False, definitions=definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^abc"
    assert field.format == "email"
    assert field.allow_blank is False

def test_from_json_schema_type_boolean():
    data = {"default": True}
    definitions = Definitions()
    field = from_json_schema_type(data, "boolean", allow_null=True, definitions=definitions)
    assert isinstance(field, Boolean)
    assert field.default is True
    assert field.allow_null is True

def test_from_json_schema_type_array():
    data = {
        "items": {"type": "string"},
        "additionalItems": {"type": "integer"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    definitions = Definitions()
    field = from_json_schema_type(data, "array", allow_null=False, definitions=definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert isinstance(field.additional_items, Integer)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items is True

def test_from_json_schema_type_object():
    data = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "patternProperties": {
            "^id_": {"type": "integer"}
        },
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "required": ["name"],
        "minProperties": 1
    }
    definitions = Definitions()
    field = from_json_schema_type(data, "object", allow_null=True, definitions=definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert isinstance(field.pattern_properties["^id_"], Integer)
    assert field.additional_properties is False
    assert isinstance(field.property_names, String)
    assert "name" in field.required
    assert field.min_properties == 1
```


# LLM-generated content at query #3
#--------------------------

```python
def test_to_json_schema_boolean_field():
    from typesystem.fields import Boolean
    field = Boolean(allow_null=True, default="not_actually_default")
    # Note: The provided code has a bug where it sets default to 'not_actually_default' 
    # if it is not NO_DEFAULT, but Boolean constructor doesn't handle string defaults via logic.
    # We rely on the behavior of get_standard_properties and Field.__init__.
    # Since we can't fix the source, we test what the code does.
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

def test_to_json_schema_integer_field():
    from typesystem.fields import Integer
    field = Integer(minimum=0, maximum=10, allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 10

def test_to_json_schema_string_field():
    from typesystem.fields import String
    field = String(min_length=5, max_length=10)
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 5
    assert result["maxLength"] == 10

def test_to_json_schema_array_field():
    from typesystem.fields import Array, String
    field = Array(items=String(), min_items=1)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["items"] == {"type": "string"}

def test_to_json_schema_object_field():
    from typesystem.fields import Object, String, Integer
    props = {"name": String(), "age": Integer()}
    field = Object(properties=props, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["properties"]["name"] == {"type": "string"}
    assert "age" in result["properties"]
    assert result["properties"]["age"] == {"type": "integer"}
    assert "name" in result["required"]

def test_to_json_schema_union_field():
    from typesystem.fields import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0] == {"type": "string"}
    assert result["anyOf"][1] == {"type": "integer"}

def test_to_json_schema_definitions():
    from typesystem.schemas import Definitions
    from typesystem.fields import String
    defs = Definitions({"MyString": String()})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["MyString"] == {"type": "string"}

def test_to_json_schema_const_field():
    from typesystem.fields import Const
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

def test_to_json_schema_choice_field():
    from typesystem.fields import Choice
    field = Choice(choices=[("A", None), ("B", None)])
    result = to_json_schema(field)
    assert "enum" in result
    assert "A" in result["enum"]
    assert "B" in result["enum"]
```


# LLM-generated content at query #4
#--------------------------

```python
def test_to_json_schema_evaluates_true_at_line_122():
    test_schema = Schema(fields={"test_field": String(allow_null=True)})
    result = to_json_schema(test_schema)
    assert "properties" in result
    assert "test_field" in result["properties"]
```


# LLM-generated content at query #5
#--------------------------

def test_from_json_schema_type_number():
    data = {"minimum": 1, "maximum": 10, "default": 5}
    field = from_json_schema_type(data, "number", allow_null=False, definitions=Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.default == 5

def test_from_json_schema_type_integer():
    data = {"minimum": 0, "exclusiveMinimum": 0}
    field = from_json_schema_type(data, "integer", allow_null=True, definitions=Definitions())
    assert isinstance(field, Integer)
    assert field.allow_null is True
    assert field.minimum == 0
    assert field.exclusive_minimum == 0

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 10, "pattern": "^abc", "format": "email"}
    field = from_json_schema_type(data, "string", allow_null=False, definitions=Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^abc"
    assert field.format == "email"

def test_from_json_schema_type_string_blank_allowed():
    data = {"minLength": 0}
    field = from_json_schema_type(data, "string", allow_null=False, definitions=Definitions())
    assert field.allow_blank is True

def test_from_json_schema_type_boolean():
    data = {"default": True}
    field = from_json_schema_type(data, "boolean", allow_null=True, definitions=Definitions())
    assert isinstance(field, Boolean)
    assert field.default is True
    assert field.allow_null is True

def test_from_json_schema_type_array():
    data = {
        "items": {"type": "string"},
        "additionalItems": {"type": "integer"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    field = from_json_schema_type(data, "array", allow_null=False, definitions=Definitions())
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items is True
    assert isinstance(field.items, String)
    assert isinstance(field.additional_items, Integer)

def test_from_json_schema_type_object():
    data = {
        "properties": {"name": {"type": "string"}},
        "patternProperties": {"^id_": {"type": "integer"}},
        "additionalProperties": False,
        "required": ["name"],
        "minProperties": 1
    }
    defs = Definitions()
    field = from_json_schema_type(data, "object", allow_null=False, definitions=defs)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert "^id_" in field.pattern_properties
    assert isinstance(field.pattern_properties["^id_"], Integer)
    assert field.additional_properties is False
    assert "name" in field.required
    assert field.min_properties == 1


# LLM-generated content at query #6
#--------------------------

```python
def test_to_json_schema_exclusive_minimum_is_not_none():
    integer_field = Integer(exclusive_minimum=5)
    result = to_json_schema(integer_field)
    assert result["exclusiveMinimum"] == 5
```


# LLM-generated content at query #7
#--------------------------

def test_test_type_from_json_schema_string():
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.allow_null is False

def test_test_type_from_json_schema_integer():
    data = {"type": "integer", "minimum": 0}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.allow_null is False

def test_test_type_from_json_schema_union():
    data = {"type": ["string", "integer"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2

def test_test_type_from_json_schema_null_allowed():
    data = {"type": ["string", "null"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert field.allow_null is True

def test_test_type_from_json_schema_empty_type_list():
    data = {"type": []}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    # get_valid_types defaults to all types if empty. 
    # Since it contains 'string', 'integer' etc., it becomes a Union.
    assert isinstance(field, Union)

def test_test_type_from_json_schema_const_none():
    data = {"type": []}
    # If type_strings is empty after processing nulls (e.g. only 'null' provided)
    # The function returns Const(None) if allow_null is True
    data_only_null = {"type": ["null"]}
    definitions = Definitions()
    field = type_from_json_schema(data_only_null, definitions)
    assert isinstance(field, Const)
    assert field.const is None


# LLM-generated content at query #8
#--------------------------

```python
def test_to_json_schema_pattern_properties_evaluates_true():
    pattern_field = Object(pattern_properties={"^abc$": String(allow_null=True)})
    result = to_json_schema(pattern_field)
    assert "patternProperties" in result
```


# LLM-generated content at query #9
#--------------------------

```python
def test_to_json_schema_evaluates_line_46_true():
    integer_field = Integer(allow_null=True)
    result = to_json_schema(integer_field)
    assert result["type"] == ["integer", "null"]

def test_to_json_schema_evaluates_line_46_true_no_null():
    integer_field = Integer(allow_null=False)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"

def test_to_json_schema_evaluates_line_46_true_float():
    float_field = Float(allow_null=True)
    result = to_json_schema(float_field)
    assert result["type"] == ["number", "null"]

def test_to_json_schema_evaluates_line_46_true_decimal():
    decimal_field = Decimal(allow_null=False)
    result = to_json_schema(decimal_field)
    assert result["type"] == "number"
```


# LLM-generated content at query #10
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem import String, Integer, Boolean, Array, Object, Choice, Const, Union, Any, NeverMatch
    field = String(allow_null=True, default="hello")
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["default"] == "hello"
    assert result["minLength"] == 1

def test_to_json_schema_integer_constraints():
    from typesystem import Integer
    field = Integer(minimum=0, maximum=10, allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 10

def test_to_json_schema_boolean():
    from typesystem import Boolean
    field = Boolean(default=True)
    result = to_json_schema(field)
    assert result["type"] == "boolean"
    assert result["default"] is True

def test_to_json_schema_array():
    from typesystem import Array, Integer
    field = Array(items=Integer, min_items=1, max_items=5, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] is True
    assert result["items"] == {"type": "integer"}

def test_to_json_schema_object():
    from typesystem import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert "name" in result["required"]

def test_to_json_schema_choice():
    from typesystem import Choice
    field = Choice(choices=[("a", 1), ("b", 2)])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]

def test_to_json_schema_const():
    from typesystem import Const
    field = Const(value="static")
    result = to_json_schema(field)
    assert result["const"] == "static"

def test_to_json_schema_union():
    from typesystem import Union, String, Integer
    field = Union([String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

def test_to_json_schema_any_and_nevermatch():
    from typesystem import Any, NeverMatch
    assert to_json_schema(Any()) is True
    assert to_json_schema(NeverMatch()) is False

def test_to_json_schema_definitions_handling():
    from typesystem import Definitions, String, Integer
    defs = Definitions({"User": String()})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["User"] == {"type": "string", "minLength": 1}
```


# LLM-generated content at query #11
#--------------------------

```python
def test_from_json_schema_bool_true():
    from_json_schema.py:
    assert isinstance(from_json_schema(True), Any)

def test_from_json_schema_bool_false():
    from_json_schema.py:
    assert isinstance(from_json_schema(False), NeverMatch)

def test_from_json_schema_ref():
    from_json_schema.py:
    definitions = Definitions({"#/components/schemas/MyType": Any()})
    data = {"$ref": "#/components/schemas/MyType"}
    result = from_json_schema(data, definitions=definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/MyType"

def test_from_json_schema_components_parsing():
    from_json_schema.py:
    data = {
        "components": {
            "schemas": {
                "User": {"type": "string"}
            }
        }
    }
    result = from_json_schema(data)
    # Since type_from_json_schema is called internally, it should result in a String field (or similar)
    # We check if the ref exists in the internal definitions by observing behavior or structure
    assert hasattr(result, 'validate')

def test_from_json_schema_enum():
    from_json_schema.py:
    data = {"enum": ["A", "B"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)
    assert result.choices == [("A", "A"), ("B", "B")]

def test_from_json_schema_const():
    from_json_schema.py:
    data = {"const": 123}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == 123

def test_from_json_schema_all_of():
    from_json_schema.py:
    data = {
        "allOf": [
            {"type": "string"},
            {"enum": ["foo"]}
        ]
    }
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_any_of():
    from_json_schema.py:
    data = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result = from_json_schema(data)
    assert isinstance(result, Union)

def test_from_json_schema_one_of():
    from_json_schema.py:
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result = from_json_schema(data)
    assert isinstance(result, OneOf)

def test_from_json_schema_not():
    from_json_schema.py:
    data = {
        "not": {"type": "string"}
    }
    result = from_json_schema(data)
    assert isinstance(result, Not)

def test_from_json_schema_if_then_else():
    from_json_schema.py:
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"}
    }
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_any():
    from_json_schema.py:
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_to_json_schema_basic_string():
    from typesystem.fields import String
    field = String(allow_null=True, min_length=5)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 5

def test_to_json_schema_integer_with_defaults():
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

def test_to_json_schema_object_properties():
    from typesystem.fields import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string", "minLength": 1}
    assert result["properties"]["age"] == {"type": "integer"}

def test_to_json_schema_union():
    from typesystem.fields import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0] == {"type": "string", "minLength": 1}
    assert result["anyOf"][1] == {"type": "integer"}

def test_to_json_schema_definitions():
    from typesystem import Definitions
    from typesystem.fields import String
    defs = Definitions({"User": String()})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["User"] == {"type": "string", "minLength": 1}

def test_to_json_schema_const():
    from typesystem.fields import Const
    field = Const(const="fixed")
    result = to_json_schema(field)
    assert result["const"] == "fixed"

def test_to_json_schema_choice():
    from typesystem.fields import Choice
    field = Choice(choices=[("a", None), ("b", None)])
    result = to_json_schema(field)
    assert "enum" in result
    assert "a" in result["enum"]
    assert "b" in result["enum"]

def test_to_json_schema_float():
    from typesystem.fields import Float
    field = Float(maximum=1.5)
    result = to_json_schema(field)
    assert result["type"] == "number"
    assert result["maximum"] == 1.5

def test_to_json_schema_error_on_unknown_type():
    from typesystem.fields import Field
    class UnknownField(Field):
        pass
    field = UnknownField()
    try:
        to_json_schema(field)
    except ValueError as e:
        assert "Cannot convert field type" in str(e)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_if_then_else_from_json_schema_with_all_clauses():
    definitions = Definitions()
    if_data = {"type": "string"}
    then_data = {"type": "number"}
    else_data = {"type": "boolean"}
    data = {
        "if": if_data,
        "then": then_data,
        "else": else_data,
        "default": 123
    }

    # Note: from_json_schema is needed because it's called inside if_then_else_from_json_schema
    # and it handles the recursive construction of fields.
    import typesystem.json_schema as jss
    
    result = jss.if_then_else_from_json_schema(data, definitions)

    assert isinstance(result, jss.IfThenElse)
    assert result.if_clause.validate("test") is not None # String matches
    assert result.then_clause.validate(123) is not None # Number matches
    assert result.else_clause.validate(True) is not None # Boolean matches
    assert result.default == 123

def test_if_then_else_from_json_schema_with_only_if_and_then():
    definitions = Definitions()
    if_data = {"type": "string"}
    then_data = {"type": "number"}
    data = {
        "if": if_data,
        "then": then_data
    }
    
    import typesystem.json_schema as jss
    result = jss.if_then_else_from_json_schema(data, definitions)

    assert isinstance(result, jss.IfThenElse)
    # When 'then' is not provided, it defaults to Any() per code logic
    # We check if the logic flows to then_clause correctly
    assert result.then_clause.validate(123) == 123

def test_if_then_else_from_json_schema_with_only_if_and_else():
    definitions = Definitions()
    if_data = {"type": "string"}
    else_data = {"type": "boolean"}
    data = {
        "if": if_data,
        "else": else_data
    }

    import typesystem.json_schema as jss
    result = jss.if_then_else_from_json_schema(data, definitions)

    assert isinstance(result, jss.IfThenElse)
    # When 'then' is not provided, it defaults to Any()
    assert result.then_clause.validate(555) == 555
```


# LLM-generated content at query #14
#--------------------------

```python
def test_from_json_schema_with_components_schemas_dict():
    from typesystem.json_schema import from_json_schema
    from typesystem.fields import Any
    from typesystem.schemas import Definitions

    data = {
        "components": {
            "schemas": {
                "MySchema": {"type": "string"}
            }
        }
    }
    definitions = Definitions()
    result = from_json_schema(data, definitions=definitions)
    
    assert "#/components/schemas/MySchema" in definitions
    assert isinstance(result, Any)
```


# LLM-generated content at query #15
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"minimum": 5, "maximum": 10, "default": 7}
    field = from_json_schema_type(data, "number", allow_null=True, definitions=None)
    assert isinstance(field, Float)
    assert field.min_items is None  # Note: checking attributes based on provided Float/Integer logic
    assert field.allow_null is True

def test_from_json_schema_type_integer():
    data = {"minimum": 1, "exclusiveMinimum": 0}
    field = from_json_schema_type(data, "integer", allow_null=False, definitions=None)
    assert isinstance(field, Integer)
    assert field.allow_null is False

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 10, "pattern": "^abc", "format": "email"}
    field = from_json_schema_type(data, "string", allow_null=True, definitions=None)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^abc"

def test_from_json_schema_type_boolean():
    data = {"default": True}
    field = from_json_schema_type(data, "boolean", allow_null=False, definitions=None)
    assert isinstance(field, Boolean)
    assert field.default is True

def test_from_json_schema_type_array_simple():
    data = {"items": {"type": "string"}, "minItems": 1, "uniqueItems": True}
    field = from_json_schema_type(data, "array", allow_null=False, definitions=None)
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.unique_items is True

def test_from_json_schema_type_array_list_items():
    data = {"items": [{"type": "integer"}, {"type": "string"}]}
    field = from_json_schema_type(data, "array", allow_null=False, definitions=None)
    assert isinstance(field, Array)
    assert len(field.items) == 2
    assert isinstance(field.items[0], Integer)
    assert isinstance(field.items[1], String)

def test_from_json_schema_type_array_additional_items():
    data = {"items": {"type": "integer"}, "additionalItems": {"type": "string"}}
    field = from_json_schema_type(data, "array", allow_null=False, definitions=None)
    assert isinstance(field.additional_items, String)

def test_from_json_schema_type_object_properties():
    data = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    field = from_json_schema_type(data, "object", allow_null=False, definitions=None)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert "age" in field.properties
    assert isinstance(field.properties["age"], Integer)
```


# LLM-generated content at query #16
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem.fields import String
    field = String(allow_null=True, default="hello")
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["default"] == "hello"

def test_to_json_schema_integer_constraints():
    from typesystem.fields import Integer
    field = Integer(minimum=0, maximum=10, allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 10

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
    assert result["items"] == {"type": "string", "minLength": 1}
    assert result["minItems"] == 1

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
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0] == {"type": "string", "minLength": 1}

def test_to_json_schema_definitions_and_ref():
    from typesystem.schemas import Definitions
    from typesystem.fields import Reference, String
    # Mocking Reference and target for the scope of this test
    class MockRef:
        def __init__(self, to, target):
            self.to = to
            self.target = target
    class MockString(String):
        pass
    
    defs = Definitions({"MySchema": MockString(title="Test")})
    # We need a way to trigger the Reference logic in to_json_schema
    # Since Reference is not provided in the snippet, we assume it exists as per usage
    from typesystem.fields import Field
    class Reference(Field):
        def __init__(self, to, target):
            super().__init__()
            self.to = to
            self.target = target

    ref_field = Reference(to="MySchema", target=MockString())
    # Using a dict as arg to simulate the Definitions logic in to_json_schema
    result = to_json_schema(defs)
    assert "components" in result
    assert "MySchema" in result["components"]["schemas"]

def test_to_json_schema_error_unsupported_type():
    from typesystem.fields import Field
    class UnhandledField(Field):
        pass
    field = UnhandledField()
    try:
        to_json_schema(field)
    except ValueError as e:
        assert "Cannot convert field type" in str(e)

def test_to_json_schema_const():
    from typesystem.fields import Const
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"
```


# LLM-generated content at query #17
#--------------------------

```python
def test_all_of_from_json_schema_valid_input():
    definitions = Definitions()
    data = {
        "allOf": [
            {"type": "string"},
            {"type": "integer"}
        ],
        "default": 123
    }
    # Note: from_json_schema is used internally by all_of_from_json_schema.
    # We assume from_json_schema and related helpers are available in the same scope as the module.
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_all_of_from_json_schema_empty_allOf():
    definitions = Definitions()
    data = {"allOf": []}
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 0

def test_all_of_from_json_schema_with_default():
    definitions = Definitions()
    data = {
        "allOf": [{"type": "boolean"}],
        "default": True
    }
    result = all_of_from_json_schema(data, definitions)
    assert result.default == True
```


# LLM-generated content at query #18
#--------------------------

```python
def test_to_json_schema_string_field_evaluates_true():
    field = String(allow_null=True, allow_blank=False, min_length=1, max_length=10, pattern_regex=re.compile(r"^[a-z]+$"), format="email")
    result = to_json_schema(field)
    assert isinstance(field, String)
    assert result["type"] == ["string", "null"]
```


# LLM-generated content at query #19
#--------------------------

```python
def test_from_json_schema_multiple_constraints_returns_all_of():
    from typesystem.json_schema import from_json_schema
    from typesystem.fields import Const, Any
    # We need to mock the behavior of the other functions (type_from_json_schema, etc.)
    # by providing a dictionary that triggers at least two constraint checks.
    # Since we don't have the implementation of those helpers, 
    # we assume they return a Field object when given valid JSON schema keys.
    # We use 'enum' and 'const' which are known to be in the logic.
    # Note: This test assumes TYPE_CONSTRAINTS is a global list containing some strings.
    data = {"enum": ["a", "b"], "const": "a"}
    
    # Since we cannot easily mock the internal function calls (type_from_json_schema, etc.) 
    # without 'unittest.mock', and I am forbidden from using imports other than the ones 
    # provided in the context, this test relies on the existence of these functions 
    # in the same module being tested.
    
    # To make len(constraints) > 1, we need two keys that are both present in TYPE_CONSTRAINTS or enum/const/etc.
    # Let's assume 'enum' and 'const' exist in the logic.
    result = from_json_schema(data)
    
    from typesystem.composites import AllOf
    assert isinstance(result, AllOf)
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"minimum": 10, "maximum": 20, "exclusiveMinimum": 5, "exclusiveMaximum": 25, "multipleOf": 2, "default": 15}
    field = from_json_schema_type(data, "number", allow_null=True, definitions=None)
    assert isinstance(field, Float)
    assert field.min_length is None  # checking attribute presence via proxy if available or just verifying it doesn't crash

def test_from_json_schema_type_integer():
    data = {"minimum": 1, "maximum": 5, "default": 3}
    field = from_json_schema_type(data, "integer", allow_null=False, definitions=None)
    assert isinstance(field, Integer)

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 10, "format": "email", "pattern": "^[a-z]+$", "default": "test"}
    field = from_json_schema_type(data, "string", allow_null=False, definitions=None)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-z]+$"

def test_from_json_schema_type_string_empty():
    data = {"minLength": 0}
    field = from_json_schema_type(data, "string", allow_null=True, definitions=None)
    assert field.allow_blank is True

def test_from_json_schema_type_boolean():
    data = {"default": True}
    field = from_json_schema_type(data, "boolean", allow_null=False, definitions=None)
    assert isinstance(field, Boolean)

def test_from_json_schema_type_array_simple():
    data = {"items": {"type": "string"}, "minItems": 1, "maxItems": 5, "uniqueItems": True}
    field = from_json_schema_type(data, "array", allow_null=False, definitions=None)
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items is True

def test_from_json_schema_type_array_complex():
    data = {
        "items": [{"type": "string"}, {"type": "integer"}],
        "additionalItems": {"type": "boolean"},
        "minItems": 2
    }
    field = from_json_schema_type(data, "array", allow_null=False, definitions=None)
    assert isinstance(field.items, list)
    assert len(field.items) == 2
    assert isinstance(field.additional_items, Boolean)

def test_from_json_schema_type_object_simple():
    data = {
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1
    }
    field = from_json_schema_type(data, "object", allow_null=False, definitions=None)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert field.min_properties == 1

def test_from_json_schema_type_object_advanced():
    data = {
        "properties": {"id": {"type": "integer"}},
        "patternProperties": {"^prefix_": {"type": "string"}},
        "additionalProperties": False,
        "propertyNames": {"minLength": 3}
    }
    field = from_json_schema_type(data, "object", allow_null=True, definitions=None)
    assert field.properties["id"].numeric_type == int
    assert field.pattern_properties["^prefix_"].__class__ is String
    assert field.additional_properties is False
    assert field.property_names.min_length == 3
```


# LLM-generated content at query #2
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem.fields import String
    field = String()
    result = to_json_schema(field)
    assert result == {"type": "string", "minLength": 1}

def test_to_json_schema_string_with_attributes():
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
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
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
    assert result["anyOf"][0] == {"type": "string", "minLength": 1}
    assert result["anyOf"][1] == {"type": "integer", "minLength": 1}

def test_to_json_schema_definitions():
    from typesystem.schemas import Definitions
    from typesystem.fields import String
    defs = Definitions({"User": String()})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["User"] == {"type": "string", "minLength": 1}

def test_to_json_schema_const():
    from typesystem.fields import Const
    field = Const(value="fixed")
    result = to_json_schema(field)
    assert result["const"] == "fixed"

def test_to_json_schema_choice():
    from typesystem.fields import Choice
    field = Choice(choices=[("a", None), ("b", None)])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]

def test_to_json_schema_error_unsupported_type():
    class UnsupportedField:
        pass
    field = UnsupportedField()
    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(field)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_to_json_schema_string_field():
    from typesystem import String, Integer, Boolean, Float, Decimal
    import re

    field = String(min_length=5, max_length=10, allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["string", "null"], "minLength": 5, "maxLength": 10}

def test_to_json_schema_integer_field():
    from typesystem import Integer
    field = Integer(minimum=0, maximum=100, default=10)
    schema = to_json_schema(field)
    assert schema == {"type": "integer", "default": 10, "minimum": 0, "maximum": 100}

def test_to_json_schema_boolean_field():
    from typesystem import Boolean
    field = Boolean(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["boolean", "null"]}

def test_to_json_schema_array_field():
    from typesystem import Array, String, Integer
    field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    schema = to_json_schema(field)
    assert schema == {
        "type": "array",
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string", "minLength": 1},
        "uniqueItems": True
    }

def test_to_json_schema_choice_field():
    from typesystem import Choice
    field = Choice(choices=[("a", "Alpha"), ("b", "Beta")])
    schema = to_json_schema(field)
    assert schema == {"enum": ["a", "b"]}

def test_to_json_schema_const_field():
    from typesystem import Const
    field = Const(value="fixed")
    schema = to_json_schema(field)
    assert schema == {"const": "fixed"}

def test_to_json_schema_union_field():
    from typesystem import Union, String, Integer
    field = Union([String(), Integer()])
    schema = to_json_schema(field)
    assert schema == {
        "anyOf": [
            {"type": "string", "minLength": 1},
            {"type": "integer"}
        ]
    }

def test_to_json_schema_definitions():
    from typesystem import String, Integer, Definitions
    defs = Definitions({"MyString": String(), "MyInt": Integer()})
    # to_json_schema with Definitions returns the dict of schemas
    schema = to_json_schema(defs)
    assert "MyString" in schema
    assert schema["MyString"] == {"type": "string", "minLength": 1}
    assert "MyInt" in schema
    assert schema["MyInt"] == {"type": "integer"}

def test_to_json_schema_root_with_definitions():
    from typesystem import String, Definitions
    defs = Definitions({"Shared": String()})
    # When passing a Field and an existing _definitions dict as root
    field = String()
    schema = to_json_schema(field, _definitions={"Shared": String()})
    assert "components" in schema
    assert "schemas" in schema["components"]
    assert schema["components"]["schemas"]["Shared"] == {"type": "string", "minLength": 1}

def test_to_json_schema_error_on_unsupported_type():
    from typesystem import Field
    class UnknownField(Field):
        pass
    
    field = UnknownField()
    try:
        to_json_schema(field)
    except ValueError as e:
        assert "Cannot convert field type" in str(e)

def test_get_standard_properties_with_default():
    from typesystem import Integer
    field = Integer(default=42)
    props = get_standard_properties(field)
    assert props == {"default": 42}

def test_get_standard_properties_no_default():
    from typesystem import Integer
    field = Integer()
    props = get_standard_properties(field)
    assert props == {}
```


# LLM-generated content at query #4
#--------------------------

```python
def test_to_json_schema_evaluates_format_predicate_true():
    import re
    field = String(format="email", allow_null=False, allow_blank=True)
    result = to_json_schema(field)
    assert result["format"] == "email"
```


# LLM-generated content at query #5
#--------------------------

```python
def test_to_json_schema_object_predicate_evaluates_to_true():
    field_obj = Object(allow_null=False, properties={}, pattern_properties={}, additional_properties=None, property_names=None, max_properties=None, min_properties=None, required=[])
    result = to_json_schema(field_obj)
    assert isinstance(result, dict)
```


# LLM-generated content at query #6
#--------------------------

```python
def test_to_json_schema_evaluates_schema_branch():
    schema_field = Schema(allow_null=False, fields={}, required=[])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
```


# LLM-generated content at query #7
#--------------------------

def test_if_then_else_from_json_schema_full_clauses():
    import typing
    from typesystem.schemas import Definitions
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    
    # Mocking the necessary components for the test context
    class MockField:
        def __init__(self, name):
            self.name = name
        def validate_or_error(self, value):
            return value, None

    # We need to mock from_json_schema as it is called within if_then_else_from_json_schema
    # Since we cannot redefine functions in the test, we rely on the implementation's behavior.
    # However, since the instruction forbids custom functions/classes for logic, 
    # and we must use the provided code, we assume from_json_schema is available in scope.
    
    data = {
        "if": {"type": "string"},
        "then": {"const": "yes"},
        "else": {"const": "no"},
        "default": "maybe"
    }
    definitions = Definitions()
    
    # This test assumes from_json_schema is in the same module or accessible.
    # Because we cannot use 'import', we assume the environment allows execution of the provided snippet.
    result = if_then_else_from_json_schema(data, definitions)
    
    assert isinstance(result, IfThenElse)
    assert result.if_clause.validate_or_error is not None
    assert result.then_clause.validate_or_error is not None
    assert result.else_clause.validate_or_error is not None

def test_if_then_else_from_json_schema_no_else():
    import typing
    from typesystem.schemas import Definitions
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any

    data = {
        "if": {"type": "string"},
        "then": {"type": "string"}
    }
    definitions = Definitions()
    
    result = if_then_else_from_json_schema(data, definitions)
    
    assert isinstance(result, IfThenElse)
    # According to the code: then_clause is from_json_schema, else_clause is Any()
    assert result.else_clause.validate(None) is None

def test_if_then_else_from_json_schema_no_then():
    import typing
    from typesystem.schemas import Definitions
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any

    data = {
        "if": {"type": "string"},
        "else": {"type": "string"}
    }
    definitions = Definitions()
    
    result = if_then_else_from_json_schema(data, definitions)
    
    assert isinstance(result, IfThenElse)
    # According to the code: then_clause is Any()
    assert result.then_clause.validate("anything") == "anything"


# LLM-generated content at query #8
#--------------------------

```python
def test_to_json_schema_evaluates_object_branch():
    obj_field = Object(allow_null=False)
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
```


# LLM-generated content at query #9
#--------------------------

```python
def test_to_json_schema_if_then_else_then_clause_is_none():
    field = IfThenElse(if_clause=String(allow_null=True), then_clause=None, else_clause=None)
    result = to_json_schema(field)
    assert "then" not in result
```


# LLM-generated content at query #10
#--------------------------

```python
def test_to_json_schema_min_properties_predicate():
    field = Object(min_properties=5)
    result = to_json_schema(field)
    assert result["minProperties"] == 5
```


# LLM-generated content at query #11
#--------------------------

```python
def test_from_json_schema_bool_true():
    definitions = Definitions()
    field = from_json_schema(True, definitions=definitions)
    assert isinstance(field, Any)

def test_from_json_schema_bool_false():
    definitions = Definitions()
    field = from_json_schema(False, definitions=definitions)
    assert isinstance(field, NeverMatch)

def test_from_json_schema_simple_string():
    definitions = Definitions()
    data = {"type": "string", "minLength": 5}
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.allow_blank is False

def test_from_json_schema_simple_integer():
    definitions = Definitions()
    data = {"type": "integer", "minimum": 10}
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 10

def test_from_json_schema_enum():
    definitions = Definitions()
    data = {"type": "string", "enum": ["a", "b"]}
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, Choice)
    assert ("a", "a") in field.choices

def test_from_json_schema_const():
    definitions = Definitions()
    data = {"const": 123}
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, Const)
    assert field.const == 123

def test_from_json_schema_all_of():
    definitions = Definitions()
    data = {
        "allOf": [{"type": "string"}, {"const": "foo"}]
    }
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2

def test_from_json_schema_any_of():
    definitions = Definitions()
    data = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, Union)

def test_from_json_schema_ref():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/MyType"}
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/MyType"

def test_from_json_schema_with_components():
    data = {
        "components": {
            "schemas": {
                "User": {"type": "object", "properties": {"id": {"type": "integer"}}}
            }
        },
        "$ref": "#/components/schemas/User"
    }
    definitions = Definitions()
    field = from_json_schema(data, definitions=definitions)
    assert "#/components/schemas/User" in definitions
    assert isinstance(field, Reference)

def test_from_json_schema_not():
    definitions = Definitions()
    data = {"not": {"type": "string"}}
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, Not)

def test_from_json_schema_if_then_else():
    definitions = Definitions()
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    field = from_json_schema(data, definitions=definitions)
    assert isinstance(field, IfThenElse)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem.fields import String
    field = String(min_length=5, max_length=10)
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 5
    assert result["maxLength"] == 10

def test_to_json_schema_string_nullable():
    from typesystem.fields import String
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

def test_to_json_schema_integer_with_bounds():
    from typesystem.fields import Integer
    field = Integer(minimum=0, maximum=100, exclusive_minimum=5)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5

def test_to_json_schema_boolean():
    from typesystem.fields import Boolean
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

def test_to_json_schema_array():
    from typesystem.fields import Array, Integer
    field = Array(items=Integer(), min_items=1, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["uniqueItems"] is True
    assert result["items"] == {"type": "integer"}

def test_to_json_schema_choice():
    from typesystem.fields import Choice
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "bo"] # Note: logic in provided code uses key for enum

def test_to_json_schema_union():
    from typesystem.fields import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

def test_to_json_schema_definitions_and_reference():
    from typesystem.fields import Reference, String
    class MockReference(Reference):
        def __init__(self, to, target):
            self.to = to
            self.target = target
    
    class MockSchema(Field):
        def __init__(self, properties):
            super().__init__()
            self.properties = properties

    defs = Definitions({"User": String()})
    ref_field = MockReference(to="User", target=String())
    
    # Testing the logic where arg is Definitions
    result = to_json_schema(defs)
    assert "components" in result
    assert "User" in result["components"]["schemas"]
    assert result["components"]["schemas"]["User"]["type"] == "string"

def test_to_json_schema_error_unsupported_type():
    from typesystem.fields import Field
    class UnimplementedField(Field):
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(UnimplementedField())

def test_get_standard_properties_default():
    from typesystem.fields import Integer
    field = Integer(default=10)
    result = get_standard_properties(field)
    assert result["default"] == 10

def test_to_json_schema_const():
    from typesystem.fields import Const
    field = Const(const="static_value")
    result = to_json_schema(field)
    assert result["const"] == "static_value"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_to_json_schema_evaluates_true_for_any_type():
    from typing import Any
    arg = Any
    result = to_json_schema(arg)
    assert result is True
```


# LLM-generated content at query #14
#--------------------------

```python
def test_to_json_schema_array_items_as_tuple():
    array_field = Array(items=(String(), Integer()))
    result = to_json_schema(array_field)
    assert "items" in result
    assert isinstance(result["items"], list)
```


# LLM-generated content at query #15
#--------------------------

```python
def test_to_json_schema_pattern_regex_is_not_none():
    import re
    # We need to mock/provide the necessary classes used in the function scope.
    # Assuming String, Field, etc., are available in the environment as per the snippet.
    # To trigger line 33, field.pattern_regex must be NOT None.
    # To avoid the ValueError at line 34, flags must be re.RegexFlag.UNICODE (which is default).
    
    field_with_pattern = String(pattern_regex=re.compile(r"^[a-z]+$"))
    result = to_json_schema(field_with_pattern)
    assert result["pattern"] == "^[a-z]+$"
```


# LLM-generated content at query #16
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem import String, Field
    field = String(min_length=5, max_length=10)
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 5
    assert result["maxLength"] == 10

def test_to_json_schema_string_allow_null():
    from typesystem import String
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

def test_to_json_schema_integer_with_bounds():
    from typesystem import Integer
    field = Integer(minimum=0, maximum=100, exclusive_minimum=5)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5

def test_to_json_schema_boolean_with_default():
    from typesystem import Boolean
    field = Boolean(default=True)
    result = to_json_schema(field)
    assert result["type"] == "boolean"
    assert result["default"] is True

def test_to_json_schema_array_basic():
    from typesystem import Array, String
    field = Array(items=String())
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["items"] == {"type": "string"}

def test_to_json_schema_object_properties():
    from typesystem import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string"}
    assert result["properties"]["age"] == {"type": "integer"}

def test_to_json_schema_union():
    from typesystem import Union, String, Integer
    field = Union([String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0] == {"type": "string"}
    assert result["anyOf"][1] == {"type": "integer"}

def test_to_json_schema_definitions_and_reference():
    from typesystem import Definitions, String, Reference, Object
    class UserSchema(Object):
        name = String()
    
    defs = Definitions({"User": UserSchema()})
    # Simulate a reference to the definition
    field = Reference(to="User", target=UserSchema())
    result = to_json_schema(defs)
    assert "$ref" in result["components"]["schemas"]["User"] or "User" in result["components"]["schemas"]
    # Since we are passing definitions as root, it should wrap them in components/schemas
    assert "User" in result["components"]["schemas"]

def test_to_json_schema_const():
    from typesystem import Const
    field = Const(value="fixed")
    result = to_json_schema(field)
    assert result["const"] == "fixed"

def test_to_json_schema_choice():
    from typesystem import Choice
    field = Choice([("A", None), ("B", None)])
    result = to_json_schema(field)
    assert "A" in result["enum"]
    assert "B" in result["enum"]

def test_to_json_schema_error_on_unsupported():
    from typesystem import Field
    class UnknownField(Field):
        pass
    field = UnknownField()
    try:
        to_json_schema(field)
    except ValueError as e:
        assert "Cannot convert field type" in str(e)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_one_of_from_json_schema():
    definitions = Definitions()
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "some_value"
    }
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert result.default == "some_value"

def test_one_of_from_json_schema_with_empty_list():
    definitions = Definitions()
    data = {"oneOf": []}
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 0

def test_one_of_from_json_schema_with_complex_items():
    definitions = Definitions()
    data = {
        "oneOf": [
            {"type": "integer", "enum": [1, 2]},
            {"const": "fixed"}
        ]
    }
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert isinstance(result.one_of[0], Choice)
    assert isinstance(result.one_of[1], Const)
```


# LLM-generated content at query #18
#--------------------------

```python
def test_to_json_schema_max_properties_not_none():
    field_obj = Object(properties={}, pattern_properties={}, additional_properties=True, property_names=None, max_properties=5, min_properties=None, required=[])
    result = to_json_schema(field_obj)
    assert result["maxProperties"] == 5
```


# LLM-generated content at query #19
#--------------------------

```python
def test_ref_from_json_schema_valid():
    definitions = Definitions()
    definitions["#/user"] = type("MockField", (), {"validate": lambda x: x})()
    data = {"$ref": "#/user"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/user"
    assert result.target is definitions["#/user"]

def test_ref_from_json_schema_invalid_ref_format():
    definitions = Definitions()
    data = {"$ref": "user"}
    import pytest
    with pytest.raises(AssertionError, match="Unsupported \$ref style in document."):
        ref_from_json_schema(data, definitions)

def test_ref_from_json_schema_missing_ref_key():
    definitions = Definitions()
    data = {"not_a_ref": "#/user"}
    import pytest
    with pytest.raises(KeyError):
        ref_from_json_schema(data, definitions)
```


# LLM-generated content at query #20
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem import String
    field = String()
    result = to_json_schema(field)
    assert result == {"type": "string", "minLength": 1}

def test_to_json_schema_string_with_options():
    from typesystem import String
    field = String(allow_null=True, min_length=5, max_length=10)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"], "minLength": 5, "maxLength": 10}

def test_to_json_schema_integer():
    from typesystem import Integer
    field = Integer(default=10, minimum=0)
    result = to_json_schema(field)
    assert result == {"type": "integer", "default": 10, "minimum": 0}

def test_to_json_schema_boolean():
    from typesystem import Boolean
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["boolean", "null"]}

def test_to_json_schema_array():
    from typesystem import Array, String
    field = Array(items=String(), min_items=1)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["items"] == {"type": "string", "minLength": 1}
    assert result["minItems"] == 1

def test_to_json_schema_object():
    from typesystem import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string", "minLength": 1}
    assert result["properties"]["age"] == {"type": "integer", "minLength": 1}

def test_to_json_schema_union():
    from typesystem import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

def test_to_json_schema_definitions():
    from typesystem import Definitions, String, Object
    defs = Definitions(User=Object(properties={"name": String()}))
    result = to_json_schema(defs)
    assert "components" in result
    assert "User" in result["components"]["schemas"]
    assert result["components"]["schemas"]["User"]["type"] == "object"

def test_to_json_schema_const():
    from typesystem import Const
    field = Const(value="fixed")
    result = to_json_schema(field)
    assert result["const"] == "fixed"

def test_to_json_schema_choice():
    from typesystem import Choice
    field = Choice(choices=[("a", 1), ("b", 2)])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]
```


# LLM-generated content at query #21
#--------------------------

```python
def test_to_json_schema_evaluates_schema_branch():
    schema_field = Schema(allow_null=False, fields={}, required=[])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
```


# LLM-generated content at query #22
#--------------------------

```python
import re
from typesystem.fields import (
    String, Integer, Float, Boolean, Array, Object, 
    Union, Choice, Const, Any, NeverMatch
)
from typesystem.schemas import Definitions
from typesystem.json_schema import to_json_schema

def test_to_json_schema_any_and_nevermatch():
    assert to_json_schema(Any()) is True
    assert to_json_schema(NeverMatch()) is False

def test_to_json_schema_string_basic():
    field = String()
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 1

def test_to_json_schema_string_with_constraints():
    field = String(min_length=5, max_length=10, allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 5
    assert result["maxLength"] == 10

def test_to_json_schema_integer():
    field = Integer(minimum=0, maximum=100, default=10)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["default"] == 10

def test_to_json_schema_float():
    field = Float(exclusive_minimum=0.5)
    result = to_json_schema(field)
    assert result["type"] == "number"
    assert result["exclusiveMinimum"] == 0.5

def test_to_json_schema_boolean():
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

def test_to_json_schema_array():
    items_field = String()
    field = Array(items=items_field, min_items=1, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["uniqueItems"] is True
    assert result["items"] == {"type": "string", "minLength": 1}

def test_to_json_schema_object():
    properties = {"name": String(), "age": Integer()}
    field = Object(properties=properties, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string", "minLength": 1}
    assert result["properties"]["age"] == {"type": "integer", "minLength": 1}
    assert result["required"] == ["name"]

def test_to_json_schema_choice():
    field = Choice(choices=[("a", 1), ("b", 2)])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]

def test_to_json_schema_const():
    field = Const(const="fixed")
    result = to_json_schema(field)
    assert result["const"] == "fixed"

def test_to_json_schema_union():
    field = Union([String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

def test_to_json_schema_definitions():
    defs = Definitions({"User": Object(properties={"id": Integer()})})
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"]["User"]["type"] == "object"
    assert result["components"]["schemas"]["User"]["properties"]["id"]["type"] == "integer"

def test_to_json_schema_regex_error():
    import re
    # Using a non-unicode flag to trigger the error in the provided code logic
    field = String(pattern_regex=re.compile(r'abc', re.ASCII))
    try:
        to_json_schema(field)
    except ValueError as e:
        assert "Cannot convert regular expression with non-standard flags" in str(e)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_to_json_schema_string_basic():
    from typesystem import String
    field = String(min_length=5, max_length=10)
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 5
    assert result["maxLength"] == 10

def test_to_json_schema_string_nullable():
    from typesystem import String
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

def test_to_json_schema_integer_with_bounds():
    from typesystem import Integer
    field = Integer(minimum=0, maximum=100, exclusive_minimum=5)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5

def test_to_json_schema_boolean():
    from typesystem import Boolean
    field = Boolean(allow_null=True, default=False)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]
    assert result["default"] is False

def test_to_json_schema_array():
    from typesystem import Array, String, Integer
    field = Array(items=String(), min_items=1, max_items=5)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["items"] == {"type": "string"}

def test_to_json_schema_object():
    from typesystem import Object, String, Integer
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string"}
    assert result["properties"]["age"] == {"type": "integer"}
    assert result["required"] == ["name"]

def test_to_json_schema_union():
    from typesystem import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0] == {"type": "string"}
    assert result["anyOf"][1] == {"type": "integer"}

def test_to_json_schema_definitions_and_reference():
    from typesystem import Definitions, Reference, String
    class MockRefTarget(Field):
        pass
    
    target = String()
    defs = Definitions({"User": target})
    ref = Reference(to="User", target=target)
    result = to_json_schema(defs, _definitions={})
    # Note: The implementation of to_json_schema with Definitions iterates keys.
    # We check the structure created via the recursive call logic in the provided snippet.
    assert "User" in result["components"]["schemas"]
    assert result["components"]["schemas"]["User"] == {"type": "string"}

def test_to_json_schema_const():
    from typesystem import Const
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

def test_to_json_schema_choice():
    from typesystem import Choice
    field = Choice(choices=[("A", None), ("B", None)])
    result = to_json_schema(field)
    assert "A" in result["enum"]
    assert "B" in result["enum"]

def test_to_json_schema_boolean_return():
    # Testing the 'isinstance(arg, Any)' path if it exists or similar early exits
    # Since 'Any' is not a standard type used for logic here but part of typing, 
    # we assume testing valid types.
    assert to_json_schema(None) == {} # Based on code: if field is None, returns empty dict/logic falls through
```


