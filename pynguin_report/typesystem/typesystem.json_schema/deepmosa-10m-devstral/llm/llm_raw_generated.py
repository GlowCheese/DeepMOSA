####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_from_json_schema_with_bool_true():
    result = from_json_schema(True)
    assert isinstance(result, Any)

def test_from_json_schema_with_bool_false():
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

def test_from_json_schema_with_ref():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/test"}
    definitions["#/components/schemas/test"] = String()
    result = from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/test"

def test_from_json_schema_with_type_constraint():
    data = {"type": "string"}
    result = from_json_schema(data)
    assert isinstance(result, String)

def test_from_json_schema_with_enum():
    data = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]

def test_from_json_schema_with_const():
    data = {"const": "test"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == "test"

def test_from_json_schema_with_all_of():
    data = {"allOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_from_json_schema_with_any_of():
    data = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

def test_from_json_schema_with_one_of():
    data = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2

def test_from_json_schema_with_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)
    assert isinstance(result.negated, String)

def test_from_json_schema_with_if_then_else():
    data = {"if": {"type": "string"}, "then": {"type": "number"}, "else": {"type": "boolean"}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, Number)
    assert isinstance(result.else_clause, Boolean)

def test_from_json_schema_with_multiple_constraints():
    data = {"type": "string", "enum": ["a", "b"]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], String)
    assert isinstance(result.all_of[1], Choice)

def test_from_json_schema_with_no_constraints():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)


# LLM-generated content at query #2
#--------------------------

```python
def test_to_json_schema_with_any_field():
    assert to_json_schema(Any()) == True

def test_to_json_schema_with_never_match_field():
    assert to_json_schema(NeverMatch()) == False

def test_to_json_schema_with_string_field():
    field = String()
    assert to_json_schema(field) == {"type": "string"}

def test_to_json_schema_with_string_field_allow_null():
    field = String(allow_null=True)
    assert to_json_schema(field) == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_min_length():
    field = String(min_length=5)
    assert to_json_schema(field) == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_max_length():
    field = String(max_length=10)
    assert to_json_schema(field) == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    assert to_json_schema(field) == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_format():
    field = String(format="email")
    assert to_json_schema(field) == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    assert to_json_schema(field) == {"type": "integer"}

def test_to_json_schema_with_integer_field_allow_null():
    field = Integer(allow_null=True)
    assert to_json_schema(field) == {"type": ["integer", "null"]}

def test_to_json_schema_with_integer_field_minimum():
    field = Integer(minimum=0)
    assert to_json_schema(field) == {"type": "integer", "minimum": 0}

def test_to_json_schema_with_integer_field_maximum():
    field = Integer(maximum=100)
    assert to_json_schema(field) == {"type": "integer", "maximum": 100}

def test_to_json_schema_with_float_field():
    field = Float()
    assert to_json_schema(field) == {"type": "number"}

def test_to_json_schema_with_float_field_allow_null():
    field = Float(allow_null=True)
    assert to_json_schema(field) == {"type": ["number", "null"]}

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    assert to_json_schema(field) == {"type": "number"}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    assert to_json_schema(field) == {"type": "boolean"}

def test_to_json_schema_with_boolean_field_allow_null():
    field = Boolean(allow_null=True)
    assert to_json_schema(field) == {"type": ["boolean", "null"]}

def test_to_json_schema_with_array_field():
    field = Array()
    assert to_json_schema(field) == {"type": "array"}

def test_to_json_schema_with_array_field_allow_null():
    field = Array(allow_null=True)
    assert to_json_schema(field) == {"type": ["array", "null"]}

def test_to_json_schema_with_array_field_min_items():
    field = Array(min_items=1)
    assert to_json_schema(field) == {"type": "array", "minItems": 1}

def test_to_json_schema_with_array_field_max_items():
    field = Array(max_items=10)
    assert to_json_schema(field) == {"type": "array", "maxItems": 10}

def test_to_json_schema_with_array_field_items():
    field = Array(items=String())
    assert to_json_schema(field) == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_array_field_additional_items():
    field = Array(additional_items=Boolean())
    assert to_json_schema(field) == {"type": "array", "additionalItems": {"type": "boolean"}}

def test_to_json_schema_with_array_field_unique_items():
    field = Array(unique_items=True)
    assert to_json_schema(field) == {"type": "array", "uniqueItems": True}

def test_to_json_schema_with_object_field():
    field = Object()
    assert to_json_schema(field) == {"type": "object"}

def test_to_json_schema_with_object_field_allow_null():
    field = Object(allow_null=True)
    assert to_json_schema(field) == {"type": ["object", "null"]}

def test_to_json_schema_with_object_field_properties():
    field = Object(properties={"name": String()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_object_field_pattern_properties():
    field = Object(pattern_properties={r"^[a-z]+$": Integer()})
    assert to_json_schema(field) == {"type": "object", "patternProperties": {r"^[a-z]+$": {"type": "integer"}}}

def test_to_json_schema_with_object_field_additional_properties():
    field = Object(additional_properties=Boolean())
    assert to_json_schema(field) == {"type": "object", "additionalProperties": {"type": "boolean"}}

def test_to_json_schema_with_object_field_property_names():
    field = Object(property_names=String())
    assert to_json_schema(field) == {"type": "object", "propertyNames": {"type": "string"}}

def test_to_json_schema_with_object_field_max_properties():
    field = Object(max_properties=5)
    assert to_json_schema(field) == {"type": "object", "maxProperties": 5}

def test_to_json_schema_with_object_field_min_properties():
    field = Object(min_properties=1)
    assert to_json_schema(field) == {"type": "object", "minProperties": 1}

def test_to_json_schema_with_object_field_required():
    field = Object(required=["name"])
    assert to_json_schema(field) == {"type": "object", "required": ["name"]}

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_schema_field_allow_null():
    field = Schema(allow_null=True, fields={"name": String()})
    assert to_json_schema(field) == {"type": ["object", "null"], "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_schema_field_required():
    field = Schema(fields={"name": String()}, required=["name"])
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", 1), ("b", 2)])
    assert to_json_schema(field) == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="value")
    assert to_json_schema(field) == {"const": "value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    assert to_json_schema(field) == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    assert to_json_schema(field) == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_all_of_field():
    field = AllOf(all_of=[String(), Integer()])
    assert to_json_schema(field) == {"allOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_if_then_else_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    assert to_json_schema(field) == {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    assert to_json_schema(field) == {"not": {"type": "string"}}

def test_to_json_schema_with_definitions():
    definitions = Definitions({"string": String(), "integer": Integer()})
    assert to_json_schema(definitions) == {"components": {"schemas": {"string": {"type": "string"}, "integer": {"type": "integer"}}}}


# LLM-generated content at query #3
#--------------------------

```python
def test_ifthenelse_field_type_check():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=None)
    assert isinstance(field, IfThenElse)


# LLM-generated content at query #4
#--------------------------

```python
def test_array_field_with_list_items():
    field = Array(items=[String(), Integer()])
    result = to_json_schema(field)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2


# LLM-generated content at query #5
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"type": "number", "minimum": 0, "maximum": 100, "default": 50}
    result = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.default == 50
    assert result.allow_null is False

def test_from_json_schema_type_integer():
    data = {"type": "integer", "minimum": 0, "maximum": 100, "default": 50}
    result = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.default == 50
    assert result.allow_null is False

def test_from_json_schema_type_string():
    data = {"type": "string", "minLength": 5, "maxLength": 10, "format": "email", "default": "test@example.com"}
    result = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.format == "email"
    assert result.default == "test@example.com"
    assert result.allow_null is False

def test_from_json_schema_type_boolean():
    data = {"type": "boolean", "default": True}
    result = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(result, Boolean)
    assert result.default is True
    assert result.allow_null is False

def test_from_json_schema_type_array():
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5, "default": ["test"]}
    result = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    assert result.min_items == 1
    assert result.max_items == 5
    assert result.default == ["test"]
    assert result.allow_null is False

def test_from_json_schema_type_object():
    data = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "default": {"name": "test"}}
    result = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(result, Object)
    assert isinstance(result.properties["name"], String)
    assert result.required == ["name"]
    assert result.default == {"name": "test"}
    assert result.allow_null is False


# LLM-generated content at query #6
#--------------------------

```python
def test_type_from_json_schema_with_single_type():
    data = {"type": "string"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    assert result.allow_null is False

def test_type_from_json_schema_with_nullable_single_type():
    data = {"type": ["string", "null"]}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    assert result.allow_null is True

def test_type_from_json_schema_with_multiple_types():
    data = {"type": ["string", "number"]}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Float)
    assert result.allow_null is False

def test_type_from_json_schema_with_nullable_multiple_types():
    data = {"type": ["string", "number", "null"]}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Float)
    assert result.allow_null is True

def test_type_from_json_schema_with_no_type():
    data = {}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Any)

def test_type_from_json_schema_with_nullable_no_type():
    data = {"type": ["null"]}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)
    assert result.const is None

def test_type_from_json_schema_with_integer_type():
    data = {"type": "integer"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)
    assert result.allow_null is False

def test_type_from_json_schema_with_boolean_type():
    data = {"type": "boolean"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Boolean)
    assert result.allow_null is False

def test_type_from_json_schema_with_array_type():
    data = {"type": "array"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Array)
    assert result.allow_null is False

def test_type_from_json_schema_with_object_type():
    data = {"type": "object"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Object)
    assert result.allow_null is False

def test_type_from_json_schema_with_number_type():
    data = {"type": "number"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Float)
    assert result.allow_null is False


# LLM-generated content at query #7
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"type": "number", "minimum": 0, "maximum": 100, "default": 50}
    result = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.default == 50
    assert result.allow_null is False
    assert result.coerce_types is False

def test_from_json_schema_type_integer():
    data = {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}
    result = from_json_schema_type(data, "integer", True, Definitions())
    assert isinstance(result, Integer)
    assert result.minimum == 1
    assert result.maximum == 10
    assert result.default == 5
    assert result.allow_null is True
    assert result.coerce_types is False

def test_from_json_schema_type_string():
    data = {"type": "string", "minLength": 5, "maxLength": 10, "format": "email", "default": "test@example.com"}
    result = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.format == "email"
    assert result.default == "test@example.com"
    assert result.allow_null is False
    assert result.allow_blank is False
    assert result.coerce_types is False

def test_from_json_schema_type_string_with_min_length_zero():
    data = {"type": "string", "minLength": 0, "default": ""}
    result = from_json_schema_type(data, "string", True, Definitions())
    assert isinstance(result, String)
    assert result.min_length is None
    assert result.default == ""
    assert result.allow_null is True
    assert result.allow_blank is True
    assert result.coerce_types is False

def test_from_json_schema_type_boolean():
    data = {"type": "boolean", "default": True}
    result = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(result, Boolean)
    assert result.default is True
    assert result.allow_null is False
    assert result.coerce_types is False

def test_from_json_schema_type_array():
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5, "default": ["a", "b"]}
    result = from_json_schema_type(data, "array", True, Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    assert result.min_items == 1
    assert result.max_items == 5
    assert result.default == ["a", "b"]
    assert result.allow_null is True
    assert result.additional_items is True
    assert result.unique_items is False

def test_from_json_schema_type_array_with_list_items():
    data = {"type": "array", "items": [{"type": "string"}, {"type": "number"}], "default": ["a", 1]}
    result = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, list)
    assert len(result.items) == 2
    assert isinstance(result.items[0], String)
    assert isinstance(result.items[1], Float)
    assert result.default == ["a", 1]
    assert result.allow_null is False
    assert result.min_items == 2
    assert result.max_items == 2

def test_from_json_schema_type_object():
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    result = from_json_schema_type(data, "object", True, Definitions())
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert "age" in result.properties
    assert isinstance(result.properties["name"], String)
    assert isinstance(result.properties["age"], Integer)
    assert result.required == ["name"]
    assert result.default == {"name": "John", "age": 30}
    assert result.allow_null is True


# LLM-generated content at query #8
#--------------------------

```python
def test_to_json_schema_with_any_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import Any
    result = to_json_schema(Any())
    assert result is True

def test_to_json_schema_with_never_match_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import NeverMatch
    result = to_json_schema(NeverMatch())
    assert result is False

def test_to_json_schema_with_string_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import String
    field = String(min_length=1, max_length=10, allow_blank=False)
    result = to_json_schema(field)
    assert result == {
        "type": "string",
        "minLength": 1,
        "maxLength": 10
    }

def test_to_json_schema_with_string_field_allow_null():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import String
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"]}

def test_to_json_schema_with_integer_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import Integer
    field = Integer(minimum=0, maximum=100)
    result = to_json_schema(field)
    assert result == {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }

def test_to_json_schema_with_float_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import Float
    field = Float(multiple_of=0.5)
    result = to_json_schema(field)
    assert result == {
        "type": "number",
        "multipleOf": 0.5
    }

def test_to_json_schema_with_boolean_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import Boolean
    field = Boolean()
    result = to_json_schema(field)
    assert result == {"type": "boolean"}

def test_to_json_schema_with_array_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import Array, String
    field = Array(items=String(), min_items=1, max_items=10)
    result = to_json_schema(field)
    assert result == {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10
    }

def test_to_json_schema_with_object_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import Object, String
    field = Object(properties={"name": String()})
    result = to_json_schema(field)
    assert result == {
        "type": "object",
        "properties": {"name": {"type": "string"}}
    }

def test_to_json_schema_with_choice_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import Choice
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(field)
    assert result == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import Const
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result == {"const": "fixed_value"}

def test_to_json_schema_with_union_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import Union, String, Integer
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }

def test_to_json_schema_with_definitions():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import String
    from typesystem.schemas import Definitions
    definitions = Definitions({"string_field": String()})
    result = to_json_schema(definitions)
    assert result == {
        "components": {
            "schemas": {
                "string_field": {"type": "string"}
            }
        }
    }

def test_to_json_schema_with_reference_field():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import Reference, String
    target = String()
    field = Reference(to="string_ref", target=target)
    result = to_json_schema(field)
    assert result == {
        "$ref": "#/components/schemas/string_ref",
        "components": {
            "schemas": {
                "string_ref": {"type": "string"}
            }
        }
    }

def test_to_json_schema_with_default_value():
    from typesystem.json_schema import to_json_schema
    from typesystem.fields import String
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result == {
        "type": "string",
        "default": "default_value"
    }


# LLM-generated content at query #9
#--------------------------

```python
def test_to_json_schema_with_any_field():
    field = Any()
    result = to_json_schema(field)
    assert result is True

def test_to_json_schema_with_never_match_field():
    field = NeverMatch()
    result = to_json_schema(field)
    assert result is False

def test_to_json_schema_with_string_field():
    field = String()
    result = to_json_schema(field)
    assert result == {"type": "string"}

def test_to_json_schema_with_string_field_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_min_length():
    field = String(min_length=5)
    result = to_json_schema(field)
    assert result == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_max_length():
    field = String(max_length=10)
    result = to_json_schema(field)
    assert result == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    result = to_json_schema(field)
    assert result == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_format():
    field = String(format="email")
    result = to_json_schema(field)
    assert result == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    result = to_json_schema(field)
    assert result == {"type": "integer"}

def test_to_json_schema_with_integer_field_allow_null():
    field = Integer(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["integer", "null"]}

def test_to_json_schema_with_float_field():
    field = Float()
    result = to_json_schema(field)
    assert result == {"type": "number"}

def test_to_json_schema_with_float_field_allow_null():
    field = Float(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["number", "null"]}

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    result = to_json_schema(field)
    assert result == {"type": "number"}

def test_to_json_schema_with_decimal_field_allow_null():
    field = Decimal(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["number", "null"]}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    assert result == {"type": "boolean"}

def test_to_json_schema_with_boolean_field_allow_null():
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["boolean", "null"]}

def test_to_json_schema_with_array_field():
    field = Array()
    result = to_json_schema(field)
    assert result == {"type": "array"}

def test_to_json_schema_with_array_field_allow_null():
    field = Array(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["array", "null"]}

def test_to_json_schema_with_array_field_items():
    field = Array(items=String())
    result = to_json_schema(field)
    assert result == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_array_field_additional_items():
    field = Array(additional_items=Integer())
    result = to_json_schema(field)
    assert result == {"type": "array", "additionalItems": {"type": "integer"}}

def test_to_json_schema_with_array_field_unique_items():
    field = Array(unique_items=True)
    result = to_json_schema(field)
    assert result == {"type": "array", "uniqueItems": True}

def test_to_json_schema_with_object_field():
    field = Object()
    result = to_json_schema(field)
    assert result == {"type": "object"}

def test_to_json_schema_with_object_field_allow_null():
    field = Object(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["object", "null"]}

def test_to_json_schema_with_object_field_properties():
    field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_object_field_required():
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    assert result == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_schema_field_allow_null():
    field = Schema(fields={"name": String(), "age": Integer()}, allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["object", "null"], "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(field)
    assert result == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="constant_value")
    result = to_json_schema(field)
    assert result == {"const": "constant_value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_all_of_field():
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"allOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_if_then_else_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(field)
    assert result == {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    result = to_json_schema(field)
    assert result == {"not": {"type": "string"}}

def test_to_json_schema_with_definitions():
    definitions = Definitions({"string_field": String(), "integer_field": Integer()})
    result = to_json_schema(definitions)
    assert result == {"components": {"schemas": {"string_field": {"type": "string"}, "integer_field": {"type": "integer"}}}}

def test_to_json_schema_with_reference_field():
    field = Reference(to="string_field", target=String())
    result = to_json_schema(field)
    assert result == {"$ref": "#/components/schemas/string_field", "components": {"schemas": {"string_field": {"type": "string"}}}}


# LLM-generated content at query #10
#--------------------------

```python
def test_to_json_schema_with_any_field():
    schema = to_json_schema(Any())
    assert schema is True

def test_to_json_schema_with_never_match_field():
    schema = to_json_schema(NeverMatch())
    assert schema is False

def test_to_json_schema_with_string_field():
    field = String()
    schema = to_json_schema(field)
    assert schema == {"type": "string"}

def test_to_json_schema_with_nullable_string_field():
    field = String(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_with_default():
    field = String(default="default_value")
    schema = to_json_schema(field)
    assert schema == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_string_field_with_min_length():
    field = String(min_length=5)
    schema = to_json_schema(field)
    assert schema == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_with_max_length():
    field = String(max_length=10)
    schema = to_json_schema(field)
    assert schema == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_with_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    schema = to_json_schema(field)
    assert schema == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_with_format():
    field = String(format="email")
    schema = to_json_schema(field)
    assert schema == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    schema = to_json_schema(field)
    assert schema == {"type": "integer"}

def test_to_json_schema_with_nullable_integer_field():
    field = Integer(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["integer", "null"]}

def test_to_json_schema_with_integer_field_with_minimum():
    field = Integer(minimum=0)
    schema = to_json_schema(field)
    assert schema == {"type": "integer", "minimum": 0}

def test_to_json_schema_with_integer_field_with_maximum():
    field = Integer(maximum=100)
    schema = to_json_schema(field)
    assert schema == {"type": "integer", "maximum": 100}

def test_to_json_schema_with_float_field():
    field = Float()
    schema = to_json_schema(field)
    assert schema == {"type": "number"}

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    schema = to_json_schema(field)
    assert schema == {"type": "number"}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    schema = to_json_schema(field)
    assert schema == {"type": "boolean"}

def test_to_json_schema_with_nullable_boolean_field():
    field = Boolean(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["boolean", "null"]}

def test_to_json_schema_with_array_field():
    field = Array()
    schema = to_json_schema(field)
    assert schema == {"type": "array"}

def test_to_json_schema_with_nullable_array_field():
    field = Array(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["array", "null"]}

def test_to_json_schema_with_array_field_with_min_items():
    field = Array(min_items=1)
    schema = to_json_schema(field)
    assert schema == {"type": "array", "minItems": 1}

def test_to_json_schema_with_array_field_with_max_items():
    field = Array(max_items=10)
    schema = to_json_schema(field)
    assert schema == {"type": "array", "maxItems": 10}

def test_to_json_schema_with_array_field_with_items():
    field = Array(items=String())
    schema = to_json_schema(field)
    assert schema == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_array_field_with_tuple_items():
    field = Array(items=[String(), Integer()])
    schema = to_json_schema(field)
    assert schema == {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_array_field_with_additional_items():
    field = Array(additional_items=Boolean())
    schema = to_json_schema(field)
    assert schema == {"type": "array", "additionalItems": {"type": "boolean"}}

def test_to_json_schema_with_array_field_with_unique_items():
    field = Array(unique_items=True)
    schema = to_json_schema(field)
    assert schema == {"type": "array", "uniqueItems": True}

def test_to_json_schema_with_object_field():
    field = Object()
    schema = to_json_schema(field)
    assert schema == {"type": "object"}

def test_to_json_schema_with_nullable_object_field():
    field = Object(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["object", "null"]}

def test_to_json_schema_with_object_field_with_properties():
    field = Object(properties={"name": String(), "age": Integer()})
    schema = to_json_schema(field)
    assert schema == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_object_field_with_pattern_properties():
    field = Object(pattern_properties={"^S_": String(), "^I_": Integer()})
    schema = to_json_schema(field)
    assert schema == {"type": "object", "patternProperties": {"^S_": {"type": "string"}, "^I_": {"type": "integer"}}}

def test_to_json_schema_with_object_field_with_additional_properties():
    field = Object(additional_properties=Boolean())
    schema = to_json_schema(field)
    assert schema == {"type": "object", "additionalProperties": {"type": "boolean"}}

def test_to_json_schema_with_object_field_with_property_names():
    field = Object(property_names=String())
    schema = to_json_schema(field)
    assert schema == {"type": "object", "propertyNames": {"type": "string"}}

def test_to_json_schema_with_object_field_with_max_properties():
    field = Object(max_properties=10)
    schema = to_json_schema(field)
    assert schema == {"type": "object", "maxProperties": 10}

def test_to_json_schema_with_object_field_with_min_properties():
    field = Object(min_properties=1)
    schema = to_json_schema(field)
    assert schema == {"type": "object", "minProperties": 1}

def test_to_json_schema_with_object_field_with_required():
    field = Object(required=["name"])
    schema = to_json_schema(field)
    assert schema == {"type": "object", "required": ["name"]}

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String(), "age": Integer()})
    schema = to_json_schema(field)
    assert schema == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_schema_field_with_required():
    field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    schema = to_json_schema(field)
    assert schema == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    schema = to_json_schema(field)
    assert schema == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="constant_value")
    schema = to_json_schema(field)
    assert schema == {"const": "constant_value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert schema == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert schema == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_all_of_field():
    field = AllOf(all_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert schema == {"allOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_if_then_else_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    schema = to_json_schema(field)
    assert schema == {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    schema = to_json_schema(field)
    assert schema == {"not": {"type": "string"}}

def test_to_json_schema_with_definitions():
    definitions =


# LLM-generated content at query #11
#--------------------------

```python
def test_string_field_with_pattern_regex():
    field = String(pattern_regex=re.compile(r"^[a-z]+$", re.RegexFlag.UNICODE))
    result = to_json_schema(field)
    assert "pattern" in result


# LLM-generated content at query #12
#--------------------------

```python
def test_from_json_schema_type_array_additional_items_none():
    data = {"type": "array"}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert result.additional_items is True


# LLM-generated content at query #13
#--------------------------

```python
def test_array_field_with_none_additional_items():
    field = Array(additional_items=None)
    result = to_json_schema(field)
    assert "additionalItems" not in result


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_160_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=None)
    assert not field.else_clause


# LLM-generated content at query #15
#--------------------------

```python
def test_pattern_regex_is_not_none():
    field = String(pattern_regex=re.compile("test"))
    assert field.pattern_regex is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_from_json_schema_type_object_additional_properties_none():
    data = {"type": "object"}
    result = from_json_schema_type(data, "object", False, {})
    assert result.additional_properties is None


# LLM-generated content at query #17
#--------------------------

```python
def test_from_json_schema_type_object_additional_properties_none():
    data = {"type": "object"}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert result.additional_properties is None


# LLM-generated content at query #18
#--------------------------

```python
def test_to_json_schema_with_any_field():
    result = to_json_schema(Any())
    assert result is True

def test_to_json_schema_with_never_match_field():
    result = to_json_schema(NeverMatch())
    assert result is False

def test_to_json_schema_with_string_field():
    field = String()
    result = to_json_schema(field)
    assert result == {"type": "string"}

def test_to_json_schema_with_string_field_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_min_length():
    field = String(min_length=5)
    result = to_json_schema(field)
    assert result == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_max_length():
    field = String(max_length=10)
    result = to_json_schema(field)
    assert result == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    result = to_json_schema(field)
    assert result == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_format():
    field = String(format="email")
    result = to_json_schema(field)
    assert result == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    result = to_json_schema(field)
    assert result == {"type": "integer"}

def test_to_json_schema_with_float_field():
    field = Float()
    result = to_json_schema(field)
    assert result == {"type": "number"}

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    result = to_json_schema(field)
    assert result == {"type": "number"}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    assert result == {"type": "boolean"}

def test_to_json_schema_with_array_field():
    field = Array()
    result = to_json_schema(field)
    assert result == {"type": "array"}

def test_to_json_schema_with_object_field():
    field = Object()
    result = to_json_schema(field)
    assert result == {"type": "object"}

def test_to_json_schema_with_schema_field():
    field = Schema()
    result = to_json_schema(field)
    assert result == {"type": "object"}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", 1), ("b", 2)])
    result = to_json_schema(field)
    assert result == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="constant")
    result = to_json_schema(field)
    assert result == {"const": "constant"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_all_of_field():
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"allOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_if_then_else_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(field)
    assert result == {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    result = to_json_schema(field)
    assert result == {"not": {"type": "string"}}

def test_to_json_schema_with_definitions():
    definitions = Definitions()
    definitions["string_field"] = String()
    result = to_json_schema(definitions)
    assert result == {"components": {"schemas": {"string_field": {"type": "string"}}}}

def test_to_json_schema_with_reference_field():
    target = String()
    field = Reference(to="string_ref", target=target)
    result = to_json_schema(field)
    assert result == {"$ref": "#/components/schemas/string_ref", "components": {"schemas": {"string_ref": {"type": "string"}}}}


# LLM-generated content at query #19
#--------------------------

```python
def test_exclusive_minimum_predicate():
    field = Integer(exclusive_minimum=5)
    result = to_json_schema(field)
    assert "exclusiveMinimum" in result
    assert result["exclusiveMinimum"] is True


# LLM-generated content at query #20
#--------------------------

```python
def test_to_json_schema_with_any_field():
    field = Any()
    result = to_json_schema(field)
    assert result is True

def test_to_json_schema_with_never_match_field():
    field = NeverMatch()
    result = to_json_schema(field)
    assert result is False

def test_to_json_schema_with_string_field():
    field = String()
    result = to_json_schema(field)
    assert result == {"type": "string"}

def test_to_json_schema_with_string_field_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_with_default():
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_string_field_with_min_length():
    field = String(min_length=5)
    result = to_json_schema(field)
    assert result == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_with_max_length():
    field = String(max_length=10)
    result = to_json_schema(field)
    assert result == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_with_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    result = to_json_schema(field)
    assert result == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_with_format():
    field = String(format="email")
    result = to_json_schema(field)
    assert result == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    result = to_json_schema(field)
    assert result == {"type": "integer"}

def test_to_json_schema_with_integer_field_allow_null():
    field = Integer(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["integer", "null"]}

def test_to_json_schema_with_integer_field_with_minimum():
    field = Integer(minimum=0)
    result = to_json_schema(field)
    assert result == {"type": "integer", "minimum": 0}

def test_to_json_schema_with_integer_field_with_maximum():
    field = Integer(maximum=100)
    result = to_json_schema(field)
    assert result == {"type": "integer", "maximum": 100}

def test_to_json_schema_with_integer_field_with_exclusive_minimum():
    field = Integer(exclusive_minimum=0)
    result = to_json_schema(field)
    assert result == {"type": "integer", "exclusiveMinimum": 0}

def test_to_json_schema_with_integer_field_with_exclusive_maximum():
    field = Integer(exclusive_maximum=100)
    result = to_json_schema(field)
    assert result == {"type": "integer", "exclusiveMaximum": 100}

def test_to_json_schema_with_integer_field_with_multiple_of():
    field = Integer(multiple_of=2)
    result = to_json_schema(field)
    assert result == {"type": "integer", "multipleOf": 2}

def test_to_json_schema_with_float_field():
    field = Float()
    result = to_json_schema(field)
    assert result == {"type": "number"}

def test_to_json_schema_with_float_field_allow_null():
    field = Float(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["number", "null"]}

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    result = to_json_schema(field)
    assert result == {"type": "number"}

def test_to_json_schema_with_decimal_field_allow_null():
    field = Decimal(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["number", "null"]}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    assert result == {"type": "boolean"}

def test_to_json_schema_with_boolean_field_allow_null():
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["boolean", "null"]}

def test_to_json_schema_with_array_field():
    field = Array()
    result = to_json_schema(field)
    assert result == {"type": "array"}

def test_to_json_schema_with_array_field_allow_null():
    field = Array(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["array", "null"]}

def test_to_json_schema_with_array_field_with_min_items():
    field = Array(min_items=1)
    result = to_json_schema(field)
    assert result == {"type": "array", "minItems": 1}

def test_to_json_schema_with_array_field_with_max_items():
    field = Array(max_items=10)
    result = to_json_schema(field)
    assert result == {"type": "array", "maxItems": 10}

def test_to_json_schema_with_array_field_with_items():
    field = Array(items=String())
    result = to_json_schema(field)
    assert result == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_array_field_with_additional_items():
    field = Array(additional_items=Boolean())
    result = to_json_schema(field)
    assert result == {"type": "array", "additionalItems": {"type": "boolean"}}

def test_to_json_schema_with_array_field_with_unique_items():
    field = Array(unique_items=True)
    result = to_json_schema(field)
    assert result == {"type": "array", "uniqueItems": True}

def test_to_json_schema_with_object_field():
    field = Object()
    result = to_json_schema(field)
    assert result == {"type": "object"}

def test_to_json_schema_with_object_field_allow_null():
    field = Object(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["object", "null"]}

def test_to_json_schema_with_object_field_with_properties():
    field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_object_field_with_pattern_properties():
    field = Object(pattern_properties={"^S_": String(), "^I_": Integer()})
    result = to_json_schema(field)
    assert result == {"type": "object", "patternProperties": {"^S_": {"type": "string"}, "^I_": {"type": "integer"}}}

def test_to_json_schema_with_object_field_with_additional_properties():
    field = Object(additional_properties=Boolean())
    result = to_json_schema(field)
    assert result == {"type": "object", "additionalProperties": {"type": "boolean"}}

def test_to_json_schema_with_object_field_with_property_names():
    field = Object(property_names=String())
    result = to_json_schema(field)
    assert result == {"type": "object", "propertyNames": {"type": "string"}}

def test_to_json_schema_with_object_field_with_max_properties():
    field = Object(max_properties=10)
    result = to_json_schema(field)
    assert result == {"type": "object", "maxProperties": 10}

def test_to_json_schema_with_object_field_with_min_properties():
    field = Object(min_properties=1)
    result = to_json_schema(field)
    assert result == {"type": "object", "minProperties": 1}

def test_to_json_schema_with_object_field_with_required():
    field = Object(required=["name"])
    result = to_json_schema(field)
    assert result == {"type": "object", "required": ["name"]}

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_schema_field_allow_null():
    field = Schema(allow_null=True, fields={"name": String()})
    result = to_json_schema(field)
    assert result == {"type": ["object", "null"], "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_schema_field_with_required():
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert result == {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(field)
    assert result == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="constant_value")
    result = to_json_schema(field)
    assert result == {"const": "constant_value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"anyOf": [{"type": "string"},


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_158_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=Integer())
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #22
#--------------------------

```python
def test_from_json_schema_type_pattern_properties_none():
    result = from_json_schema_type(
        data={"type": "object"},
        type_string="object",
        allow_null=False,
        definitions={}
    )
    assert result.pattern_properties is None


# LLM-generated content at query #23
#--------------------------

```python
def test_from_json_schema_type_pattern_properties_none():
    data = {"type": "object"}
    type_string = "object"
    allow_null = False
    definitions = Definitions()
    result = from_json_schema_type(data, type_string, allow_null, definitions)
    assert result.pattern_properties is None


# LLM-generated content at query #24
#--------------------------

```python
def test_from_json_schema_type_property_names_none():
    data = {}
    type_string = "object"
    allow_null = False
    definitions = {}
    result = from_json_schema_type(data, type_string, allow_null, definitions)
    assert result.property_names is None


# LLM-generated content at query #25
#--------------------------

```python
def test_from_json_schema_type_property_names_none():
    data = {}
    type_string = "object"
    allow_null = False
    definitions = {}
    result = from_json_schema_type(data, type_string, allow_null, definitions)
    assert result.property_names is None


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_at_line_158_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=Integer())
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #27
#--------------------------

```python
def test_additional_items_predicate_false():
    field = Array(items=String(), additional_items=None)
    result = to_json_schema(field)
    assert "additionalItems" not in result


# LLM-generated content at query #28
#--------------------------

```python
def test_pattern_regex_is_not_none():
    field = String(pattern_regex=re.compile("test"))
    assert field.pattern_regex is not None


# LLM-generated content at query #29
#--------------------------

```python
def test_from_json_schema_type_object_with_additional_properties_none():
    data = {"type": "object", "additionalProperties": None}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert result.additional_properties is None


# LLM-generated content at query #30
#--------------------------

```python
def test_from_json_schema_type_additional_properties_none():
    data = {"type": "object"}
    type_string = "object"
    allow_null = False
    definitions = {}
    result = from_json_schema_type(data, type_string, allow_null, definitions)
    assert result.additional_properties is None


# LLM-generated content at query #31
#--------------------------

```python
def test_if_then_else_without_then_clause():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=None)
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #32
#--------------------------

```python
def test_from_json_schema_type_array_additional_items_none():
    data = {"type": "array"}
    type_string = "array"
    allow_null = False
    definitions = Definitions()
    result = from_json_schema_type(data, type_string, allow_null, definitions)
    assert result.additional_items is True


# LLM-generated content at query #33
#--------------------------

```python
def test_from_json_schema_type_with_invalid_type_string():
    data = {}
    type_string = "invalid_type"
    allow_null = False
    definitions = Definitions()
    try:
        from_json_schema_type(data, type_string, allow_null, definitions)
        assert False, "Expected AssertionError but none was raised"
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid_type'"


# LLM-generated content at query #34
#--------------------------

```python
def test_from_json_schema_type_array_additional_items_none():
    data = {"type": "array"}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert result.additional_items is True


# LLM-generated content at query #35
#--------------------------

```python
def test_from_json_schema_type_with_additional_items_as_bool():
    data = {"type": "array", "additionalItems": False}
    result = from_json_schema_type(data, "array", False, {})
    assert isinstance(result, Array)
    assert result.additional_items is False


# LLM-generated content at query #36
#--------------------------

```python
def test_pattern_regex_predicate():
    field = String(pattern_regex=re.compile("test", re.UNICODE))
    assert field.pattern_regex is not None


# LLM-generated content at query #37
#--------------------------

```python
def test_pattern_properties_is_none():
    data = {}
    type_string = "object"
    allow_null = False
    definitions = Definitions()
    result = from_json_schema_type(data, type_string, allow_null, definitions)
    assert result.pattern_properties is None


# LLM-generated content at query #38
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"type": "number", "minimum": 0, "maximum": 100}
    definitions = Definitions()
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.allow_null is False

def test_from_json_schema_type_integer():
    data = {"type": "integer", "minimum": 0, "maximum": 100}
    definitions = Definitions()
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.allow_null is False

def test_from_json_schema_type_string():
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    definitions = Definitions()
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.allow_null is False

def test_from_json_schema_type_boolean():
    data = {"type": "boolean"}
    definitions = Definitions()
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.allow_null is False

def test_from_json_schema_type_array():
    data = {"type": "array", "minItems": 1, "maxItems": 10}
    definitions = Definitions()
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.allow_null is False

def test_from_json_schema_type_object():
    data = {"type": "object", "minProperties": 1, "maxProperties": 10}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.allow_null is False

def test_from_json_schema_type_with_default():
    data = {"type": "number", "default": 42}
    definitions = Definitions()
    field = from_json_schema_type(data, "number", False, definitions)
    assert field.get_default_value() == 42

def test_from_json_schema_type_with_nullable():
    data = {"type": "number"}
    definitions = Definitions()
    field = from_json_schema_type(data, "number", True, definitions)
    assert field.allow_null is True


# LLM-generated content at query #39
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 100, "default": 50}
    result = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(result, Float)
    assert result.allow_null == False
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.default == 50
    assert result.coerce_types == False

def test_from_json_schema_type_integer():
    data = {"minimum": 0, "maximum": 100, "default": 50}
    result = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(result, Integer)
    assert result.allow_null == False
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.default == 50
    assert result.coerce_types == False

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 10, "format": "email", "pattern": "^[a-zA-Z0-9]+$", "default": "test"}
    result = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(result, String)
    assert result.allow_null == False
    assert result.allow_blank == False
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.format == "email"
    assert result.pattern == "^[a-zA-Z0-9]+$"
    assert result.default == "test"
    assert result.coerce_types == False

def test_from_json_schema_type_boolean():
    data = {"default": True}
    result = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(result, Boolean)
    assert result.allow_null == False
    assert result.default == True
    assert result.coerce_types == False

def test_from_json_schema_type_array():
    data = {"minItems": 1, "maxItems": 5, "uniqueItems": True, "default": []}
    result = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(result, Array)
    assert result.allow_null == False
    assert result.min_items == 1
    assert result.max_items == 5
    assert result.unique_items == True
    assert result.default == []

def test_from_json_schema_type_object():
    data = {"minProperties": 1, "maxProperties": 5, "default": {}}
    result = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(result, Object)
    assert result.allow_null == False
    assert result.min_properties == 1
    assert result.max_properties == 5
    assert result.default == {}

def test_from_json_schema_type_array_with_items():
    data = {"items": {"type": "string"}, "additionalItems": False}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    assert result.additional_items == False

def test_from_json_schema_type_object_with_properties():
    data = {"properties": {"name": {"type": "string"}}, "additionalProperties": False}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.properties["name"], String)
    assert result.additional_properties == False


# LLM-generated content at query #40
#--------------------------

```python
def test_additional_items_none():
    field = Array(additional_items=None)
    result = to_json_schema(field)
    assert "additionalItems" not in result


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_to_json_schema_with_any_field():
    assert to_json_schema(Any()) == True

def test_to_json_schema_with_never_match_field():
    assert to_json_schema(NeverMatch()) == False

def test_to_json_schema_with_string_field():
    field = String()
    assert to_json_schema(field) == {"type": "string"}

def test_to_json_schema_with_string_field_allow_null():
    field = String(allow_null=True)
    assert to_json_schema(field) == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_min_length():
    field = String(min_length=5)
    assert to_json_schema(field) == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_max_length():
    field = String(max_length=10)
    assert to_json_schema(field) == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    assert to_json_schema(field) == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_format():
    field = String(format="email")
    assert to_json_schema(field) == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    assert to_json_schema(field) == {"type": "integer"}

def test_to_json_schema_with_integer_field_allow_null():
    field = Integer(allow_null=True)
    assert to_json_schema(field) == {"type": ["integer", "null"]}

def test_to_json_schema_with_integer_field_minimum():
    field = Integer(minimum=0)
    assert to_json_schema(field) == {"type": "integer", "minimum": 0}

def test_to_json_schema_with_integer_field_maximum():
    field = Integer(maximum=100)
    assert to_json_schema(field) == {"type": "integer", "maximum": 100}

def test_to_json_schema_with_float_field():
    field = Float()
    assert to_json_schema(field) == {"type": "number"}

def test_to_json_schema_with_float_field_allow_null():
    field = Float(allow_null=True)
    assert to_json_schema(field) == {"type": ["number", "null"]}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    assert to_json_schema(field) == {"type": "boolean"}

def test_to_json_schema_with_boolean_field_allow_null():
    field = Boolean(allow_null=True)
    assert to_json_schema(field) == {"type": ["boolean", "null"]}

def test_to_json_schema_with_array_field():
    field = Array()
    assert to_json_schema(field) == {"type": "array"}

def test_to_json_schema_with_array_field_allow_null():
    field = Array(allow_null=True)
    assert to_json_schema(field) == {"type": ["array", "null"]}

def test_to_json_schema_with_array_field_min_items():
    field = Array(min_items=1)
    assert to_json_schema(field) == {"type": "array", "minItems": 1}

def test_to_json_schema_with_array_field_max_items():
    field = Array(max_items=10)
    assert to_json_schema(field) == {"type": "array", "maxItems": 10}

def test_to_json_schema_with_array_field_items():
    field = Array(items=String())
    assert to_json_schema(field) == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_array_field_additional_items():
    field = Array(additional_items=Boolean())
    assert to_json_schema(field) == {"type": "array", "additionalItems": {"type": "boolean"}}

def test_to_json_schema_with_array_field_unique_items():
    field = Array(unique_items=True)
    assert to_json_schema(field) == {"type": "array", "uniqueItems": True}

def test_to_json_schema_with_object_field():
    field = Object()
    assert to_json_schema(field) == {"type": "object"}

def test_to_json_schema_with_object_field_allow_null():
    field = Object(allow_null=True)
    assert to_json_schema(field) == {"type": ["object", "null"]}

def test_to_json_schema_with_object_field_properties():
    field = Object(properties={"name": String(), "age": Integer()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_object_field_pattern_properties():
    field = Object(pattern_properties={r"^[a-z]+$": String()})
    assert to_json_schema(field) == {"type": "object", "patternProperties": {"^[a-z]+$": {"type": "string"}}}

def test_to_json_schema_with_object_field_additional_properties():
    field = Object(additional_properties=Boolean())
    assert to_json_schema(field) == {"type": "object", "additionalProperties": {"type": "boolean"}}

def test_to_json_schema_with_object_field_property_names():
    field = Object(property_names=String())
    assert to_json_schema(field) == {"type": "object", "propertyNames": {"type": "string"}}

def test_to_json_schema_with_object_field_max_properties():
    field = Object(max_properties=10)
    assert to_json_schema(field) == {"type": "object", "maxProperties": 10}

def test_to_json_schema_with_object_field_min_properties():
    field = Object(min_properties=1)
    assert to_json_schema(field) == {"type": "object", "minProperties": 1}

def test_to_json_schema_with_object_field_required():
    field = Object(required=["name"])
    assert to_json_schema(field) == {"type": "object", "required": ["name"]}

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String(), "age": Integer()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_schema_field_allow_null():
    field = Schema(allow_null=True, fields={"name": String()})
    assert to_json_schema(field) == {"type": ["object", "null"], "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_schema_field_required():
    field = Schema(required=["name"], fields={"name": String()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert to_json_schema(field) == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="fixed_value")
    assert to_json_schema(field) == {"const": "fixed_value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    assert to_json_schema(field) == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    assert to_json_schema(field) == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_all_of_field():
    field = AllOf(all_of=[String(), Integer()])
    assert to_json_schema(field) == {"allOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_if_then_else_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    assert to_json_schema(field) == {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    assert to_json_schema(field) == {"not": {"type": "string"}}

def test_to_json_schema_with_definitions():
    definitions = Definitions()
    definitions["string_field"] = String()
    definitions["integer_field"] = Integer()
    assert to_json_schema(definitions) == {"components": {"schemas": {"string_field": {"type": "string"}, "integer_field": {"type": "integer"}}}}


# LLM-generated content at query #2
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"type": "number", "minimum": 0, "maximum": 100}
    result = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.allow_null is False

def test_from_json_schema_type_integer():
    data = {"type": "integer", "minimum": 0, "maximum": 100}
    result = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.allow_null is False

def test_from_json_schema_type_string():
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    result = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.allow_null is False

def test_from_json_schema_type_boolean():
    data = {"type": "boolean"}
    result = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(result, Boolean)
    assert result.allow_null is False

def test_from_json_schema_type_array():
    data = {"type": "array", "minItems": 1, "maxItems": 10}
    result = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(result, Array)
    assert result.min_items == 1
    assert result.max_items == 10
    assert result.allow_null is False

def test_from_json_schema_type_object():
    data = {"type": "object", "minProperties": 1, "maxProperties": 10}
    result = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(result, Object)
    assert result.min_properties == 1
    assert result.max_properties == 10
    assert result.allow_null is False

def test_from_json_schema_type_with_allow_null():
    data = {"type": "number"}
    result = from_json_schema_type(data, "number", True, Definitions())
    assert isinstance(result, Float)
    assert result.allow_null is True

def test_from_json_schema_type_with_default():
    data = {"type": "number", "default": 42}
    result = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(result, Float)
    assert result.default == 42

def test_from_json_schema_type_with_items():
    data = {"type": "array", "items": {"type": "string"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

def test_from_json_schema_type_with_additional_items():
    data = {"type": "array", "additionalItems": {"type": "number"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.additional_items, Float)

def test_from_json_schema_type_with_properties():
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.properties["name"], String)

def test_from_json_schema_type_with_pattern_properties():
    data = {"type": "object", "patternProperties": {"^S_": {"type": "string"}}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.pattern_properties["^S_"], String)

def test_from_json_schema_type_with_additional_properties():
    data = {"type": "object", "additionalProperties": {"type": "number"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.additional_properties, Float)

def test_from_json_schema_type_with_property_names():
    data = {"type": "object", "propertyNames": {"type": "string"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.property_names, String)


# LLM-generated content at query #3
#--------------------------

```python
def test_to_json_schema_with_any_field():
    result = to_json_schema(Any())
    assert result is True

def test_to_json_schema_with_never_match_field():
    result = to_json_schema(NeverMatch())
    assert result is False

def test_to_json_schema_with_string_field():
    field = String(min_length=5, max_length=10, pattern_regex=re.compile(r"^[a-z]+$"), format="email")
    result = to_json_schema(field)
    assert result == {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "format": "email"
    }

def test_to_json_schema_with_integer_field():
    field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=True, multiple_of=2)
    result = to_json_schema(field)
    assert result == {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True,
        "multipleOf": 2
    }

def test_to_json_schema_with_float_field():
    field = Float(minimum=0.0, maximum=100.0)
    result = to_json_schema(field)
    assert result == {
        "type": "number",
        "minimum": 0.0,
        "maximum": 100.0
    }

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    assert result == {"type": "boolean"}

def test_to_json_schema_with_array_field():
    field = Array(min_items=1, max_items=10, items=String(), additional_items=False, unique_items=True)
    result = to_json_schema(field)
    assert result == {
        "type": "array",
        "minItems": 1,
        "maxItems": 10,
        "items": {"type": "string"},
        "additionalItems": False,
        "uniqueItems": True
    }

def test_to_json_schema_with_object_field():
    field = Object(
        properties={"name": String()},
        pattern_properties={r"^S_": String()},
        additional_properties=False,
        property_names=String(),
        max_properties=10,
        min_properties=1,
        required=["name"]
    )
    result = to_json_schema(field)
    assert result == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "patternProperties": {"^S_": {"type": "string"}},
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "maxProperties": 10,
        "minProperties": 1,
        "required": ["name"]
    }

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert result == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(field)
    assert result == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result == {"const": "fixed_value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_all_of_field():
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"allOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_if_then_else_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(field)
    assert result == {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    result = to_json_schema(field)
    assert result == {"not": {"type": "string"}}

def test_to_json_schema_with_definitions():
    definitions = Definitions()
    definitions["string_field"] = String()
    result = to_json_schema(definitions)
    assert result == {
        "components": {
            "schemas": {
                "string_field": {"type": "string"}
            }
        }
    }

def test_to_json_schema_with_reference_field():
    target = String()
    field = Reference(to="string_ref", target=target)
    result = to_json_schema(field)
    assert result == {
        "$ref": "#/components/schemas/string_ref",
        "components": {
            "schemas": {
                "string_ref": {"type": "string"}
            }
        }
    }

def test_to_json_schema_with_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"]}

def test_to_json_schema_with_default_value():
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_invalid_field_type():
    with pytest.raises(ValueError):
        to_json_schema("invalid_field_type")


# LLM-generated content at query #4
#--------------------------

```python
def test_type_from_json_schema_single_type():
    data = {"type": "string"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    assert result.allow_null is False

def test_type_from_json_schema_multiple_types():
    data = {"type": ["string", "integer"]}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Integer)
    assert result.allow_null is False

def test_type_from_json_schema_with_null():
    data = {"type": ["string", "null"]}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True

def test_type_from_json_schema_no_type():
    data = {}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, NeverMatch)

def test_type_from_json_schema_null_only():
    data = {"type": "null"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)
    assert result.const is None

def test_type_from_json_schema_integer():
    data = {"type": "integer", "minimum": 0, "maximum": 100}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.maximum == 100

def test_type_from_json_schema_number():
    data = {"type": "number", "multipleOf": 0.5}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Float)
    assert result.multiple_of == 0.5

def test_type_from_json_schema_boolean():
    data = {"type": "boolean"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Boolean)
    assert result.allow_null is False

def test_type_from_json_schema_array():
    data = {"type": "array", "items": {"type": "string"}}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

def test_type_from_json_schema_object():
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert isinstance(result.properties["name"], String)


# LLM-generated content at query #5
#--------------------------

```python
def test_isinstance_items_list():
    items = [{"type": "string"}]
    assert isinstance(items, list)


# LLM-generated content at query #6
#--------------------------

```python
def test_if_clause_without_else_clause():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=None)
    result = to_json_schema(field)
    assert "else" not in result


# LLM-generated content at query #7
#--------------------------

```python
def test_if_then_else_from_json_schema_with_all_clauses():
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"},
        "default": 42
    }
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Float)
    assert isinstance(field.else_clause, Boolean)
    assert field.default == 42

def test_if_then_else_from_json_schema_without_then_clause():
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
        "default": 42
    }
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Boolean)
    assert field.default == 42

def test_if_then_else_from_json_schema_without_else_clause():
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "default": 42
    }
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Float)
    assert isinstance(field.else_clause, Any)
    assert field.default == 42

def test_if_then_else_from_json_schema_without_default():
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"}
    }
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Float)
    assert isinstance(field.else_clause, Boolean)
    assert field.default == NO_DEFAULT


# LLM-generated content at query #8
#--------------------------

```python
def test_additional_items_predicate_false():
    field = Array(items=String(), additional_items=None)
    result = to_json_schema(field)
    assert "additionalItems" not in result


# LLM-generated content at query #9
#--------------------------

```python
def test_from_json_schema_type_object_additional_properties_bool():
    data = {"type": "object", "additionalProperties": True}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert field.additional_properties is True


# LLM-generated content at query #10
#--------------------------

```python
def test_from_json_schema_with_bool_true():
    result = from_json_schema(True)
    assert isinstance(result, Any)

def test_from_json_schema_with_bool_false():
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

def test_from_json_schema_with_ref():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/test"}
    definitions["#/components/schemas/test"] = Any()
    result = from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/test"

def test_from_json_schema_with_type_constraint():
    data = {"type": "string"}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 1
    assert isinstance(result.any_of[0], String)

def test_from_json_schema_with_enum():
    data = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)
    assert len(result.choices) == 3

def test_from_json_schema_with_const():
    data = {"const": "test"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == "test"

def test_from_json_schema_with_all_of():
    data = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_from_json_schema_with_any_of():
    data = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

def test_from_json_schema_with_one_of():
    data = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2

def test_from_json_schema_with_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)
    assert isinstance(result.negated, Union)

def test_from_json_schema_with_if_then_else():
    data = {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Union)
    assert isinstance(result.then_clause, Union)
    assert isinstance(result.else_clause, Union)

def test_from_json_schema_with_multiple_constraints():
    data = {"type": "string", "enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_from_json_schema_with_no_constraints():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_with_definitions():
    data = {"components": {"schemas": {"test": {"type": "string"}}}}
    result = from_json_schema(data)
    assert isinstance(result, Any)


# LLM-generated content at query #11
#--------------------------

```python
def test_to_json_schema_with_any_field():
    result = to_json_schema(Any())
    assert result is True

def test_to_json_schema_with_never_match_field():
    result = to_json_schema(NeverMatch())
    assert result is False

def test_to_json_schema_with_string_field():
    field = String(min_length=1, max_length=10, pattern_regex=re.compile(r"^[a-z]+$"), format="email")
    result = to_json_schema(field)
    assert result == {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "format": "email"
    }

def test_to_json_schema_with_integer_field():
    field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=True, multiple_of=2)
    result = to_json_schema(field)
    assert result == {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True,
        "multipleOf": 2
    }

def test_to_json_schema_with_float_field():
    field = Float(minimum=0.0, maximum=100.0, exclusive_minimum=True, exclusive_maximum=True, multiple_of=0.5)
    result = to_json_schema(field)
    assert result == {
        "type": "number",
        "minimum": 0.0,
        "maximum": 100.0,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True,
        "multipleOf": 0.5
    }

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    assert result == {"type": "boolean"}

def test_to_json_schema_with_array_field():
    field = Array(min_items=1, max_items=10, items=String(), additional_items=False, unique_items=True)
    result = to_json_schema(field)
    assert result == {
        "type": "array",
        "minItems": 1,
        "maxItems": 10,
        "items": {"type": "string"},
        "additionalItems": False,
        "uniqueItems": True
    }

def test_to_json_schema_with_object_field():
    field = Object(
        properties={"name": String()},
        pattern_properties={"^S_": String()},
        additional_properties=False,
        property_names=String(),
        max_properties=10,
        min_properties=1,
        required=["name"]
    )
    result = to_json_schema(field)
    assert result == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "patternProperties": {"^S_": {"type": "string"}},
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "maxProperties": 10,
        "minProperties": 1,
        "required": ["name"]
    }

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert result == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", 1), ("b", 2)])
    result = to_json_schema(field)
    assert result == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="constant_value")
    result = to_json_schema(field)
    assert result == {"const": "constant_value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_all_of_field():
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"allOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_if_then_else_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(field)
    assert result == {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    result = to_json_schema(field)
    assert result == {"not": {"type": "string"}}

def test_to_json_schema_with_reference_field():
    target = String()
    field = Reference(to="test", target=target)
    result = to_json_schema(field)
    assert result == {
        "$ref": "#/components/schemas/test",
        "components": {"schemas": {"test": {"type": "string"}}}
    }

def test_to_json_schema_with_definitions():
    definitions = Definitions({"field1": String(), "field2": Integer()})
    result = to_json_schema(definitions)
    assert result == {
        "components": {
            "schemas": {
                "field1": {"type": "string"},
                "field2": {"type": "integer"}
            }
        }
    }

def test_to_json_schema_with_default_value():
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"]}


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_158_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=Integer())
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_158_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=Integer())
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #14
#--------------------------

```python
def test_pattern_regex_is_not_none():
    field = String(pattern_regex=re.compile(r"^test$", re.UNICODE))
    assert field.pattern_regex is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_string_field_with_pattern_regex():
    field = String(pattern_regex=re.compile(r"^[a-z]+$", re.RegexFlag.UNICODE))
    result = to_json_schema(field)
    assert "pattern" in result
    assert result["pattern"] == r"^[a-z]+$"


# LLM-generated content at query #16
#--------------------------

```python
def test_array_items_is_list_or_tuple():
    field = Array(items=[String(), Integer()])
    result = to_json_schema(field)
    assert isinstance(result["items"], list)


# LLM-generated content at query #17
#--------------------------

```python
def test_from_json_schema_type_invalid_type_string():
    data = {}
    type_string = "invalid_type"
    allow_null = False
    definitions = Definitions()
    try:
        from_json_schema_type(data, type_string, allow_null, definitions)
        assert False, "Expected AssertionError but none was raised"
    except AssertionError as e:
        assert str(e) == f"Invalid argument type_string={type_string!r}"


# LLM-generated content at query #18
#--------------------------

```python
def test_additional_items_predicate():
    array_field = Array(additional_items=None)
    result = to_json_schema(array_field)
    assert "additionalItems" not in result


# LLM-generated content at query #19
#--------------------------

```python
def test_to_json_schema_with_any_field():
    field = Any()
    result = to_json_schema(field)
    assert result is True

def test_to_json_schema_with_never_match_field():
    field = NeverMatch()
    result = to_json_schema(field)
    assert result is False

def test_to_json_schema_with_string_field():
    field = String()
    result = to_json_schema(field)
    assert result == {"type": "string"}

def test_to_json_schema_with_string_field_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_with_default():
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_string_field_with_min_length():
    field = String(min_length=5)
    result = to_json_schema(field)
    assert result == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_with_max_length():
    field = String(max_length=10)
    result = to_json_schema(field)
    assert result == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_with_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    result = to_json_schema(field)
    assert result == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_with_format():
    field = String(format="email")
    result = to_json_schema(field)
    assert result == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    result = to_json_schema(field)
    assert result == {"type": "integer"}

def test_to_json_schema_with_integer_field_allow_null():
    field = Integer(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["integer", "null"]}

def test_to_json_schema_with_integer_field_with_default():
    field = Integer(default=42)
    result = to_json_schema(field)
    assert result == {"type": "integer", "default": 42}

def test_to_json_schema_with_integer_field_with_minimum():
    field = Integer(minimum=0)
    result = to_json_schema(field)
    assert result == {"type": "integer", "minimum": 0}

def test_to_json_schema_with_integer_field_with_maximum():
    field = Integer(maximum=100)
    result = to_json_schema(field)
    assert result == {"type": "integer", "maximum": 100}

def test_to_json_schema_with_integer_field_with_exclusive_minimum():
    field = Integer(exclusive_minimum=0)
    result = to_json_schema(field)
    assert result == {"type": "integer", "exclusiveMinimum": 0}

def test_to_json_schema_with_integer_field_with_exclusive_maximum():
    field = Integer(exclusive_maximum=100)
    result = to_json_schema(field)
    assert result == {"type": "integer", "exclusiveMaximum": 100}

def test_to_json_schema_with_integer_field_with_multiple_of():
    field = Integer(multiple_of=2)
    result = to_json_schema(field)
    assert result == {"type": "integer", "multipleOf": 2}

def test_to_json_schema_with_float_field():
    field = Float()
    result = to_json_schema(field)
    assert result == {"type": "number"}

def test_to_json_schema_with_float_field_allow_null():
    field = Float(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["number", "null"]}

def test_to_json_schema_with_float_field_with_default():
    field = Float(default=3.14)
    result = to_json_schema(field)
    assert result == {"type": "number", "default": 3.14}

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    result = to_json_schema(field)
    assert result == {"type": "number"}

def test_to_json_schema_with_decimal_field_allow_null():
    field = Decimal(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["number", "null"]}

def test_to_json_schema_with_decimal_field_with_default():
    field = Decimal(default=Decimal("3.14"))
    result = to_json_schema(field)
    assert result == {"type": "number", "default": "3.14"}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    assert result == {"type": "boolean"}

def test_to_json_schema_with_boolean_field_allow_null():
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["boolean", "null"]}

def test_to_json_schema_with_boolean_field_with_default():
    field = Boolean(default=True)
    result = to_json_schema(field)
    assert result == {"type": "boolean", "default": True}

def test_to_json_schema_with_array_field():
    field = Array()
    result = to_json_schema(field)
    assert result == {"type": "array"}

def test_to_json_schema_with_array_field_allow_null():
    field = Array(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["array", "null"]}

def test_to_json_schema_with_array_field_with_default():
    field = Array(default=[])
    result = to_json_schema(field)
    assert result == {"type": "array", "default": []}

def test_to_json_schema_with_array_field_with_min_items():
    field = Array(min_items=1)
    result = to_json_schema(field)
    assert result == {"type": "array", "minItems": 1}

def test_to_json_schema_with_array_field_with_max_items():
    field = Array(max_items=10)
    result = to_json_schema(field)
    assert result == {"type": "array", "maxItems": 10}

def test_to_json_schema_with_array_field_with_items():
    field = Array(items=String())
    result = to_json_schema(field)
    assert result == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_array_field_with_additional_items():
    field = Array(items=[String(), Integer()], additional_items=Boolean())
    result = to_json_schema(field)
    assert result == {"type": "array", "items": [{"type": "string"}, {"type": "integer"}], "additionalItems": {"type": "boolean"}}

def test_to_json_schema_with_array_field_with_unique_items():
    field = Array(unique_items=True)
    result = to_json_schema(field)
    assert result == {"type": "array", "uniqueItems": True}

def test_to_json_schema_with_object_field():
    field = Object()
    result = to_json_schema(field)
    assert result == {"type": "object"}

def test_to_json_schema_with_object_field_allow_null():
    field = Object(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["object", "null"]}

def test_to_json_schema_with_object_field_with_default():
    field = Object(default={})
    result = to_json_schema(field)
    assert result == {"type": "object", "default": {}}

def test_to_json_schema_with_object_field_with_properties():
    field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_object_field_with_pattern_properties():
    field = Object(pattern_properties={r"^S_": String(), r"^I_": Integer()})
    result = to_json_schema(field)
    assert result == {"type": "object", "patternProperties": {"^S_": {"type": "string"}, "^I_": {"type": "integer"}}}

def test_to_json_schema_with_object_field_with_additional_properties():
    field = Object(additional_properties=Boolean())
    result = to_json_schema(field)
    assert result == {"type": "object", "additionalProperties": {"type": "boolean"}}

def test_to_json_schema_with_object_field_with_property_names():
    field = Object(property_names=String())
    result = to_json_schema(field)
    assert result == {"type": "object", "propertyNames": {"type": "string"}}

def test_to_json_schema_with_object_field_with_max_properties():
    field = Object(max_properties=10)
    result = to_json_schema(field)
    assert result == {"type": "object", "maxProperties": 10}

def test_to_json_schema_with_object_field_with_min_properties():
    field = Object(min_properties=1)
    result = to_json_schema(field)
    assert result == {"type": "object", "minProperties": 1}

def test_to_json_schema_with_object_field_with_required():
    field = Object(required=["name"])
    result = to_json_schema(field)
    assert result == {"type": "object", "required": ["name"]}

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String(), "age": Integer()})
    result = to


# LLM-generated content at query #20
#--------------------------

```python
def test_to_json_schema_with_any_field():
    field = Any()
    result = to_json_schema(field)
    assert result is True

def test_to_json_schema_with_never_match_field():
    field = NeverMatch()
    result = to_json_schema(field)
    assert result is False

def test_to_json_schema_with_string_field():
    field = String()
    result = to_json_schema(field)
    assert result == {"type": "string"}

def test_to_json_schema_with_string_field_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_with_default():
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_string_field_with_min_length():
    field = String(min_length=5)
    result = to_json_schema(field)
    assert result == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_with_max_length():
    field = String(max_length=10)
    result = to_json_schema(field)
    assert result == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_with_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    result = to_json_schema(field)
    assert result == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_with_format():
    field = String(format="email")
    result = to_json_schema(field)
    assert result == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    result = to_json_schema(field)
    assert result == {"type": "integer"}

def test_to_json_schema_with_float_field():
    field = Float()
    result = to_json_schema(field)
    assert result == {"type": "number"}

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    result = to_json_schema(field)
    assert result == {"type": "number"}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    assert result == {"type": "boolean"}

def test_to_json_schema_with_array_field():
    field = Array()
    result = to_json_schema(field)
    assert result == {"type": "array"}

def test_to_json_schema_with_object_field():
    field = Object()
    result = to_json_schema(field)
    assert result == {"type": "object"}

def test_to_json_schema_with_schema_field():
    field = Schema()
    result = to_json_schema(field)
    assert result == {"type": "object"}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(field)
    assert result == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="constant_value")
    result = to_json_schema(field)
    assert result == {"const": "constant_value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_definitions():
    definitions = Definitions()
    definitions["string_field"] = String()
    definitions["integer_field"] = Integer()
    result = to_json_schema(definitions)
    assert result == {
        "components": {
            "schemas": {
                "string_field": {"type": "string"},
                "integer_field": {"type": "integer"}
            }
        }
    }

def test_to_json_schema_with_reference_field():
    target_field = String()
    reference_field = Reference(to="string_field", target=target_field)
    result = to_json_schema(reference_field)
    assert result == {
        "$ref": "#/components/schemas/string_field",
        "components": {
            "schemas": {
                "string_field": {"type": "string"}
            }
        }
    }

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_all_of_field():
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {"allOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_if_then_else_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(field)
    assert result == {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    result = to_json_schema(field)
    assert result == {"not": {"type": "string"}}


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_34():
    field = String(pattern_regex=re.compile(r"test", re.UNICODE))
    assert field.pattern_regex.flags == re.RegexFlag.UNICODE


# LLM-generated content at query #22
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"type": "number", "minimum": 0, "maximum": 100}
    result = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.allow_null is False

def test_from_json_schema_type_integer():
    data = {"type": "integer", "minimum": 0, "maximum": 100}
    result = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.allow_null is False

def test_from_json_schema_type_string():
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    result = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.allow_null is False

def test_from_json_schema_type_boolean():
    data = {"type": "boolean"}
    result = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(result, Boolean)
    assert result.allow_null is False

def test_from_json_schema_type_array():
    data = {"type": "array", "minItems": 1, "maxItems": 5}
    result = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(result, Array)
    assert result.min_items == 1
    assert result.max_items == 5
    assert result.allow_null is False

def test_from_json_schema_type_object():
    data = {"type": "object", "minProperties": 1, "maxProperties": 5}
    result = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(result, Object)
    assert result.min_properties == 1
    assert result.max_properties == 5
    assert result.allow_null is False

def test_from_json_schema_type_with_default():
    data = {"type": "string", "default": "test"}
    result = from_json_schema_type(data, "string", False, Definitions())
    assert result.default == "test"

def test_from_json_schema_type_with_nullable():
    data = {"type": "string"}
    result = from_json_schema_type(data, "string", True, Definitions())
    assert result.allow_null is True

def test_from_json_schema_type_array_with_items():
    data = {"type": "array", "items": {"type": "string"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

def test_from_json_schema_type_array_with_additional_items():
    data = {"type": "array", "items": {"type": "string"}, "additionalItems": {"type": "number"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.additional_items, Float)

def test_from_json_schema_type_object_with_properties():
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.properties["name"], String)

def test_from_json_schema_type_object_with_pattern_properties():
    data = {"type": "object", "patternProperties": {"^S_": {"type": "string"}}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.pattern_properties["^S_"], String)

def test_from_json_schema_type_object_with_additional_properties():
    data = {"type": "object", "additionalProperties": {"type": "string"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.additional_properties, String)

def test_from_json_schema_type_object_with_property_names():
    data = {"type": "object", "propertyNames": {"type": "string"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.property_names, String)

def test_from_json_schema_type_object_with_required():
    data = {"type": "object", "required": ["name"]}
    result = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(result, Object)
    assert result.required == ["name"]


# LLM-generated content at query #23
#--------------------------

```python
def test_pattern_regex_unicode_flag():
    field = String(pattern_regex=re.compile(r"^test$", re.UNICODE))
    result = to_json_schema(field)
    assert result["pattern"] == "^test$"


