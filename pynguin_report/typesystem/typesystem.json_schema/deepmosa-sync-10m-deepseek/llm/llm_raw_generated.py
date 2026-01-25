####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_to_json_schema_with_union_field():
    from typesystem.fields import Field, Union
    from typesystem.json_schema import to_json_schema
    field1 = Field(allow_null=True)
    field2 = Field()
    union_field = Union(any_of=[field1, field2])
    result = to_json_schema(union_field)
    expected = {"anyOf": [{"type": ["null"]}, {}], "components": {"schemas": {}}}
    assert result == expected

def test_to_json_schema_with_definitions():
    from typesystem.schemas import Definitions
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    definitions = Definitions({"Test": Field()})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Test": {}}}}
    assert result == expected

def test_to_json_schema_with_union_and_definitions():
    from typesystem.schemas import Definitions
    from typesystem.fields import Field, Union
    from typesystem.json_schema import to_json_schema
    field1 = Field()
    field2 = Field()
    union_field = Union(any_of=[field1, field2])
    definitions = Definitions({"UnionField": union_field})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"UnionField": {"anyOf": [{}, {}]}}}}
    assert result == expected

def test_to_json_schema_with_union_allow_null():
    from typesystem.fields import Field, Union
    from typesystem.json_schema import to_json_schema
    field1 = Field(allow_null=True)
    field2 = Field(allow_null=False)
    union_field = Union(any_of=[field1, field2])
    result = to_json_schema(union_field)
    expected = {"anyOf": [{"type": ["null"]}, {}], "components": {"schemas": {}}}
    assert result == expected

def test_to_json_schema_with_union_default():
    from typesystem.fields import Field, Union
    from typesystem.json_schema import to_json_schema
    field1 = Field(default="default1")
    field2 = Field(default="default2")
    union_field = Union(any_of=[field1, field2])
    result = to_json_schema(union_field)
    expected = {"anyOf": [{"default": "default1"}, {"default": "default2"}], "components": {"schemas": {}}}
    assert result == expected

def test_to_json_schema_with_union_and_reference():
    from typesystem.fields import Field, Union, Reference
    from typesystem.json_schema import to_json_schema
    field1 = Field()
    field2 = Reference(to="Target")
    union_field = Union(any_of=[field1, field2])
    result = to_json_schema(union_field)
    expected = {"anyOf": [{}, {"$ref": "#/components/schemas/Target"}], "components": {"schemas": {"Target": {}}}}
    assert result == expected

def test_to_json_schema_with_union_and_nested_definitions():
    from typesystem.schemas import Definitions
    from typesystem.fields import Field, Union
    from typesystem.json_schema import to_json_schema
    inner_field = Field()
    union_field = Union(any_of=[inner_field])
    definitions = Definitions({"Inner": inner_field, "Union": union_field})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Inner": {}, "Union": {"anyOf": [{}]}}}}
    assert result == expected

def test_to_json_schema_with_union_and_complex_types():
    from typesystem.fields import Field, Union, String, Integer
    from typesystem.json_schema import to_json_schema
    string_field = String(min_length=1, max_length=10)
    integer_field = Integer(minimum=0, maximum=100)
    union_field = Union(any_of=[string_field, integer_field])
    result = to_json_schema(union_field)
    expected = {"anyOf": [{"type": "string", "minLength": 1, "maxLength": 10}, {"type": "integer", "minimum": 0, "maximum": 100}], "components": {"schemas": {}}}
    assert result == expected

def test_to_json_schema_with_union_and_allow_null_in_child():
    from typesystem.fields import Field, Union, String
    from typesystem.json_schema import to_json_schema
    string_field = String(allow_null=True)
    field = Field()
    union_field = Union(any_of=[string_field, field])
    result = to_json_schema(union_field)
    expected = {"anyOf": [{"type": ["string", "null"]}, {}], "components": {"schemas": {}}}
    assert result == expected

def test_to_json_schema_with_union_and_multiple_allow_null():
    from typesystem.fields import Field, Union
    from typesystem.json_schema import to_json_schema
    field1 = Field(allow_null=True)
    field2 = Field(allow_null=True)
    union_field = Union(any_of=[field1, field2])
    result = to_json_schema(union_field)
    expected = {"anyOf": [{"type": ["null"]}, {"type": ["null"]}], "components": {"schemas": {}}}
    assert result == expected

def test_to_json_schema_with_union_and_no_allow_null():
    from typesystem.fields import Field, Union
    from typesystem.json_schema import to_json_schema
    field1 = Field(allow_null=False)
    field2 = Field(allow_null=False)
    union_field = Union(any_of=[field1, field2])
    result = to_json_schema(union_field)
    expected = {"anyOf": [{}, {}], "components": {"schemas": {}}}
    assert result == expected


# LLM-generated content at query #2
#--------------------------

def test_to_json_schema_with_union_field():
    from typesystem.fields import String, Integer, Union
    from typesystem.json_schema import to_json_schema
    field = String() | Integer()
    result = to_json_schema(field)
    expected = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    assert result == expected

def test_to_json_schema_with_union_field_allow_null():
    from typesystem.fields import String, Integer, Union
    from typesystem.json_schema import to_json_schema
    field = String(allow_null=True) | Integer()
    result = to_json_schema(field)
    expected = {
        "anyOf": [
            {"type": ["string", "null"]},
            {"type": "integer"}
        ]
    }
    assert result == expected

def test_to_json_schema_with_union_field_and_default():
    from typesystem.fields import String, Integer, Union
    from typesystem.json_schema import to_json_schema
    field = String(default="default") | Integer()
    result = to_json_schema(field)
    expected = {
        "anyOf": [
            {"type": "string", "default": "default"},
            {"type": "integer"}
        ]
    }
    assert result == expected

def test_to_json_schema_with_definitions():
    from typesystem.fields import String, Integer, Union
    from typesystem.schemas import Definitions
    from typesystem.json_schema import to_json_schema
    definitions = Definitions({"StringField": String(), "IntegerField": Integer()})
    result = to_json_schema(definitions)
    expected = {
        "components": {
            "schemas": {
                "StringField": {"type": "string"},
                "IntegerField": {"type": "integer"}
            }
        }
    }
    assert result == expected

def test_to_json_schema_with_reference_field():
    from typesystem.fields import String, Reference
    from typesystem.json_schema import to_json_schema
    target = String()
    field = Reference(to="TargetSchema", target=target)
    result = to_json_schema(field)
    expected = {
        "$ref": "#/components/schemas/TargetSchema",
        "components": {
            "schemas": {
                "TargetSchema": {"type": "string"}
            }
        }
    }
    assert result == expected

def test_to_json_schema_with_string_field():
    from typesystem.fields import String
    from typesystem.json_schema import to_json_schema
    field = String(min_length=1, max_length=10, pattern_regex="^[a-z]+$", format="email")
    result = to_json_schema(field)
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "format": "email"
    }
    assert result == expected

def test_to_json_schema_with_integer_field():
    from typesystem.fields import Integer
    from typesystem.json_schema import to_json_schema
    field = Integer(minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100, multiple_of=5)
    result = to_json_schema(field)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 5
    }
    assert result == expected

def test_to_json_schema_with_boolean_field():
    from typesystem.fields import Boolean
    from typesystem.json_schema import to_json_schema
    field = Boolean(default=True)
    result = to_json_schema(field)
    expected = {
        "type": "boolean",
        "default": True
    }
    assert result == expected

def test_to_json_schema_with_array_field():
    from typesystem.fields import Array, String
    from typesystem.json_schema import to_json_schema
    items = String()
    field = Array(items=items, min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(field)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True
    }
    assert result == expected

def test_to_json_schema_with_object_field():
    from typesystem.fields import Object, String, Integer
    from typesystem.json_schema import to_json_schema
    properties = {"name": String(), "age": Integer()}
    field = Object(properties=properties, required=["name"], min_properties=1, max_properties=2)
    result = to_json_schema(field)
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 2
    }
    assert result == expected

def test_to_json_schema_with_choice_field():
    from typesystem.fields import Choice
    from typesystem.json_schema import to_json_schema
    choices = [("option1", "Option 1"), ("option2", "Option 2")]
    field = Choice(choices=choices)
    result = to_json_schema(field)
    expected = {
        "enum": ["option1", "option2"]
    }
    assert result == expected

def test_to_json_schema_with_const_field():
    from typesystem.fields import Const
    from typesystem.json_schema import to_json_schema
    field = Const(const="constant_value")
    result = to_json_schema(field)
    expected = {
        "const": "constant_value"
    }
    assert result == expected

def test_to_json_schema_with_oneof_field():
    from typesystem.fields import String, Integer, OneOf
    from typesystem.json_schema import to_json_schema
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    assert result == expected

def test_to_json_schema_with_allof_field():
    from typesystem.fields import String, Integer, AllOf
    from typesystem.json_schema import to_json_schema
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {
        "allOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    from typesystem.fields import String, Integer, IfThenElse
    from typesystem.json_schema import to_json_schema
    if_clause = String()
    then_clause = Integer()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause)
    result = to_json_schema(field)
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    assert result == expected

def test_to_json_schema_with_not_field():
    from typesystem.fields import String, Not
    from typesystem.json_schema import to_json_schema
    negated = String()
    field = Not(negated=negated)
    result = to_json_schema(field)
    expected = {
        "not": {"type": "string"}
    }
    assert result == expected

def test_to_json_schema_with_unknown_field_type():
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    class UnknownField(Field):
        pass
    field = UnknownField()
    try:
        to_json_schema(field)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


# LLM-generated content at query #3
#--------------------------

def test_from_json_schema_with_boolean_true():
    result = from_json_schema(True)
    assert isinstance(result, Any)

def test_from_json_schema_with_boolean_false():
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

def test_from_json_schema_with_ref():
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = Const("test")
    data = {"$ref": "#/components/schemas/Test"}
    result = from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/Test"

def test_from_json_schema_with_type_string():
    data = {"type": "string"}
    result = from_json_schema(data)
    assert isinstance(result, Field)

def test_from_json_schema_with_type_array():
    data = {"type": ["string", "number"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)

def test_from_json_schema_with_enum():
    data = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)

def test_from_json_schema_with_const():
    data = {"const": 42}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == 42

def test_from_json_schema_with_allOf():
    data = {"allOf": [{"type": "string"}, {"maxLength": 10}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_with_anyOf():
    data = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)

def test_from_json_schema_with_oneOf():
    data = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)

def test_from_json_schema_with_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)

def test_from_json_schema_with_if_then():
    data = {"if": {"type": "string"}, "then": {"maxLength": 10}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_with_if_then_else():
    data = {"if": {"type": "string"}, "then": {"maxLength": 10}, "else": {"type": "number"}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_with_multiple_constraints():
    data = {"type": "string", "enum": ["a", "b"], "maxLength": 5}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_with_no_constraints():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_with_null_type():
    data = {"type": "null"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const is None

def test_from_json_schema_with_null_in_union():
    data = {"type": ["null", "string"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert result.allow_null is True

def test_from_json_schema_with_components():
    data = {"components": {"schemas": {"Test": {"type": "string"}}}}
    result = from_json_schema(data)
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "string"}, definitions)
    assert isinstance(result, Any)

def test_from_json_schema_with_default():
    data = {"type": "string", "default": "hello"}
    result = from_json_schema(data)
    assert result.default == "hello"

def test_from_json_schema_with_number_type_includes_integer():
    data = {"type": "number"}
    result = from_json_schema(data)
    assert isinstance(result, Field)

def test_from_json_schema_with_integer_type_excludes_number():
    data = {"type": "integer"}
    result = from_json_schema(data)
    assert isinstance(result, Field)


# LLM-generated content at query #4
#--------------------------

def test_additional_properties_is_bool_false():
    field = Object(additional_properties=False)
    result = to_json_schema(field)
    assert result["additionalProperties"] == False


# LLM-generated content at query #5
#--------------------------

def test_if_then_else_without_then_clause():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=String())
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #6
#--------------------------

def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5.0}
    definitions = Definitions()
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5.0
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_integer():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    definitions = Definitions()
    field = from_json_schema_type(data, "integer", True, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5
    assert field.allow_null == True
    assert field.coerce_types == False

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 10, "format": "email", "pattern": "^[a-z]+$", "default": "hello"}
    definitions = Definitions()
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-z]+$"
    assert field.default == "hello"
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_boolean():
    data = {"default": True}
    definitions = Definitions()
    field = from_json_schema_type(data, "boolean", True, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True
    assert field.allow_null == True
    assert field.coerce_types == False

def test_from_json_schema_type_array():
    data = {"minItems": 1, "maxItems": 10, "uniqueItems": True, "default": [1, 2, 3]}
    definitions = Definitions()
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == [1, 2, 3]
    assert field.allow_null == False

def test_from_json_schema_type_object():
    data = {"minProperties": 1, "maxProperties": 10, "default": {"key": "value"}}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", True, definitions)
    assert isinstance(field, Object)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.default == {"key": "value"}
    assert field.allow_null == True


# LLM-generated content at query #7
#--------------------------

def test_if_then_else_from_json_schema_basic():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"type": "string", "minLength": 5}, "else": {"type": "integer"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result == "hello"
    result = field.validate("hi")
    assert result == "hi"
    result = field.validate(123)
    assert result == 123

def test_if_then_else_from_json_schema_without_then():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "else": {"type": "integer"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("test")
    assert result == "test"
    result = field.validate(42)
    assert result == 42

def test_if_then_else_from_json_schema_without_else():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"type": "string", "maxLength": 3}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("abc")
    assert result == "abc"
    result = field.validate(100)
    assert result == 100

def test_if_then_else_from_json_schema_with_nested_schemas():
    definitions = Definitions()
    data = {"if": {"type": "array"}, "then": {"items": {"type": "number"}}, "else": {"type": "boolean"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    result = field.validate(True)
    assert result == True

def test_if_then_else_from_json_schema_with_const():
    definitions = Definitions()
    data = {"if": {"const": "special"}, "then": {"type": "string"}, "else": {"type": "number"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("special")
    assert result == "special"
    result = field.validate(99)
    assert result == 99

def test_if_then_else_from_json_schema_with_ref_in_if():
    definitions = Definitions()
    definitions["#/components/schemas/MyString"] = from_json_schema({"type": "string"}, definitions)
    data = {"if": {"$ref": "#/components/schemas/MyString"}, "then": {"type": "string"}, "else": {"type": "integer"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("text")
    assert result == "text"
    result = field.validate(456)
    assert result == 456

def test_if_then_else_from_json_schema_with_all_of_in_then():
    definitions = Definitions()
    data = {"if": {"type": "object"}, "then": {"allOf": [{"type": "object"}, {"required": ["id"]}]}, "else": {"type": "null"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate({"id": 1})
    assert result == {"id": 1}
    result = field.validate(None)
    assert result == None

def test_if_then_else_from_json_schema_with_any_of_in_else():
    definitions = Definitions()
    data = {"if": {"type": "number"}, "then": {"type": "number"}, "else": {"anyOf": [{"type": "string"}, {"type": "boolean"}]}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate(3.14)
    assert result == 3.14
    result = field.validate("text")
    assert result == "text"
    result = field.validate(False)
    assert result == False

def test_if_then_else_from_json_schema_with_one_of_in_if():
    definitions = Definitions()
    data = {"if": {"oneOf": [{"type": "string"}, {"type": "number"}]}, "then": {"type": "string"}, "else": {"type": "boolean"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("test")
    assert result == "test"
    result = field.validate(42)
    assert result == 42
    result = field.validate(True)
    assert result == True

def test_if_then_else_from_json_schema_with_not_in_then():
    definitions = Definitions()
    data = {"if": {"type": "array"}, "then": {"not": {"type": "string"}}, "else": {"type": "string"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate([1, 2])
    assert result == [1, 2]
    result = field.validate("allowed")
    assert result == "allowed"

def test_if_then_else_from_json_schema_default_handling():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"type": "string"}, "else": {"type": "integer"}, "default": "default_value"}
    field = if_then_else_from_json_schema(data, definitions)
    assert field.has_default()
    assert field.get_default_value() == "default_value"

def test_if_then_else_from_json_schema_complex_nesting():
    definitions = Definitions()
    data = {"if": {"type": "object", "properties": {"active": {"type": "boolean"}}}, "then": {"type": "object", "required": ["active"]}, "else": {"type": "null"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate({"active": True})
    assert result == {"active": True}
    result = field.validate(None)
    assert result == None

def test_if_then_else_from_json_schema_empty_then_and_else():
    definitions = Definitions()
    data = {"if": {"type": "string"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("anything")
    assert result == "anything"
    result = field.validate(123)
    assert result == 123

def test_if_then_else_from_json_schema_with_enum_in_if():
    definitions = Definitions()
    data = {"if": {"enum": ["yes", "no"]}, "then": {"type": "string"}, "else": {"type": "integer"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("yes")
    assert result == "yes"
    result = field.validate("no")
    assert result == "no"
    result = field.validate(0)
    assert result == 0


# LLM-generated content at query #8
#--------------------------

```python
def test_from_json_schema_type_string_with_min_length_0_sets_allow_blank_true():
    from typesystem.json_schema import from_json_schema_type
    from typesystem.schemas import Definitions
    data = {"minLength": 0}
    definitions = Definitions()
    field = from_json_schema_type(data, "string", False, definitions)
    result = field.allow_blank
    assert result == True


# LLM-generated content at query #9
#--------------------------

def test_property_names_is_none():
    data = {}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", False, definitions)
    assert field.property_names is None


# LLM-generated content at query #10
#--------------------------

```python
def test_from_json_schema_with_ref_returns_ref_field():
    data = {"$ref": "#/components/schemas/User"}
    definitions = Definitions()
    definitions["#/components/schemas/User"] = Any()
    result = from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"


# LLM-generated content at query #11
#--------------------------

def test_additional_items_is_not_bool():
    field = Array(additional_items=String())
    result = to_json_schema(field)
    assert isinstance(result["additionalItems"], dict)


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_160_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=String(), else_clause=None)
    result = to_json_schema(field)
    assert "else" not in result


# LLM-generated content at query #13
#--------------------------

def test_boolean_field_with_null_allowed():
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

def test_boolean_field_without_null():
    field = Boolean(allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "boolean"


# LLM-generated content at query #14
#--------------------------

def test_additional_properties_none_returns_none():
    from typesystem.json_schema import from_json_schema_type
    from typesystem.schemas import Definitions
    data = {"additionalProperties": None}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    additional_properties_argument = result.additional_properties
    assert additional_properties_argument is None


# LLM-generated content at query #15
#--------------------------

def test_to_json_schema_with_definitions_loops_through_items():
    definitions = {}
    arg = Definitions({"key1": String(), "key2": Integer()})
    result = to_json_schema(arg, _definitions=definitions)
    assert "key1" in definitions
    assert "key2" in definitions
    assert definitions["key1"]["type"] == "string"
    assert definitions["key2"]["type"] == "integer"


# LLM-generated content at query #16
#--------------------------

```python
def test_from_json_schema_with_ref():
    data = {"$ref": "#/components/schemas/User"}
    definitions = Definitions()
    definitions["#/components/schemas/User"] = Integer()
    result = from_json_schema(data, definitions=definitions)
    assert isinstance(result, Integer)


# LLM-generated content at query #17
#--------------------------

def test_from_json_schema_object_with_properties():
    from typesystem.json_schema import from_json_schema
    from typesystem.fields import Object
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    field = from_json_schema(data, definitions={})
    assert isinstance(field, Object)
    assert field.properties is not None
    assert "name" in field.properties


# LLM-generated content at query #18
#--------------------------

def test_object_field_with_allow_null_true():
    obj = Object(allow_null=True)
    result = to_json_schema(obj)
    assert result["type"] == ["object", "null"]


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_71_evaluates_to_true():
    field = Array(items=[String(), Integer()])
    result = to_json_schema(field)
    assert isinstance(result.get("items"), list)
    field = Array(items=String())
    result = to_json_schema(field)
    assert not isinstance(result.get("items"), list)


# LLM-generated content at query #20
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"minimum": 1.0, "maximum": 10.0, "exclusiveMinimum": 0.5, "exclusiveMaximum": 10.5, "multipleOf": 0.5, "default": 5.0}
    definitions = Definitions()
    result = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(result, Float)
    assert result.minimum == 1.0
    assert result.maximum == 10.0
    assert result.exclusive_minimum == 0.5
    assert result.exclusive_maximum == 10.5
    assert result.multiple_of == 0.5
    assert result.default == 5.0
    assert result.allow_null == False
    assert result.coerce_types == False

def test_from_json_schema_type_integer():
    data = {"minimum": 1, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 11, "multipleOf": 2, "default": 6}
    definitions = Definitions()
    result = from_json_schema_type(data, "integer", True, definitions)
    assert isinstance(result, Integer)
    assert result.minimum == 1
    assert result.maximum == 10
    assert result.exclusive_minimum == 0
    assert result.exclusive_maximum == 11
    assert result.multiple_of == 2
    assert result.default == 6
    assert result.allow_null == True
    assert result.coerce_types == False

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 20, "format": "email", "pattern": "^[a-z]+$", "default": "hello"}
    definitions = Definitions()
    result = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 20
    assert result.format == "email"
    assert result.pattern == "^[a-z]+$"
    assert result.default == "hello"
    assert result.allow_null == False
    assert result.coerce_types == False
    assert result.allow_blank == False

def test_from_json_schema_type_boolean():
    data = {"default": True}
    definitions = Definitions()
    result = from_json_schema_type(data, "boolean", True, definitions)
    assert isinstance(result, Boolean)
    assert result.default == True
    assert result.allow_null == True
    assert result.coerce_types == False

def test_from_json_schema_type_array_no_items():
    data = {"minItems": 2, "maxItems": 5, "uniqueItems": True, "default": []}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert result.min_items == 2
    assert result.max_items == 5
    assert result.unique_items == True
    assert result.default == []
    assert result.allow_null == False
    assert result.items == None
    assert result.additional_items == True

def test_from_json_schema_type_array_with_single_item():
    data = {"items": {"type": "string"}, "additionalItems": False, "minItems": 1, "maxItems": 3}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", True, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, Field)
    assert result.additional_items == False
    assert result.min_items == 1
    assert result.max_items == 3
    assert result.allow_null == True

def test_from_json_schema_type_array_with_list_items():
    data = {"items": [{"type": "string"}, {"type": "integer"}], "additionalItems": {"type": "boolean"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, list)
    assert len(result.items) == 2
    assert isinstance(result.items[0], Field)
    assert isinstance(result.items[1], Field)
    assert isinstance(result.additional_items, Field)

def test_from_json_schema_type_object_no_properties():
    data = {"minProperties": 1, "maxProperties": 5, "required": ["id"], "default": {}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert result.min_properties == 1
    assert result.max_properties == 5
    assert result.required == ["id"]
    assert result.default == {}
    assert result.allow_null == False
    assert result.properties == {}
    assert result.additional_properties == None

def test_from_json_schema_type_object_with_properties():
    data = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "additionalProperties": False, "patternProperties": {"^test_": {"type": "boolean"}}, "propertyNames": {"pattern": "^[a-z]+$"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", True, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.properties, dict)
    assert len(result.properties) == 2
    assert "name" in result.properties
    assert "age" in result.properties
    assert result.additional_properties == False
    assert isinstance(result.pattern_properties, dict)
    assert "^test_" in result.pattern_properties
    assert isinstance(result.property_names, Field)
    assert result.allow_null == True

def test_from_json_schema_type_object_with_additional_properties_field():
    data = {"additionalProperties": {"type": "number"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.additional_properties, Field)

def test_from_json_schema_type_invalid_type_string():
    data = {}
    definitions = Definitions()
    try:
        from_json_schema_type(data, "invalid", False, definitions)
        assert False
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid'"


# LLM-generated content at query #21
#--------------------------

def test_array_field_with_allow_null_true_has_type_array_null():
    field = Array(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["array", "null"]


# LLM-generated content at query #22
#--------------------------

def test_predicate_at_line_46_evaluates_to_true_for_integer_without_null():
    field = Integer(allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "integer"

def test_predicate_at_line_46_evaluates_to_true_for_float_without_null():
    field = Float(allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "number"

def test_predicate_at_line_46_evaluates_to_true_for_decimal_without_null():
    field = Decimal(allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "number"

def test_predicate_at_line_46_evaluates_to_true_for_integer_with_null():
    field = Integer(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["integer", "null"]

def test_predicate_at_line_46_evaluates_to_true_for_float_with_null():
    field = Float(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]

def test_predicate_at_line_46_evaluates_to_true_for_decimal_with_null():
    field = Decimal(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]


# LLM-generated content at query #23
#--------------------------

def test_predicate_at_line_120_evaluates_to_true():
    from your_module import Schema, to_json_schema
    schema_field = Schema(allow_null=True)
    result = to_json_schema(schema_field)
    assert result["type"] == ["object", "null"]
    schema_field_not_null = Schema(allow_null=False)
    result_not_null = to_json_schema(schema_field_not_null)
    assert result_not_null["type"] == "object"


# LLM-generated content at query #24
#--------------------------

def test_to_json_schema_with_union_field():
    from typesystem.fields import String, Integer, Union
    from typesystem.json_schema import to_json_schema
    field = String() | Integer()
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_union_field_allow_null():
    from typesystem.fields import String, Integer, Union
    from typesystem.json_schema import to_json_schema
    field = String(allow_null=True) | Integer()
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": ["string", "null"]}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_union_field_and_default():
    from typesystem.fields import String, Integer, Union
    from typesystem.json_schema import to_json_schema
    field = String(default="hello") | Integer(default=42)
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string", "default": "hello"}, {"type": "integer", "default": 42}]}
    assert result == expected

def test_to_json_schema_with_nested_union():
    from typesystem.fields import String, Integer, Boolean, Union
    from typesystem.json_schema import to_json_schema
    inner_union = String() | Integer()
    field = inner_union | Boolean()
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}]}
    assert result == expected

def test_to_json_schema_with_definitions_and_union():
    from typesystem.fields import String, Integer, Union
    from typesystem.schemas import Definitions
    from typesystem.json_schema import to_json_schema
    definitions = Definitions({"MyUnion": String() | Integer()})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"MyUnion": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}}
    assert result == expected

def test_to_json_schema_with_reference_in_union():
    from typesystem.fields import String, Reference, Union
    from typesystem.schemas import Definitions
    from typesystem.json_schema import to_json_schema
    definitions = Definitions({"MyString": String()})
    field = Reference(to="MyString", target=String()) | String()
    result = to_json_schema(field, _definitions=definitions)
    expected = {"anyOf": [{"$ref": "#/components/schemas/MyString"}, {"type": "string"}]}
    assert result == expected

def test_to_json_schema_union_with_complex_fields():
    from typesystem.fields import String, Array, Integer, Union
    from typesystem.json_schema import to_json_schema
    array_field = Array(items=String())
    field = array_field | Integer()
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_union_with_object_field():
    from typesystem.fields import String, Object, Integer, Union
    from typesystem.json_schema import to_json_schema
    object_field = Object(properties={"name": String()})
    field = object_field | Integer()
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "object", "properties": {"name": {"type": "string"}}}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_union_with_choice_field():
    from typesystem.fields import String, Choice, Integer, Union
    from typesystem.json_schema import to_json_schema
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    field = choice_field | Integer()
    result = to_json_schema(field)
    expected = {"anyOf": [{"enum": ["a", "b"]}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_union_with_const_field():
    from typesystem.fields import String, Const, Integer, Union
    from typesystem.json_schema import to_json_schema
    const_field = Const(const="fixed")
    field = const_field | Integer()
    result = to_json_schema(field)
    expected = {"anyOf": [{"const": "fixed"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_union_with_allof_field():
    from typesystem.fields import String, AllOf, Integer, Union
    from typesystem.json_schema import to_json_schema
    allof_field = AllOf(all_of=[String(), String(max_length=10)])
    field = allof_field | Integer()
    result = to_json_schema(field)
    expected = {"anyOf": [{"allOf": [{"type": "string"}, {"type": "string", "maxLength": 10}]}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_union_with_oneof_field():
    from typesystem.fields import String, OneOf, Integer, Union
    from typesystem.json_schema import to_json_schema
    oneof_field = OneOf(one_of=[String(), Integer()])
    field = oneof_field | String()
    result = to_json_schema(field)
    expected = {"anyOf": [{"oneOf": [{"type": "string"}, {"type": "integer"}]}, {"type": "string"}]}
    assert result == expected

def test_to_json_schema_union_with_ifthenelse_field():
    from typesystem.fields import String, IfThenElse, Integer, Union
    from typesystem.json_schema import to_json_schema
    ifthenelse_field = IfThenElse(if_clause=String(), then_clause=Integer())
    field = ifthenelse_field | String()
    result = to_json_schema(field)
    expected = {"anyOf": [{"if": {"type": "string"}, "then": {"type": "integer"}}, {"type": "string"}]}
    assert result == expected

def test_to_json_schema_union_with_not_field():
    from typesystem.fields import String, Not, Integer, Union
    from typesystem.json_schema import to_json_schema
    not_field = Not(negated=String())
    field = not_field | Integer()
    result = to_json_schema(field)
    expected = {"anyOf": [{"not": {"type": "string"}}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_union_with_schema_field():
    from typesystem.fields import String, Schema, Integer, Union
    from typesystem.json_schema import to_json_schema
    schema_field = Schema(fields={"name": String()})
    field = schema_field | Integer()
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "object", "properties": {"name": {"type": "string"}}}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_union_with_any_field():
    from typesystem.fields import Any, String, Union
    from typesystem.json_schema import to_json_schema
    field = Any() | String()
    result = to_json_schema(field)
    expected = {"anyOf": [True, {"type": "string"}]}
    assert result == expected

def test_to_json_schema_union_with_nevermatch_field():
    from typesystem.fields import NeverMatch, String, Union
    from typesystem.json_schema import to_json_schema
    field = NeverMatch() | String()
    result = to_json_schema(field)
    expected = {"anyOf": [False, {"type": "string"}]}
    assert result == expected

def test_to_json_schema_union_with_multiple_nullable_children():
    from typesystem.fields import String, Integer, Union
    from typesystem.json_schema import to_json_schema
    field = String(allow_null=True) | Integer(allow_null=True)
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": ["string", "null"]}, {"type": ["integer", "null"]}]}
    assert result == expected

def test_to_json_schema_union_with_mixed_nullable_and_nonnullable():
    from typesystem.fields import String, Integer, Boolean, Union
    from typesystem.json_schema import to_json_schema
    field = String(allow_null=True) | Integer() | Boolean(allow_null=True)
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": ["string", "null"]}, {"type": "integer"}, {"type": ["boolean", "null"]}]}
    assert result == expected


# LLM-generated content at query #25
#--------------------------

def test_to_json_schema_returns_false_for_nevermatch():
    result = to_json_schema(NeverMatch())
    assert result is False


# LLM-generated content at query #26
#--------------------------

```python
def test_from_json_schema_type_string_with_min_length_0():
    from typesystem.json_schema import from_json_schema_type
    from typesystem.fields import String
    data = {"minLength": 0}
    field = from_json_schema_type(data, "string", False, {})
    assert isinstance(field, String)
    assert field.allow_blank == True
    assert field.min_length == None


# LLM-generated content at query #27
#--------------------------

def test_string_field_with_format_includes_format_in_json_schema():
    field = String(format="email")
    result = to_json_schema(field)
    assert result["format"] == "email"


# LLM-generated content at query #28
#--------------------------

def test_predicate_at_line_158_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=String())
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #29
#--------------------------

def test_to_json_schema_root_with_definitions():
    field = Reference(to="TestRef", target=String())
    result = to_json_schema(field)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "TestRef" in result["components"]["schemas"]


# LLM-generated content at query #30
#--------------------------

def test_to_json_schema_with_union_field():
    from typesystem.fields import Union, Integer, String
    from typesystem.json_schema import to_json_schema
    union_field = Union(any_of=[Integer(), String()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "integer"
    assert result["anyOf"][1]["type"] == "string"

def test_to_json_schema_with_union_field_and_null():
    from typesystem.fields import Union, Integer, String
    from typesystem.json_schema import to_json_schema
    union_field = Union(any_of=[Integer(allow_null=True), String()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == ["integer", "null"]
    assert result["anyOf"][1]["type"] == "string"

def test_to_json_schema_with_union_field_and_default():
    from typesystem.fields import Union, Integer, String
    from typesystem.json_schema import to_json_schema
    union_field = Union(any_of=[Integer(default=42), String()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert "default" in result
    assert result["default"] == 42

def test_to_json_schema_with_union_field_and_definitions():
    from typesystem.fields import Union, Integer, String, Reference
    from typesystem.schemas import Definitions
    from typesystem.json_schema import to_json_schema
    definitions = Definitions({"MyRef": Reference(to="MyRef", target=Integer())})
    union_field = Union(any_of=[Reference(to="MyRef", target=Integer()), String()])
    result = to_json_schema(union_field, _definitions={})
    assert "anyOf" in result
    assert result["anyOf"][0]["$ref"] == "#/components/schemas/MyRef"
    assert result["anyOf"][1]["type"] == "string"

def test_to_json_schema_with_union_field_root_definitions():
    from typesystem.fields import Union, Integer, String, Reference
    from typesystem.schemas import Definitions
    from typesystem.json_schema import to_json_schema
    definitions = Definitions({"MyRef": Reference(to="MyRef", target=Integer())})
    union_field = Union(any_of=[Reference(to="MyRef", target=Integer()), String()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert result["anyOf"][0]["$ref"] == "#/components/schemas/MyRef"
    assert result["anyOf"][1]["type"] == "string"
    assert "components" in result
    assert "schemas" in result["components"]
    assert "MyRef" in result["components"]["schemas"]

def test_to_json_schema_with_union_field_and_multiple_types():
    from typesystem.fields import Union, Integer, String, Boolean
    from typesystem.json_schema import to_json_schema
    union_field = Union(any_of=[Integer(), String(), Boolean()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 3
    assert result["anyOf"][0]["type"] == "integer"
    assert result["anyOf"][1]["type"] == "string"
    assert result["anyOf"][2]["type"] == "boolean"

def test_to_json_schema_with_union_field_and_nested_union():
    from typesystem.fields import Union, Integer, String
    from typesystem.json_schema import to_json_schema
    inner_union = Union(any_of=[Integer(), String()])
    outer_union = Union(any_of=[inner_union, String()])
    result = to_json_schema(outer_union)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert "anyOf" in result["anyOf"][0]
    assert result["anyOf"][1]["type"] == "string"

def test_to_json_schema_with_union_field_and_allow_null_on_all():
    from typesystem.fields import Union, Integer, String
    from typesystem.json_schema import to_json_schema
    union_field = Union(any_of=[Integer(allow_null=True), String(allow_null=True)])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == ["integer", "null"]
    assert result["anyOf"][1]["type"] == ["string", "null"]

def test_to_json_schema_with_union_field_and_no_children():
    from typesystem.fields import Union
    from typesystem.json_schema import to_json_schema
    union_field = Union(any_of=[])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert result["anyOf"] == []

def test_to_json_schema_with_union_field_and_complex_nesting():
    from typesystem.fields import Union, Integer, String, Array
    from typesystem.json_schema import to_json_schema
    array_field = Array(items=Integer())
    union_field = Union(any_of=[array_field, String()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "array"
    assert "items" in result["anyOf"][0]
    assert result["anyOf"][0]["items"]["type"] == "integer"
    assert result["anyOf"][1]["type"] == "string"


# LLM-generated content at query #31
#--------------------------

def test_pattern_properties_evaluates_to_true():
    field = Object(pattern_properties={"^test.*": String()})
    result = to_json_schema(field)
    assert "patternProperties" in result
    assert "^test.*" in result["patternProperties"]
    assert result["patternProperties"]["^test.*"]["type"] == "string"


# LLM-generated content at query #32
#--------------------------

def test_from_json_schema_object_without_pattern_properties():
    from typesystem.json_schema import from_json_schema_type
    from typesystem.fields import Object
    data = {"type": "object"}
    field = from_json_schema_type(data, "object", False, {})
    assert isinstance(field, Object)
    assert field.pattern_properties is None


# LLM-generated content at query #33
#--------------------------

```python
def test_from_json_schema_boolean_true():
    result = from_json_schema(True)
    assert isinstance(result, Any)

def test_from_json_schema_boolean_false():
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

def test_from_json_schema_ref():
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = Integer()
    data = {"$ref": "#/components/schemas/Test"}
    result = from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/Test"

def test_from_json_schema_type_string():
    data = {"type": "string"}
    result = from_json_schema(data)
    assert isinstance(result, String)

def test_from_json_schema_type_integer():
    data = {"type": "integer"}
    result = from_json_schema(data)
    assert isinstance(result, Integer)

def test_from_json_schema_type_number():
    data = {"type": "number"}
    result = from_json_schema(data)
    assert isinstance(result, Float)

def test_from_json_schema_type_boolean():
    data = {"type": "boolean"}
    result = from_json_schema(data)
    assert isinstance(result, Boolean)

def test_from_json_schema_type_array():
    data = {"type": "array"}
    result = from_json_schema(data)
    assert isinstance(result, Array)

def test_from_json_schema_type_object():
    data = {"type": "object"}
    result = from_json_schema(data)
    assert isinstance(result, Object)

def test_from_json_schema_nullable_type():
    data = {"type": ["string", "null"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert result.allow_null == True

def test_from_json_schema_multiple_types():
    data = {"type": ["string", "integer"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

def test_from_json_schema_enum():
    data = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)
    assert len(result.choices) == 3

def test_from_json_schema_const():
    data = {"const": 42}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == 42

def test_from_json_schema_allOf():
    data = {"allOf": [{"type": "string"}, {"minLength": 1}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_from_json_schema_anyOf():
    data = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

def test_from_json_schema_oneOf():
    data = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2

def test_from_json_schema_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)

def test_from_json_schema_if_then_else():
    data = {"if": {"type": "string"}, "then": {"minLength": 1}, "else": {"type": "integer"}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_multiple_constraints():
    data = {"type": "string", "enum": ["a", "b"], "minLength": 1}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 3

def test_from_json_schema_no_constraints():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_with_components():
    data = {
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_null_type_only():
    data = {"type": "null"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const is None

def test_from_json_schema_empty_type_array():
    data = {"type": []}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const is None


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_to_json_schema_with_union_field():
    from typesystem.fields import Field, Union
    from typesystem.json_schema import to_json_schema
    class String(Field):
        errors = {}
        def __init__(self, allow_null=False, **kwargs):
            super().__init__(allow_null=allow_null, **kwargs)
        def validate(self, value):
            return value
    class Integer(Field):
        errors = {}
        def __init__(self, allow_null=False, **kwargs):
            super().__init__(allow_null=allow_null, **kwargs)
        def validate(self, value):
            return value
    string_field = String(allow_null=True)
    integer_field = Integer()
    union_field = Union(any_of=[string_field, integer_field])
    result = to_json_schema(union_field)
    expected = {
        "anyOf": [
            {"type": ["string", "null"]},
            {"type": "integer"}
        ]
    }
    assert result == expected

def test_to_json_schema_with_definitions():
    from typesystem.schemas import Definitions
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    class String(Field):
        errors = {}
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def validate(self, value):
            return value
    definitions = Definitions({"MyString": String()})
    result = to_json_schema(definitions)
    expected = {
        "components": {
            "schemas": {
                "MyString": {"type": "string"}
            }
        }
    }
    assert result == expected

def test_to_json_schema_with_field_default():
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    class String(Field):
        errors = {}
        def __init__(self, default=None, **kwargs):
            super().__init__(default=default, **kwargs)
        def validate(self, value):
            return value
    field = String(default="hello")
    result = to_json_schema(field)
    expected = {"type": "string", "default": "hello"}
    assert result == expected

def test_to_json_schema_with_allow_null():
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    class String(Field):
        errors = {}
        def __init__(self, allow_null=False, **kwargs):
            super().__init__(allow_null=allow_null, **kwargs)
        def validate(self, value):
            return value
    field = String(allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["string", "null"]}
    assert result == expected

def test_to_json_schema_with_integer_constraints():
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    class Integer(Field):
        errors = {}
        def __init__(self, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None, **kwargs):
            super().__init__(**kwargs)
            self.minimum = minimum
            self.maximum = maximum
            self.exclusive_minimum = exclusive_minimum
            self.exclusive_maximum = exclusive_maximum
            self.multiple_of = multiple_of
        def validate(self, value):
            return value
    field = Integer(minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100, multiple_of=5)
    result = to_json_schema(field)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 5
    }
    assert result == expected

def test_to_json_schema_with_array_field():
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    class Array(Field):
        errors = {}
        def __init__(self, items=None, min_items=None, max_items=None, unique_items=False, **kwargs):
            super().__init__(**kwargs)
            self.items = items
            self.min_items = min_items
            self.max_items = max_items
            self.unique_items = unique_items
            self.additional_items = None
        def validate(self, value):
            return value
    class String(Field):
        errors = {}
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def validate(self, value):
            return value
    string_field = String()
    array_field = Array(items=string_field, min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True
    }
    assert result == expected

def test_to_json_schema_with_object_field():
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    class Object(Field):
        errors = {}
        def __init__(self, properties=None, required=None, **kwargs):
            super().__init__(**kwargs)
            self.properties = properties or {}
            self.required = required or []
            self.pattern_properties = None
            self.additional_properties = None
            self.property_names = None
            self.max_properties = None
            self.min_properties = None
        def validate(self, value):
            return value
    class String(Field):
        errors = {}
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def validate(self, value):
            return value
    class Integer(Field):
        errors = {}
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def validate(self, value):
            return value
    properties = {"name": String(), "age": Integer()}
    required = ["name"]
    object_field = Object(properties=properties, required=required)
    result = to_json_schema(object_field)
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    assert result == expected

def test_to_json_schema_with_choice_field():
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    class Choice(Field):
        errors = {}
        def __init__(self, choices, **kwargs):
            super().__init__(**kwargs)
            self.choices = choices
        def validate(self, value):
            return value
    choices = [("a", "A"), ("b", "B")]
    choice_field = Choice(choices=choices)
    result = to_json_schema(choice_field)
    expected = {"enum": ["a", "b"]}
    assert result == expected

def test_to_json_schema_with_const_field():
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    class Const(Field):
        errors = {}
        def __init__(self, const, **kwargs):
            super().__init__(**kwargs)
            self.const = const
        def validate(self, value):
            return value
    const_field = Const(const=42)
    result = to_json_schema(const_field)
    expected = {"const": 42}
    assert result == expected

def test_to_json_schema_with_unknown_field_type():
    from typesystem.fields import Field
    from typesystem.json_schema import to_json_schema
    class UnknownField(Field):
        errors = {}
        def validate(self, value):
            return value
    field = UnknownField()
    try:
        to_json_schema(field)
        assert False
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


# LLM-generated content at query #2
#--------------------------

def test_to_json_schema_with_string_field():
    field = String()
    result = to_json_schema(field)
    expected = {"type": "string"}
    assert result == expected

def test_to_json_schema_with_string_field_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["string", "null"]}
    assert result == expected

def test_to_json_schema_with_string_field_default():
    field = String(default="hello")
    result = to_json_schema(field)
    expected = {"type": "string", "default": "hello"}
    assert result == expected

def test_to_json_schema_with_integer_field():
    field = Integer()
    result = to_json_schema(field)
    expected = {"type": "integer"}
    assert result == expected

def test_to_json_schema_with_integer_field_allow_null():
    field = Integer(allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["integer", "null"]}
    assert result == expected

def test_to_json_schema_with_integer_field_minimum_maximum():
    field = Integer(minimum=0, maximum=100)
    result = to_json_schema(field)
    expected = {"type": "integer", "minimum": 0, "maximum": 100}
    assert result == expected

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    expected = {"type": "boolean"}
    assert result == expected

def test_to_json_schema_with_boolean_field_default():
    field = Boolean(default=True)
    result = to_json_schema(field)
    expected = {"type": "boolean", "default": True}
    assert result == expected

def test_to_json_schema_with_array_field():
    field = Array(items=String())
    result = to_json_schema(field)
    expected = {"type": "array", "items": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_array_field_min_max_items():
    field = Array(items=Integer(), min_items=1, max_items=10)
    result = to_json_schema(field)
    expected = {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 10}
    assert result == expected

def test_to_json_schema_with_object_field():
    field = Object(properties={"name": String()})
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert result == expected

def test_to_json_schema_with_object_field_required():
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}
    assert result == expected

def test_to_json_schema_with_union_field():
    field = String() | Integer()
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    expected = {"enum": ["a", "b"]}
    assert result == expected

def test_to_json_schema_with_const_field():
    field = Const(const=42)
    result = to_json_schema(field)
    expected = {"const": 42}
    assert result == expected

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"title": String()})
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"title": {"type": "string"}}}
    assert result == expected

def test_to_json_schema_with_definitions():
    definitions = Definitions({"User": Schema(fields={"name": String()})})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}}}}
    assert result == expected

def test_to_json_schema_with_reference_field():
    user_schema = Schema(fields={"name": String()})
    field = Reference(to="User", target=user_schema)
    result = to_json_schema(field)
    expected = {"$ref": "#/components/schemas/User"}
    assert result == expected

def test_to_json_schema_with_any_field():
    field = Any()
    result = to_json_schema(field)
    assert result == True

def test_to_json_schema_with_never_match_field():
    field = NeverMatch()
    result = to_json_schema(field)
    assert result == False

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_all_of_field():
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string", "minLength": 1}, {"type": "string", "maxLength": 10}]}
    assert result == expected

def test_to_json_schema_with_if_then_else_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(field)
    expected = {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    assert result == expected

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    result = to_json_schema(field)
    expected = {"not": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_array_field_unique_items():
    field = Array(items=String(), unique_items=True)
    result = to_json_schema(field)
    expected = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
    assert result == expected

def test_to_json_schema_with_object_field_additional_properties_false():
    field = Object(properties={"name": String()}, additional_properties=False)
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False}
    assert result == expected

def test_to_json_schema_with_object_field_additional_properties_field():
    field = Object(properties={"name": String()}, additional_properties=Integer())
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": {"type": "integer"}}
    assert result == expected

def test_to_json_schema_with_string_field_min_length():
    field = String(min_length=5)
    result = to_json_schema(field)
    expected = {"type": "string", "minLength": 5}
    assert result == expected

def test_to_json_schema_with_string_field_max_length():
    field = String(max_length=10)
    result = to_json_schema(field)
    expected = {"type": "string", "maxLength": 10}
    assert result == expected

def test_to_json_schema_with_string_field_pattern():
    field = String(pattern=r"^\d+$")
    result = to_json_schema(field)
    expected = {"type": "string", "pattern": r"^\d+$"}
    assert result == expected

def test_to_json_schema_with_string_field_format():
    field = String(format="email")
    result = to_json_schema(field)
    expected = {"type": "string", "format": "email"}
    assert result == expected

def test_to_json_schema_with_float_field():
    field = Float()
    result = to_json_schema(field)
    expected = {"type": "number"}
    assert result == expected

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    result = to_json_schema(field)
    expected = {"type": "number"}
    assert result == expected

def test_to_json_schema_with_number_field_multiple_of():
    field = Integer(multiple_of=5)
    result = to_json_schema(field)
    expected = {"type": "integer", "multipleOf": 5}
    assert result == expected

def test_to_json_schema_with_number_field_exclusive_minimum_maximum():
    field = Integer(exclusive_minimum=0, exclusive_maximum=100)
    result = to_json_schema(field)
    expected = {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100}
    assert result == expected

def test_to_json_schema_with_array_field_additional_items_false():
   


# LLM-generated content at query #3
#--------------------------

def test_from_json_schema_boolean_true():
    result = from_json_schema(True)
    assert isinstance(result, Any)

def test_from_json_schema_boolean_false():
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

def test_from_json_schema_ref():
    data = {"$ref": "#/components/schemas/Test"}
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = Const("test")
    result = from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/Test"

def test_from_json_schema_type_string():
    data = {"type": "string"}
    result = from_json_schema(data)
    assert isinstance(result, Field)

def test_from_json_schema_type_array():
    data = {"type": ["string", "number"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)

def test_from_json_schema_enum():
    data = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)

def test_from_json_schema_const():
    data = {"const": 42}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == 42

def test_from_json_schema_allOf():
    data = {"allOf": [{"type": "string"}, {"maxLength": 10}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_anyOf():
    data = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)

def test_from_json_schema_oneOf():
    data = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)

def test_from_json_schema_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)

def test_from_json_schema_if_then_else():
    data = {"if": {"type": "string"}, "then": {"maxLength": 5}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_multiple_constraints():
    data = {"type": "string", "enum": ["a", "b"]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_no_constraints():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_with_null_type():
    data = {"type": ["string", "null"]}
    result = from_json_schema(data)
    assert isinstance(result, Field)
    assert result.allow_null == True

def test_from_json_schema_with_components():
    data = {"components": {"schemas": {"Test": {"type": "string"}}}}
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_type_empty_with_null():
    data = {"type": []}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const is None

def test_from_json_schema_type_empty_without_null():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)


# LLM-generated content at query #4
#--------------------------

def test_pattern_regex_flags_unicode():
    import re
    from module import String, to_json_schema
    pattern = re.compile(r"test", flags=re.RegexFlag.UNICODE)
    field = String(pattern_regex=pattern)
    result = to_json_schema(field)
    assert "pattern" in result
    assert result["pattern"] == "test"


# LLM-generated content at query #5
#--------------------------

def test_to_json_schema_with_string_field():
    field = String()
    result = to_json_schema(field)
    expected = {"type": "string"}
    assert result == expected

def test_to_json_schema_with_string_field_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["string", "null"]}
    assert result == expected

def test_to_json_schema_with_string_field_min_length():
    field = String(min_length=5)
    result = to_json_schema(field)
    expected = {"type": "string", "minLength": 5}
    assert result == expected

def test_to_json_schema_with_string_field_max_length():
    field = String(max_length=10)
    result = to_json_schema(field)
    expected = {"type": "string", "maxLength": 10}
    assert result == expected

def test_to_json_schema_with_string_field_pattern():
    field = String(pattern=r"^\d+$")
    result = to_json_schema(field)
    expected = {"type": "string", "pattern": r"^\d+$"}
    assert result == expected

def test_to_json_schema_with_string_field_format():
    field = String(format="email")
    result = to_json_schema(field)
    expected = {"type": "string", "format": "email"}
    assert result == expected

def test_to_json_schema_with_integer_field():
    field = Integer()
    result = to_json_schema(field)
    expected = {"type": "integer"}
    assert result == expected

def test_to_json_schema_with_integer_field_allow_null():
    field = Integer(allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["integer", "null"]}
    assert result == expected

def test_to_json_schema_with_integer_field_minimum():
    field = Integer(minimum=0)
    result = to_json_schema(field)
    expected = {"type": "integer", "minimum": 0}
    assert result == expected

def test_to_json_schema_with_integer_field_maximum():
    field = Integer(maximum=100)
    result = to_json_schema(field)
    expected = {"type": "integer", "maximum": 100}
    assert result == expected

def test_to_json_schema_with_integer_field_exclusive_minimum():
    field = Integer(exclusive_minimum=0)
    result = to_json_schema(field)
    expected = {"type": "integer", "exclusiveMinimum": 0}
    assert result == expected

def test_to_json_schema_with_integer_field_exclusive_maximum():
    field = Integer(exclusive_maximum=100)
    result = to_json_schema(field)
    expected = {"type": "integer", "exclusiveMaximum": 100}
    assert result == expected

def test_to_json_schema_with_integer_field_multiple_of():
    field = Integer(multiple_of=5)
    result = to_json_schema(field)
    expected = {"type": "integer", "multipleOf": 5}
    assert result == expected

def test_to_json_schema_with_float_field():
    field = Float()
    result = to_json_schema(field)
    expected = {"type": "number"}
    assert result == expected

def test_to_json_schema_with_float_field_allow_null():
    field = Float(allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["number", "null"]}
    assert result == expected

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    expected = {"type": "boolean"}
    assert result == expected

def test_to_json_schema_with_boolean_field_allow_null():
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["boolean", "null"]}
    assert result == expected

def test_to_json_schema_with_array_field():
    field = Array()
    result = to_json_schema(field)
    expected = {"type": "array"}
    assert result == expected

def test_to_json_schema_with_array_field_allow_null():
    field = Array(allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["array", "null"]}
    assert result == expected

def test_to_json_schema_with_array_field_min_items():
    field = Array(min_items=1)
    result = to_json_schema(field)
    expected = {"type": "array", "minItems": 1}
    assert result == expected

def test_to_json_schema_with_array_field_max_items():
    field = Array(max_items=10)
    result = to_json_schema(field)
    expected = {"type": "array", "maxItems": 10}
    assert result == expected

def test_to_json_schema_with_array_field_items():
    field = Array(items=String())
    result = to_json_schema(field)
    expected = {"type": "array", "items": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_array_field_items_list():
    field = Array(items=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_array_field_additional_items_bool():
    field = Array(additional_items=False)
    result = to_json_schema(field)
    expected = {"type": "array", "additionalItems": False}
    assert result == expected

def test_to_json_schema_with_array_field_additional_items_field():
    field = Array(additional_items=String())
    result = to_json_schema(field)
    expected = {"type": "array", "additionalItems": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_array_field_unique_items():
    field = Array(unique_items=True)
    result = to_json_schema(field)
    expected = {"type": "array", "uniqueItems": True}
    assert result == expected

def test_to_json_schema_with_object_field():
    field = Object()
    result = to_json_schema(field)
    expected = {"type": "object"}
    assert result == expected

def test_to_json_schema_with_object_field_allow_null():
    field = Object(allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["object", "null"]}
    assert result == expected

def test_to_json_schema_with_object_field_properties():
    field = Object(properties={"name": String()})
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert result == expected

def test_to_json_schema_with_object_field_pattern_properties():
    field = Object(pattern_properties={r"^\d+$": Integer()})
    result = to_json_schema(field)
    expected = {"type": "object", "patternProperties": {r"^\d+$": {"type": "integer"}}}
    assert result == expected

def test_to_json_schema_with_object_field_additional_properties_bool():
    field = Object(additional_properties=False)
    result = to_json_schema(field)
    expected = {"type": "object", "additionalProperties": False}
    assert result == expected

def test_to_json_schema_with_object_field_additional_properties_field():
    field = Object(additional_properties=String())
    result = to_json_schema(field)
    expected = {"type": "object", "additionalProperties": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_object_field_property_names():
    field = Object(property_names=String(pattern=r"^[a-z]+$"))
    result = to_json_schema(field)
    expected = {"type": "object", "propertyNames": {"type": "string", "pattern": r"^[a-z]+$"}}
    assert result == expected

def test_to_json_schema_with_object_field_max_properties():
    field = Object(max_properties=5)
    result = to_json_schema(field)
    expected = {"type": "object", "maxProperties": 5}
    assert result == expected

def test_to_json_schema_with_object_field_min_properties():
    field = Object(min_properties=1)
    result = to_json_schema(field)
    expected = {"type": "object", "minProperties": 1}
    assert result == expected

def test_to_json_schema_with_object_field_required():
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}
    assert result == expected

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String()})
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert result == expected

def test_to_json_schema_with_schema_field_allow


# LLM-generated content at query #6
#--------------------------

def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2}
    definitions = Definitions()
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.allow_null == False

def test_from_json_schema_type_integer():
    data = {"minimum": 5, "maximum": 15, "exclusiveMinimum": 5, "exclusiveMaximum": 15, "multipleOf": 5}
    definitions = Definitions()
    field = from_json_schema_type(data, "integer", True, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 5
    assert field.maximum == 15
    assert field.exclusive_minimum == 5
    assert field.exclusive_maximum == 15
    assert field.multiple_of == 5
    assert field.allow_null == True

def test_from_json_schema_type_string():
    data = {"minLength": 1, "maxLength": 10, "format": "email", "pattern": "^[a-z]+$"}
    definitions = Definitions()
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-z]+$"
    assert field.allow_null == False

def test_from_json_schema_type_boolean():
    data = {}
    definitions = Definitions()
    field = from_json_schema_type(data, "boolean", True, definitions)
    assert isinstance(field, Boolean)
    assert field.allow_null == True

def test_from_json_schema_type_array():
    data = {"minItems": 1, "maxItems": 5, "uniqueItems": True, "items": {"type": "string"}, "additionalItems": False}
    definitions = Definitions()
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items == True
    assert isinstance(field.items, String)
    assert field.additional_items == False
    assert field.allow_null == False

def test_from_json_schema_type_object():
    data = {"minProperties": 1, "maxProperties": 3, "required": ["id"], "properties": {"id": {"type": "string"}}, "additionalProperties": False}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", True, definitions)
    assert isinstance(field, Object)
    assert field.min_properties == 1
    assert field.max_properties == 3
    assert field.required == ["id"]
    assert isinstance(field.properties["id"], String)
    assert field.additional_properties == False
    assert field.allow_null == True


# LLM-generated content at query #7
#--------------------------

def test_additional_properties_is_none():
    data = {}
    type_string = "object"
    allow_null = False
    definitions = {}
    field = from_json_schema_type(data, type_string, allow_null, definitions)
    assert field.additional_properties is None


# LLM-generated content at query #8
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    definitions = Definitions()
    result = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 10
    assert result.exclusive_minimum == 0
    assert result.exclusive_maximum == 10
    assert result.multiple_of == 2
    assert result.default == 5
    assert result.allow_null == False
    assert result.coerce_types == False

def test_from_json_schema_type_number_allow_null():
    data = {}
    definitions = Definitions()
    result = from_json_schema_type(data, "number", True, definitions)
    assert isinstance(result, Float)
    assert result.allow_null == True
    assert result.coerce_types == False

def test_from_json_schema_type_integer():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    definitions = Definitions()
    result = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.maximum == 10
    assert result.exclusive_minimum == 0
    assert result.exclusive_maximum == 10
    assert result.multiple_of == 2
    assert result.default == 5
    assert result.allow_null == False
    assert result.coerce_types == False

def test_from_json_schema_type_string():
    data = {"minLength": 5, "maxLength": 10, "format": "email", "pattern": "^[a-z]+$", "default": "hello"}
    definitions = Definitions()
    result = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.format == "email"
    assert result.pattern == "^[a-z]+$"
    assert result.default == "hello"
    assert result.allow_null == False
    assert result.coerce_types == False
    assert result.allow_blank == False

def test_from_json_schema_type_string_min_length_zero():
    data = {"minLength": 0}
    definitions = Definitions()
    result = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(result, String)
    assert result.min_length == None
    assert result.allow_blank == True

def test_from_json_schema_type_string_min_length_one():
    data = {"minLength": 1}
    definitions = Definitions()
    result = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(result, String)
    assert result.min_length == None
    assert result.allow_blank == False

def test_from_json_schema_type_boolean():
    data = {"default": True}
    definitions = Definitions()
    result = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(result, Boolean)
    assert result.default == True
    assert result.allow_null == False
    assert result.coerce_types == False

def test_from_json_schema_type_array_no_items():
    data = {"minItems": 1, "maxItems": 10, "uniqueItems": True, "default": []}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert result.min_items == 1
    assert result.max_items == 10
    assert result.unique_items == True
    assert result.default == []
    assert result.allow_null == False
    assert result.items == None
    assert result.additional_items == True

def test_from_json_schema_type_array_with_items_single():
    data = {"items": {"type": "string"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

def test_from_json_schema_type_array_with_items_list():
    data = {"items": [{"type": "string"}, {"type": "number"}]}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, list)
    assert len(result.items) == 2
    assert isinstance(result.items[0], String)
    assert isinstance(result.items[1], Float)

def test_from_json_schema_type_array_additional_items_bool():
    data = {"additionalItems": False}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert result.additional_items == False

def test_from_json_schema_type_array_additional_items_field():
    data = {"additionalItems": {"type": "string"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.additional_items, String)

def test_from_json_schema_type_object():
    data = {"properties": {"name": {"type": "string"}}, "patternProperties": {"^[a-z]+$": {"type": "number"}}, "additionalProperties": False, "propertyNames": {"pattern": "^[a-z]+$"}, "minProperties": 1, "maxProperties": 10, "required": ["name"], "default": {}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.properties, dict)
    assert "name" in result.properties
    assert isinstance(result.properties["name"], String)
    assert isinstance(result.pattern_properties, dict)
    assert "^[a-z]+$" in result.pattern_properties
    assert isinstance(result.pattern_properties["^[a-z]+$"], Float)
    assert result.additional_properties == False
    assert isinstance(result.property_names, String)
    assert result.property_names.pattern == "^[a-z]+$"
    assert result.min_properties == 1
    assert result.max_properties == 10
    assert result.required == ["name"]
    assert result.default == {}
    assert result.allow_null == False

def test_from_json_schema_type_object_additional_properties_field():
    data = {"additionalProperties": {"type": "string"}}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.additional_properties, String)

def test_from_json_schema_type_object_property_names_none():
    data = {}
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert result.property_names == None

def test_from_json_schema_type_invalid_type_string():
    data = {}
    definitions = Definitions()
    try:
        from_json_schema_type(data, "invalid", False, definitions)
        assert False, "Expected assertion error"
    except AssertionError as e:
        assert "Invalid argument type_string='invalid'" in str(e)


# LLM-generated content at query #9
#--------------------------

def test_ref_from_json_schema_creates_reference_with_correct_to():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, definitions)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is definitions


def test_ref_from_json_schema_raises_assertion_error_for_non_hash_ref():
    definitions = Definitions()
    data = {"$ref": "http://example.com/schema"}
    try:
        ref_from_json_schema(data, definitions)
        assert False
    except AssertionError as e:
        assert str(e) == "Unsupported $ref style in document."


# LLM-generated content at query #10
#--------------------------

def test_predicate_at_line_120_evaluates_to_true():
    schema = Schema(allow_null=True)
    result = to_json_schema(schema)
    assert result["type"] == ["object", "null"]
    schema = Schema(allow_null=False)
    result = to_json_schema(schema)
    assert result["type"] == "object"


# LLM-generated content at query #11
#--------------------------

def test_if_then_else_from_json_schema_basic():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert isinstance(result, int)
    result = field.validate(False)
    assert isinstance(result, bool)

def test_if_then_else_from_json_schema_without_then():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "else": {"type": "integer"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("test")
    assert result == "test"
    result = field.validate(5)
    assert result == 5

def test_if_then_else_from_json_schema_without_else():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"type": "integer"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert isinstance(result, int)
    result = field.validate(3.14)
    assert result == 3.14

def test_if_then_else_from_json_schema_with_default():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}, "default": 42}
    field = if_then_else_from_json_schema(data, definitions)
    assert field.has_default()
    assert field.get_default_value() == 42

def test_if_then_else_from_json_schema_with_nested_schemas():
    definitions = Definitions()
    data = {"if": {"type": "object", "properties": {"x": {"type": "integer"}}}, "then": {"type": "string"}, "else": {"type": "array"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate({"x": 5})
    assert isinstance(result, str)
    result = field.validate([1, 2, 3])
    assert isinstance(result, list)

def test_if_then_else_from_json_schema_with_ref_in_if():
    definitions = Definitions()
    definitions["#/components/schemas/MyString"] = from_json_schema({"type": "string"}, definitions)
    data = {"if": {"$ref": "#/components/schemas/MyString"}, "then": {"type": "integer"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert isinstance(result, int)
    result = field.validate(123)
    assert result == 123

def test_if_then_else_from_json_schema_complex_condition():
    definitions = Definitions()
    data = {"if": {"allOf": [{"type": "number"}, {"minimum": 0}]}, "then": {"type": "string"}, "else": {"type": "null"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate(10)
    assert isinstance(result, str)
    result = field.validate(-5)
    assert result is None

def test_if_then_else_from_json_schema_with_boolean_schemas():
    definitions = Definitions()
    data = {"if": True, "then": False, "else": True}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("anything")
    assert isinstance(result, type(Any()))
    result = field.validate("other")
    assert isinstance(result, type(Any()))

def test_if_then_else_from_json_schema_empty_object_defaults():
    definitions = Definitions()
    data = {"if": {}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("value")
    assert result == "value"
    result = field.validate(None)
    assert result is None

def test_if_then_else_from_json_schema_with_enum_in_then():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"enum": ["a", "b", "c"]}, "else": {"type": "number"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result in ["a", "b", "c"]
    result = field.validate(42)
    assert result == 42


# LLM-generated content at query #12
#--------------------------

def test_additional_items_is_bool_false():
    field = Array(additional_items=False)
    result = to_json_schema(field)
    assert result["additionalItems"] == False


# LLM-generated content at query #13
#--------------------------

```python
def test_from_json_schema_type_boolean():
    from typesystem.fields import Boolean
    from typesystem.schemas import Definitions
    from typesystem.json_schema import from_json_schema_type
    data = {}
    type_string = "boolean"
    allow_null = False
    definitions = Definitions()
    result = from_json_schema_type(data, type_string, allow_null, definitions)
    assert isinstance(result, Boolean)
    assert result.allow_null == False
    assert result.coerce_types == False
    assert result.default == NO_DEFAULT


# LLM-generated content at query #14
#--------------------------

def test_property_names_is_none():
    data = {}
    type_string = "object"
    allow_null = False
    definitions = Definitions()
    field = from_json_schema_type(data, type_string, allow_null, definitions)
    assert field.property_names is None


# LLM-generated content at query #15
#--------------------------

def test_pattern_regex_flags_unicode():
    import re
    from module import String, to_json_schema
    pattern = re.compile(r"^test$", re.UNICODE)
    field = String(pattern_regex=pattern)
    result = to_json_schema(field)
    assert "pattern" in result
    assert result["pattern"] == "^test$"


# LLM-generated content at query #16
#--------------------------

def test_predicate_at_line_160_is_false():
    field = IfThenElse(if_clause=String(), then_clause=String(), else_clause=None)
    result = to_json_schema(field)
    assert "else" not in result


# LLM-generated content at query #17
#--------------------------

def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 4}
    definitions = Definitions()
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 4
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_integer():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 4}
    definitions = Definitions()
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 4
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_string():
    data = {"minLength": 1, "maxLength": 10, "format": "email", "pattern": "^a.*z$", "default": "test"}
    definitions = Definitions()
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^a.*z$"
    assert field.default == "test"
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_boolean():
    data = {"default": True}
    definitions = Definitions()
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_array():
    data = {"items": {"type": "string"}, "minItems": 1, "maxItems": 10, "uniqueItems": True, "default": ["a"]}
    definitions = Definitions()
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["a"]
    assert field.allow_null == False

def test_from_json_schema_type_object():
    data = {"properties": {"name": {"type": "string"}}, "required": ["name"], "minProperties": 1, "maxProperties": 2, "default": {"name": "John"}}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.default == {"name": "John"}
    assert field.allow_null == False

def test_from_json_schema_type_with_allow_null():
    data = {}
    definitions = Definitions()
    field = from_json_schema_type(data, "string", True, definitions)
    assert isinstance(field, String)
    assert field.allow_null == True

def test_from_json_schema_type_without_optional_fields():
    data = {}
    definitions = Definitions()
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == None
    assert field.maximum == None
    assert field.exclusive_minimum == None
    assert field.exclusive_maximum == None
    assert field.multiple_of == None
    assert field.default == NO_DEFAULT
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_array_with_list_items():
    data = {"items": [{"type": "string"}, {"type": "integer"}], "additionalItems": False}
    definitions = Definitions()
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Integer)
    assert field.additional_items == False

def test_from_json_schema_type_object_with_pattern_properties():
    data = {"patternProperties": {"^a.*$": {"type": "string"}}}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.pattern_properties["^a.*$"], String)

def test_from_json_schema_type_object_with_additional_properties_field():
    data = {"additionalProperties": {"type": "integer"}}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.additional_properties, Integer)

def test_from_json_schema_type_object_with_property_names():
    data = {"propertyNames": {"pattern": "^[a-z]+$"}}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.property_names, String)
    assert field.property_names.pattern == "^[a-z]+$"


# LLM-generated content at query #18
#--------------------------

def test_ref_from_json_schema_with_valid_ref():
    definitions = Definitions()
    definitions["#/components/schemas/User"] = "UserSchema"
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, definitions)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is definitions

def test_ref_from_json_schema_raises_assertion_error_for_non_hash_ref():
    definitions = Definitions()
    data = {"$ref": "http://example.com/schema"}
    try:
        ref_from_json_schema(data, definitions)
        assert False
    except AssertionError as e:
        assert str(e) == "Unsupported $ref style in document."

def test_ref_from_json_schema_raises_key_error_for_missing_ref_key():
    definitions = Definitions()
    data = {}
    try:
        ref_from_json_schema(data, definitions)
        assert False
    except KeyError:
        pass


# LLM-generated content at query #19
#--------------------------

def test_line_172_predicate_true_with_definitions():
    field = Reference(to="TestRef", target=String())
    result = to_json_schema(field)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "TestRef" in result["components"]["schemas"]


# LLM-generated content at query #20
#--------------------------

def test_if_then_else_from_json_schema_with_else_key_present():
    data = {"if": {}, "else": {}}
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)


# LLM-generated content at query #21
#--------------------------

```python
def test_from_json_schema_type_integer():
    from typesystem.json_schema import from_json_schema_type
    from typesystem.schemas import Definitions
    data = {}
    definitions = Definitions()
    result = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(result, type(result).__module__ + ".Integer")


# LLM-generated content at query #22
#--------------------------

def test_to_json_schema_with_definitions():
    definitions = Definitions()
    definitions["TestField"] = String()
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "TestField" in result["components"]["schemas"]
    assert result["components"]["schemas"]["TestField"]["type"] == "string"

def test_to_json_schema_string_field():
    field = String()
    result = to_json_schema(field)
    assert result["type"] == "string"

def test_to_json_schema_string_field_with_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

def test_to_json_schema_string_field_with_min_length():
    field = String(min_length=5)
    result = to_json_schema(field)
    assert result["minLength"] == 5

def test_to_json_schema_string_field_with_max_length():
    field = String(max_length=10)
    result = to_json_schema(field)
    assert result["maxLength"] == 10

def test_to_json_schema_string_field_with_pattern():
    field = String(pattern=r"^\d+$")
    result = to_json_schema(field)
    assert result["pattern"] == r"^\d+$"

def test_to_json_schema_string_field_with_format():
    field = String(format="email")
    result = to_json_schema(field)
    assert result["format"] == "email"

def test_to_json_schema_integer_field():
    field = Integer()
    result = to_json_schema(field)
    assert result["type"] == "integer"

def test_to_json_schema_integer_field_with_null():
    field = Integer(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["integer", "null"]

def test_to_json_schema_integer_field_with_minimum():
    field = Integer(minimum=0)
    result = to_json_schema(field)
    assert result["minimum"] == 0

def test_to_json_schema_integer_field_with_maximum():
    field = Integer(maximum=100)
    result = to_json_schema(field)
    assert result["maximum"] == 100

def test_to_json_schema_integer_field_with_exclusive_minimum():
    field = Integer(exclusive_minimum=0)
    result = to_json_schema(field)
    assert result["exclusiveMinimum"] == 0

def test_to_json_schema_integer_field_with_exclusive_maximum():
    field = Integer(exclusive_maximum=100)
    result = to_json_schema(field)
    assert result["exclusiveMaximum"] == 100

def test_to_json_schema_integer_field_with_multiple_of():
    field = Integer(multiple_of=5)
    result = to_json_schema(field)
    assert result["multipleOf"] == 5

def test_to_json_schema_float_field():
    field = Float()
    result = to_json_schema(field)
    assert result["type"] == "number"

def test_to_json_schema_decimal_field():
    field = Decimal()
    result = to_json_schema(field)
    assert result["type"] == "number"

def test_to_json_schema_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    assert result["type"] == "boolean"

def test_to_json_schema_boolean_field_with_null():
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

def test_to_json_schema_array_field():
    field = Array()
    result = to_json_schema(field)
    assert result["type"] == "array"

def test_to_json_schema_array_field_with_null():
    field = Array(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["array", "null"]

def test_to_json_schema_array_field_with_min_items():
    field = Array(min_items=1)
    result = to_json_schema(field)
    assert result["minItems"] == 1

def test_to_json_schema_array_field_with_max_items():
    field = Array(max_items=10)
    result = to_json_schema(field)
    assert result["maxItems"] == 10

def test_to_json_schema_array_field_with_items():
    field = Array(items=String())
    result = to_json_schema(field)
    assert "items" in result
    assert result["items"]["type"] == "string"

def test_to_json_schema_array_field_with_items_list():
    field = Array(items=[String(), Integer()])
    result = to_json_schema(field)
    assert "items" in result
    assert isinstance(result["items"], list)
    assert result["items"][0]["type"] == "string"
    assert result["items"][1]["type"] == "integer"

def test_to_json_schema_array_field_with_additional_items():
    field = Array(additional_items=False)
    result = to_json_schema(field)
    assert result["additionalItems"] == False

def test_to_json_schema_array_field_with_unique_items():
    field = Array(unique_items=True)
    result = to_json_schema(field)
    assert result["uniqueItems"] == True

def test_to_json_schema_object_field():
    field = Object()
    result = to_json_schema(field)
    assert result["type"] == "object"

def test_to_json_schema_object_field_with_null():
    field = Object(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["object", "null"]

def test_to_json_schema_object_field_with_properties():
    field = Object(properties={"name": String()})
    result = to_json_schema(field)
    assert "properties" in result
    assert "name" in result["properties"]
    assert result["properties"]["name"]["type"] == "string"

def test_to_json_schema_object_field_with_pattern_properties():
    field = Object(pattern_properties={r"^\d+$": String()})
    result = to_json_schema(field)
    assert "patternProperties" in result
    assert r"^\d+$" in result["patternProperties"]
    assert result["patternProperties"][r"^\d+$"]["type"] == "string"

def test_to_json_schema_object_field_with_additional_properties():
    field = Object(additional_properties=False)
    result = to_json_schema(field)
    assert result["additionalProperties"] == False

def test_to_json_schema_object_field_with_property_names():
    field = Object(property_names=String(pattern=r"^[a-z]+$"))
    result = to_json_schema(field)
    assert "propertyNames" in result
    assert result["propertyNames"]["type"] == "string"
    assert result["propertyNames"]["pattern"] == r"^[a-z]+$"

def test_to_json_schema_object_field_with_max_properties():
    field = Object(max_properties=5)
    result = to_json_schema(field)
    assert result["maxProperties"] == 5

def test_to_json_schema_object_field_with_min_properties():
    field = Object(min_properties=1)
    result = to_json_schema(field)
    assert result["minProperties"] == 1

def test_to_json_schema_object_field_with_required():
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    assert "required" in result
    assert result["required"] == ["name"]

def test_to_json_schema_schema_field():
    field = Schema(fields={"name": String()})
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert result["properties"]["name"]["type"] == "string"

def test_to_json_schema_schema_field_with_required():
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert "required" in result
    assert result["required"] == ["name"]

def test_to_json_schema_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    assert "enum" in result
    assert result["enum"] == ["a", "b"]

def test_to_json_schema_const_field():
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert "const" in result
    assert result["const"] == "fixed_value"

def test_to_json_schema_union_field():
    field = String() | Integer()
    result = to_json_schema(field)
    assert "anyOf" in result
    assert isinstance(result["anyOf"], list)
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

def test_to_json_schema_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "oneOf" in result
    assert isinstance(result["oneOf"], list)
    assert result["oneOf"][0]["type"] == "


