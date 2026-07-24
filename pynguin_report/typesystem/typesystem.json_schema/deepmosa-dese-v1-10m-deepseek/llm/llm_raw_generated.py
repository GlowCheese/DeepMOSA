####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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

def test_to_json_schema_with_object_field_required():
    field = Object(properties={"name": String()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    assert result == expected

def test_to_json_schema_with_object_field_min_properties():
    field = Object(min_properties=1)
    result = to_json_schema(field)
    expected = {"type": "object", "minProperties": 1}
    assert result == expected

def test_to_json_schema_with_object_field_max_properties():
    field = Object(max_properties=10)
    result = to_json_schema(field)
    expected = {"type": "object", "maxProperties": 10}
    assert result == expected

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String()})
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert result == expected

def test_to_json_schema_with_schema_field_required():
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    assert result == expected

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    expected = {"enum": ["a", "b"]}
    assert result == expected

def test_to_json_schema_with_const_field():
    field = Const(const="fixed")
    result = to_json_schema(field)
    expected = {"const": "fixed"}
    assert result == expected

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_oneof_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_allof_field():
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(field)
    expected = {"if": {"type": "


# LLM-generated content at query #2
#--------------------------

def test_additional_properties_is_not_bool():
    field = Object(additional_properties=String())
    result = to_json_schema(field)
    assert isinstance(result.get("additionalProperties"), dict)


# LLM-generated content at query #3
#--------------------------

def test_additional_items_is_not_bool():
    from my_module import Array, String
    field = Array(items=String(), additional_items=String())
    result = to_json_schema(field)
    assert isinstance(result["additionalItems"], dict)


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_17_is_true_for_definitions_instance():
    from typing import Union
    from module import to_json_schema, Definitions, Field, String
    definitions_input = Definitions({"test": String()})
    result = to_json_schema(definitions_input)
    assert isinstance(result, dict)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "test" in result["components"]["schemas"]
    assert result["components"]["schemas"]["test"]["type"] == "string"


# LLM-generated content at query #5
#--------------------------

def test_object_field_with_allow_null_sets_type_to_array():
    obj = Object(allow_null=True)
    result = to_json_schema(obj)
    assert result["type"] == ["object", "null"]


# LLM-generated content at query #6
#--------------------------

def test_ref_from_json_schema_creates_reference_with_correct_to():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
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

def test_ref_from_json_schema_raises_key_error_for_missing_ref():
    definitions = Definitions()
    data = {}
    try:
        ref_from_json_schema(data, definitions)
        assert False
    except KeyError:
        pass


# LLM-generated content at query #7
#--------------------------

def test_additional_items_is_not_bool():
    from my_module import Array, String
    field = Array(items=String(), additional_items=String())
    result = to_json_schema(field)
    assert isinstance(result["additionalItems"], dict)


# LLM-generated content at query #8
#--------------------------

def test_from_json_schema_boolean_true():
    result = from_json_schema(True)
    assert isinstance(result, Any)

def test_from_json_schema_boolean_false():
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

def test_from_json_schema_empty_object():
    result = from_json_schema({})
    assert isinstance(result, Any)

def test_from_json_schema_with_ref():
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = Integer()
    data = {"$ref": "#/components/schemas/Test"}
    result = from_json_schema(data, definitions)
    assert isinstance(result, Reference)

def test_from_json_schema_with_type_string():
    data = {"type": "string"}
    result = from_json_schema(data)
    assert isinstance(result, String)

def test_from_json_schema_with_type_integer():
    data = {"type": "integer"}
    result = from_json_schema(data)
    assert isinstance(result, Integer)

def test_from_json_schema_with_type_number():
    data = {"type": "number"}
    result = from_json_schema(data)
    assert isinstance(result, Float)

def test_from_json_schema_with_type_boolean():
    data = {"type": "boolean"}
    result = from_json_schema(data)
    assert isinstance(result, Boolean)

def test_from_json_schema_with_type_array():
    data = {"type": "array"}
    result = from_json_schema(data)
    assert isinstance(result, Array)

def test_from_json_schema_with_type_object():
    data = {"type": "object"}
    result = from_json_schema(data)
    assert isinstance(result, Object)

def test_from_json_schema_with_enum():
    data = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)

def test_from_json_schema_with_const():
    data = {"const": 42}
    result = from_json_schema(data)
    assert isinstance(result, Const)

def test_from_json_schema_with_allOf():
    data = {"allOf": [{"type": "string"}, {"minLength": 1}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_with_anyOf():
    data = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)

def test_from_json_schema_with_oneOf():
    data = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)

def test_from_json_schema_with_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)

def test_from_json_schema_with_if_then():
    data = {"if": {"type": "string"}, "then": {"minLength": 1}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_with_if_then_else():
    data = {"if": {"type": "string"}, "then": {"minLength": 1}, "else": {"type": "integer"}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_multiple_constraints():
    data = {"type": "string", "enum": ["a", "b"]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_with_components():
    data = {"components": {"schemas": {"Test": {"type": "string"}}}}
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_nullable_type():
    data = {"type": ["string", "null"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert result.allow_null == True

def test_from_json_schema_multiple_types():
    data = {"type": ["string", "integer"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert result.allow_null == False

def test_from_json_schema_no_type_but_nullable():
    data = {"type": "null"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const is None

def test_from_json_schema_no_type_not_nullable():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)


# LLM-generated content at query #9
#--------------------------

def test_additional_items_is_false():
    field = Array(additional_items=False)
    result = to_json_schema(field)
    assert result.get("additionalItems") is False


# LLM-generated content at query #10
#--------------------------

def test_line_46_predicate_evaluates_true_for_integer_with_allow_null():
    field = Integer(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["integer", "null"]

def test_line_46_predicate_evaluates_true_for_float_with_allow_null():
    field = Float(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]

def test_line_46_predicate_evaluates_true_for_decimal_with_allow_null():
    field = Decimal(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]


# LLM-generated content at query #11
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
    field = Integer(multiple_of=2)
    result = to_json_schema(field)
    expected = {"type": "integer", "multipleOf": 2}
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

def test_to_json_schema_with_object_field_additional_properties():
    field = Object(additional_properties=False)
    result = to_json_schema(field)
    expected = {"type": "object", "additionalProperties": False}
    assert result == expected

def test_to_json_schema_with_object_field_required():
    field = Object(properties={"name": String()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    assert result == expected

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    expected = {"enum": ["a", "b"]}
    assert result == expected

def test_to_json_schema_with_const_field():
    field = Const(const="fixed")
    result = to_json_schema(field)
    expected = {"const": "fixed"}
    assert result == expected

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_oneof_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_allof_field():
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(field)
    expected = {"if": {"type": "string"}, "then": {"type": "integer"}}
    assert result == expected

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    result = to_json_schema(field)
    expected = {"not": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_default_value():
    field = String(default="default")
    result = to_json_schema(field)
    expected = {"type": "string", "default": "default"}
    assert result == expected

def test_to_json_schema_with_definitions():
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}}}}
    assert result == expected

def test_to_json_schema_with_reference_field():
    field = Reference(to="User", target=Object(properties={"name": String()}))
    result = to_json_schema(field)
    expected = {"$ref": "#/components/schemas/User", "components": {"schemas": {"User


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_17_evaluates_to_true():
    arg = Definitions()
    result = isinstance(arg, Definitions)
    assert result == True


# LLM-generated content at query #13
#--------------------------

def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": -1, "exclusiveMaximum": 11, "multipleOf": 2, "default": 5.0}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == -1
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 5.0
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_integer():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": -1, "exclusiveMaximum": 11, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "integer", True, Definitions())
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == -1
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 5
    assert field.allow_null == True
    assert field.coerce_types == False

def test_from_json_schema_type_string():
    data = {"minLength": 3, "maxLength": 10, "format": "email", "pattern": "^[a-z]+$", "default": "abc"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert field.min_length == 3
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-z]+$"
    assert field.default == "abc"
    assert field.allow_null == False
    assert field.allow_blank == False
    assert field.coerce_types == False

def test_from_json_schema_type_string_allow_blank():
    data = {"minLength": 0, "maxLength": 10, "default": ""}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.min_length == None
    assert field.max_length == 10
    assert field.allow_blank == True
    assert field.allow_null == True

def test_from_json_schema_type_boolean():
    data = {"default": True}
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert field.default == True
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_array():
    items = {"type": "string"}
    data = {"items": items, "minItems": 1, "maxItems": 5, "uniqueItems": True, "default": ["a"]}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items == True
    assert field.default == ["a"]
    assert field.allow_null == False
    assert isinstance(field.items, String)

def test_from_json_schema_type_array_list_items():
    items = [{"type": "string"}, {"type": "integer"}]
    data = {"items": items, "additionalItems": False, "minItems": 2, "maxItems": 2}
    field = from_json_schema_type(data, "array", True, Definitions())
    assert field.min_items == 2
    assert field.max_items == 2
    assert field.additional_items == False
    assert field.allow_null == True
    assert isinstance(field.items, list)
    assert len(field.items) == 2
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Integer)

def test_from_json_schema_type_array_additional_items_field():
    items = {"type": "string"}
    additional_items = {"type": "integer"}
    data = {"items": items, "additionalItems": additional_items}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field.items, String)
    assert isinstance(field.additional_items, Integer)

def test_from_json_schema_type_object():
    properties = {"name": {"type": "string"}, "age": {"type": "integer"}}
    pattern_properties = {"^s_": {"type": "string"}}
    additional_properties = {"type": "boolean"}
    property_names = {"pattern": "^[a-z]+$"}
    data = {"properties": properties, "patternProperties": pattern_properties, "additionalProperties": additional_properties, "propertyNames": property_names, "minProperties": 1, "maxProperties": 5, "required": ["name"], "default": {"name": "test"}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert field.min_properties == 1
    assert field.max_properties == 5
    assert field.required == ["name"]
    assert field.default == {"name": "test"}
    assert field.allow_null == False
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert isinstance(field.pattern_properties["^s_"], String)
    assert isinstance(field.additional_properties, Boolean)
    assert isinstance(field.property_names, String)

def test_from_json_schema_type_object_additional_properties_bool():
    data = {"additionalProperties": False}
    field = from_json_schema_type(data, "object", True, Definitions())
    assert field.additional_properties == False
    assert field.allow_null == True

def test_from_json_schema_type_invalid_type_string():
    try:
        from_json_schema_type({}, "invalid", False, Definitions())
        assert False
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid'"


# LLM-generated content at query #14
#--------------------------

def test_pattern_regex_flags_unicode():
    import re
    from module import String, to_json_schema
    pattern = re.compile(r"test", flags=re.RegexFlag.UNICODE)
    field = String(pattern_regex=pattern)
    result = to_json_schema(field)
    assert "pattern" in result
    assert result["pattern"] == "test"


# LLM-generated content at query #15
#--------------------------

def test_to_json_schema_with_string_field():
    field = String(title="test", description="test desc", allow_null=True, default="default")
    result = to_json_schema(field)
    expected = {"type": ["string", "null"], "default": "default", "title": "test", "description": "test desc"}
    assert result == expected

def test_to_json_schema_with_integer_field():
    field = Integer(minimum=0, maximum=100, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "integer", "minimum": 0, "maximum": 100}
    assert result == expected

def test_to_json_schema_with_boolean_field():
    field = Boolean(default=True, allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["boolean", "null"], "default": True}
    assert result == expected

def test_to_json_schema_with_array_field():
    field = Array(items=String(), min_items=1, max_items=10, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "array", "minItems": 1, "maxItems": 10, "items": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_object_field():
    field = Object(properties={"name": String()}, required=["name"], allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["object", "null"], "properties": {"name": {"type": "string"}}, "required": ["name"]}
    assert result == expected

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()], allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["string", "null"], "anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_definitions():
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}}}}
    assert result == expected

def test_to_json_schema_with_reference_field():
    target = Object(properties={"name": String()})
    field = Reference(to="User", target=target)
    result = to_json_schema(field)
    expected = {"$ref": "#/components/schemas/User", "components": {"schemas": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}}}}
    assert result == expected

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")], default="a")
    result = to_json_schema(field)
    expected = {"enum": ["a", "b"], "default": "a"}
    assert result == expected

def test_to_json_schema_with_const_field():
    field = Const(const="fixed_value", allow_null=False)
    result = to_json_schema(field)
    expected = {"const": "fixed_value"}
    assert result == expected

def test_to_json_schema_with_allof_field():
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)], allow_null=False)
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string", "minLength": 1}, {"type": "string", "maxLength": 10}]}
    assert result == expected

def test_to_json_schema_with_oneof_field():
    field = OneOf(one_of=[String(), Integer()], allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["string", "null"], "oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    field = IfThenElse(if_clause=String(min_length=5), then_clause=Integer(), else_clause=Boolean(), allow_null=False)
    result = to_json_schema(field)
    expected = {"if": {"type": "string", "minLength": 5}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    assert result == expected

def test_to_json_schema_with_not_field():
    field = Not(negated=String(), allow_null=False)
    result = to_json_schema(field)
    expected = {"not": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_unknown_field_type():
    class UnknownField(Field):
        pass
    field = UnknownField()
    try:
        to_json_schema(field)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)

def test_to_json_schema_with_any_field():
    field = Any()
    result = to_json_schema(field)
    assert result == True

def test_to_json_schema_with_nevermatch_field():
    field = NeverMatch()
    result = to_json_schema(field)
    assert result == False

def test_to_json_schema_with_field_having_callable_default():
    field = String(default=lambda: "dynamic_default", allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "string", "default": "dynamic_default"}
    assert result == expected

def test_to_json_schema_with_field_without_default():
    field = String(allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "string"}
    assert result == expected

def test_to_json_schema_with_field_with_pattern_regex():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"), allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "string", "pattern": "^[a-z]+$"}
    assert result == expected

def test_to_json_schema_with_field_with_format():
    field = String(format="email", allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "string", "format": "email"}
    assert result == expected

def test_to_json_schema_with_field_with_exclusive_bounds():
    field = Integer(exclusive_minimum=0, exclusive_maximum=100, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100}
    assert result == expected

def test_to_json_schema_with_field_with_multiple_of():
    field = Integer(multiple_of=5, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "integer", "multipleOf": 5}
    assert result == expected

def test_to_json_schema_with_array_field_with_additional_items():
    field = Array(items=String(), additional_items=Boolean(), allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "array", "items": {"type": "string"}, "additionalItems": {"type": "boolean"}}
    assert result == expected

def test_to_json_schema_with_array_field_with_unique_items():
    field = Array(items=String(), unique_items=True, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
    assert result == expected

def test_to_json_schema_with_object_field_with_pattern_properties():
    field = Object(pattern_properties={r"^test_": String()}, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "object", "patternProperties": {"^test_": {"type": "string"}}}
    assert result == expected

def test_to_json_schema_with_object_field_with_additional_properties():
    field = Object(additional_properties=Integer(), allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert result == expected

def test_to_json_schema_with_object_field_with_property_names():
    field = Object(property_names=String(min_length=1), allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "object", "propertyNames": {"type": "string", "minLength": 1}}
    assert result == expected

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String()}, required=["name"], allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    assert result == expected


# LLM-generated content at query #16
#--------------------------

def test_to_json_schema_with_definitions():
    definitions = Definitions()
    definitions["TestField"] = String()
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"TestField": {"type": "string"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_nested():
    definitions = Definitions()
    definitions["Nested"] = Object(properties={"inner": String()})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Nested": {"type": "object", "properties": {"inner": {"type": "string"}}}}}}
    assert result == expected

def test_to_json_schema_with_definitions_multiple():
    definitions = Definitions()
    definitions["Field1"] = String()
    definitions["Field2"] = Integer()
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Field1": {"type": "string"}, "Field2": {"type": "integer"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_and_field():
    definitions = Definitions()
    definitions["RefField"] = String()
    field = Reference(to="RefField")
    result = to_json_schema(field, _definitions=definitions)
    expected = {"$ref": "#/components/schemas/RefField"}
    assert result == expected

def test_to_json_schema_with_definitions_duplicate():
    definitions = Definitions()
    definitions["Test"] = String()
    try:
        definitions["Test"] = Integer()
        assert False
    except AssertionError as e:
        assert str(e) == "Definition for 'Test' has already been set."

def test_to_json_schema_with_definitions_empty():
    definitions = Definitions()
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {}}}
    assert result == expected

def test_to_json_schema_with_definitions_and_root_field():
    definitions = Definitions()
    definitions["Ref"] = String()
    root_field = Object(properties={"ref": Reference(to="Ref")})
    result = to_json_schema(root_field, _definitions=definitions)
    expected = {"type": "object", "properties": {"ref": {"$ref": "#/components/schemas/Ref"}}, "components": {"schemas": {"Ref": {"type": "string"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_shared():
    shared_definitions = {}
    field1 = Object(properties={"name": String()})
    result1 = to_json_schema(field1, _definitions=shared_definitions)
    field2 = Object(properties={"age": Integer()})
    result2 = to_json_schema(field2, _definitions=shared_definitions)
    assert shared_definitions == {}

def test_to_json_schema_with_definitions_reference_chain():
    definitions = Definitions()
    definitions["A"] = Object(properties={"b": Reference(to="B")})
    definitions["B"] = Object(properties={"c": Reference(to="C")})
    definitions["C"] = String()
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"A": {"type": "object", "properties": {"b": {"$ref": "#/components/schemas/B"}}}, "B": {"type": "object", "properties": {"c": {"$ref": "#/components/schemas/C"}}}, "C": {"type": "string"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_circular():
    definitions = Definitions()
    definitions["Node"] = Object(properties={"next": Reference(to="Node")})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Node"}}}}}}
    assert result == expected


# LLM-generated content at query #17
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

def test_to_json_schema_with_integer_field_minimum_maximum():
    field = Integer(minimum=0, maximum=100)
    result = to_json_schema(field)
    expected = {"type": "integer", "minimum": 0, "maximum": 100}
    assert result == expected

def test_to_json_schema_with_integer_field_exclusive_minimum_maximum():
    field = Integer(exclusive_minimum=0, exclusive_maximum=100)
    result = to_json_schema(field)
    expected = {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100}
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

def test_to_json_schema_with_array_field_min_items_max_items():
    field = Array(min_items=1, max_items=10)
    result = to_json_schema(field)
    expected = {"type": "array", "minItems": 1, "maxItems": 10}
    assert result == expected

def test_to_json_schema_with_array_field_items():
    field = Array(items=String())
    result = to_json_schema(field)
    expected = {"type": "array", "items": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_array_field_additional_items():
    field = Array(additional_items=False)
    result = to_json_schema(field)
    expected = {"type": "array", "additionalItems": False}
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

def test_to_json_schema_with_object_field_additional_properties():
    field = Object(additional_properties=False)
    result = to_json_schema(field)
    expected = {"type": "object", "additionalProperties": False}
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

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    expected = {"enum": ["a", "b"]}
    assert result == expected

def test_to_json_schema_with_const_field():
    field = Const(const="fixed")
    result = to_json_schema(field)
    expected = {"const": "fixed"}
    assert result == expected

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_oneof_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_allof_field():
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(field)
    expected = {"if": {"type": "string"}, "then": {"type": "integer"}}
    assert result == expected

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    result = to_json_schema(field)
    expected = {"not": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_reference_field():
    target = String()
    field = Reference(to="MyString", target=target)
    result = to_json_schema(field)
    expected = {"$ref": "#/components/schemas/MyString", "components": {"schemas": {"MyString": {"type": "string"}}}}
    assert result == expected

def test_to_json_schema_with_definitions():
    definitions = Definitions({"MyString": String()})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"MyString": {"type": "string"}}}}
    assert result == expected

def test_to_json_schema_with_default_value():
    field = String(default="hello")
    result = to_json_schema(field)
    expected = {"type": "string", "default": "hello"}
   


# LLM-generated content at query #18
#--------------------------

def test_string_field_with_allow_null_true():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]


# LLM-generated content at query #19
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

def test_to_json_schema_with_integer_field_minimum_maximum():
    field = Integer(minimum=0, maximum=100)
    result = to_json_schema(field)
    expected = {"type": "integer", "minimum": 0, "maximum": 100}
    assert result == expected

def test_to_json_schema_with_integer_field_exclusive_minimum_maximum():
    field = Integer(exclusive_minimum=0, exclusive_maximum=100)
    result = to_json_schema(field)
    expected = {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100}
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

def test_to_json_schema_with_array_field_min_items_max_items():
    field = Array(min_items=1, max_items=10)
    result = to_json_schema(field)
    expected = {"type": "array", "minItems": 1, "maxItems": 10}
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

def test_to_json_schema_with_array_field_additional_items():
    field = Array(additional_items=False)
    result = to_json_schema(field)
    expected = {"type": "array", "additionalItems": False}
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
    field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    assert result == expected

def test_to_json_schema_with_object_field_additional_properties():
    field = Object(additional_properties=False)
    result = to_json_schema(field)
    expected = {"type": "object", "additionalProperties": False}
    assert result == expected

def test_to_json_schema_with_object_field_required():
    field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}
    assert result == expected

def test_to_json_schema_with_object_field_min_properties_max_properties():
    field = Object(min_properties=1, max_properties=5)
    result = to_json_schema(field)
    expected = {"type": "object", "minProperties": 1, "maxProperties": 5}
    assert result == expected

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    assert result == expected

def test_to_json_schema_with_schema_field_required():
    field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}
    assert result == expected

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    expected = {"enum": ["a", "b"]}
    assert result == expected

def test_to_json_schema_with_const_field():
    field = Const(const="fixed")
    result = to_json_schema(field)
    expected = {"const": "fixed"}
    assert result == expected

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_oneof_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_allof_field():
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string", "minLength": 1}, {"type": "string", "maxLength": 10}]}
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    field = IfThenElse(if_clause=String(), then_clause=


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_to_json_schema_with_definitions():
    definitions = Definitions()
    definitions["User"] = String()
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    assert result["components"]["schemas"]["User"]["type"] == "string"

def test_to_json_schema_string_field():
    field = String()
    result = to_json_schema(field)
    assert result["type"] == "string"

def test_to_json_schema_string_field_with_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

def test_to_json_schema_string_field_with_default():
    field = String(default="hello")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["default"] == "hello"

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
    assert result["type"] == "integer"
    assert result["minimum"] == 0

def test_to_json_schema_float_field():
    field = Float()
    result = to_json_schema(field)
    assert result["type"] == "number"

def test_to_json_schema_float_field_with_null():
    field = Float(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]

def test_to_json_schema_boolean_field():
    field = Boolean()
    result = to_json_schema(field)
    assert result["type"] == "boolean"

def test_to_json_schema_boolean_field_with_null():
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

def test_to_json_schema_array_field():
    field = Array(items=String())
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert "items" in result
    assert result["items"]["type"] == "string"

def test_to_json_schema_array_field_with_null():
    field = Array(items=String(), allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["array", "null"]

def test_to_json_schema_array_field_with_min_items():
    field = Array(items=String(), min_items=1)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1

def test_to_json_schema_object_field():
    field = Object(properties={"name": String()})
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert result["properties"]["name"]["type"] == "string"

def test_to_json_schema_object_field_with_null():
    field = Object(properties={"name": String()}, allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["object", "null"]

def test_to_json_schema_object_field_with_required():
    field = Object(properties={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "required" in result
    assert result["required"] == ["name"]

def test_to_json_schema_union_field():
    field = String() | Integer()
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

def test_to_json_schema_union_field_with_null():
    field = String(allow_null=True) | Integer()
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == ["string", "null"]
    assert result["anyOf"][1]["type"] == "integer"

def test_to_json_schema_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    assert "enum" in result
    assert result["enum"] == ["a", "b"]

def test_to_json_schema_const_field():
    field = Const(const="fixed")
    result = to_json_schema(field)
    assert "const" in result
    assert result["const"] == "fixed"

def test_to_json_schema_reference_field():
    definitions = Definitions()
    definitions["User"] = String()
    field = Reference(to="User", target=String())
    result = to_json_schema(field, _definitions=definitions)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/User"
    assert "User" in definitions
    assert definitions["User"]["type"] == "string"

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
    assert result["type"] == "object"
    assert "required" in result
    assert result["required"] == ["name"]

def test_to_json_schema_if_then_else_field():
    if_clause = String()
    then_clause = Integer()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause)
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert result["if"]["type"] == "string"
    assert result["then"]["type"] == "integer"

def test_to_json_schema_not_field():
    negated = String()
    field = Not(negated=negated)
    result = to_json_schema(field)
    assert "not" in result
    assert result["not"]["type"] == "string"

def test_to_json_schema_all_of_field():
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    assert result["allOf"][0]["type"] == "string"
    assert result["allOf"][1]["type"] == "integer"

def test_to_json_schema_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    assert result["oneOf"][0]["type"] == "string"
    assert result["oneOf"][1]["type"] == "integer"

def test_to_json_schema_decimal_field():
    field = Decimal()
    result = to_json_schema(field)
    assert result["type"] == "number"

def test_to_json_schema_decimal_field_with_null():
    field = Decimal(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]

def test_to_json_schema_any_field():
    field = Any()
    result = to_json_schema(field)
    assert result is True

def test_to_json_schema_never_match_field():
    field = NeverMatch()
    result = to_json_schema(field)
    assert result is False

def test_to_json_schema_string_field_with_format():
    field = String(format="email")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["format"] == "email"

def test_to_json_schema_string_field_with_pattern():
    field = String(pattern=r"^\d+$")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["pattern"] == r"^\d+$"

def test_to_json_schema_string_field_with_min_length():
    field = String(min_length=5)
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 5

def test_to_json_schema_string_field_with_max_length():
    field = String(max_length=10)
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["maxLength"] == 10

def test_to_json_schema


# LLM-generated content at query #2
#--------------------------

def test_from_json_schema_with_boolean_true():
    result = from_json_schema(True)
    assert isinstance(result, Any)

def test_from_json_schema_with_boolean_false():
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

def test_from_json_schema_with_ref():
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = Integer()
    data = {"$ref": "#/components/schemas/Test"}
    result = from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/Test"

def test_from_json_schema_with_type_string():
    data = {"type": "string"}
    result = from_json_schema(data)
    assert isinstance(result, String)
    assert result.allow_null == False

def test_from_json_schema_with_type_integer():
    data = {"type": "integer"}
    result = from_json_schema(data)
    assert isinstance(result, Integer)
    assert result.allow_null == False

def test_from_json_schema_with_type_number():
    data = {"type": "number"}
    result = from_json_schema(data)
    assert isinstance(result, Float)
    assert result.allow_null == False

def test_from_json_schema_with_type_boolean():
    data = {"type": "boolean"}
    result = from_json_schema(data)
    assert isinstance(result, Boolean)
    assert result.allow_null == False

def test_from_json_schema_with_type_array():
    data = {"type": "array"}
    result = from_json_schema(data)
    assert isinstance(result, Array)
    assert result.allow_null == False

def test_from_json_schema_with_type_object():
    data = {"type": "object"}
    result = from_json_schema(data)
    assert isinstance(result, Object)
    assert result.allow_null == False

def test_from_json_schema_with_nullable_type():
    data = {"type": ["string", "null"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert result.allow_null == True

def test_from_json_schema_with_enum():
    data = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]

def test_from_json_schema_with_const():
    data = {"const": 42}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == 42

def test_from_json_schema_with_allOf():
    data = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_from_json_schema_with_anyOf():
    data = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

def test_from_json_schema_with_oneOf():
    data = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2

def test_from_json_schema_with_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)
    assert isinstance(result.negated, String)

def test_from_json_schema_with_if_then_else():
    data = {"if": {"type": "string"}, "then": {"minLength": 5}, "else": {"type": "integer"}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, String)
    assert isinstance(result.else_clause, Integer)

def test_from_json_schema_with_multiple_constraints():
    data = {"type": "string", "enum": ["a", "b"], "const": "a"}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 3

def test_from_json_schema_with_no_constraints():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_with_components_schemas():
    data = {"components": {"schemas": {"Test": {"type": "string"}}}}
    result = from_json_schema(data)
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    assert isinstance(result, Any)

def test_from_json_schema_with_allow_null_true():
    data = {"type": "string", "nullable": True}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert result.allow_null == True

def test_from_json_schema_with_allow_null_false():
    data = {"type": "string", "nullable": False}
    result = from_json_schema(data)
    assert isinstance(result, String)
    assert result.allow_null == False

def test_from_json_schema_with_default_value():
    data = {"type": "string", "default": "hello"}
    result = from_json_schema(data)
    assert result.default == "hello"

def test_from_json_schema_with_array_items():
    data = {"type": "array", "items": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

def test_from_json_schema_with_object_properties():
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = from_json_schema(data)
    assert isinstance(result, Object)
    assert isinstance(result.properties["name"], String)

def test_from_json_schema_with_union_types():
    data = {"type": ["string", "integer"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Integer)


# LLM-generated content at query #3
#--------------------------

def test_pattern_properties_present():
    field = Object(pattern_properties={"^test.*": String()})
    result = to_json_schema(field)
    assert "patternProperties" in result


# LLM-generated content at query #4
#--------------------------

def test_if_then_else_from_json_schema_basic():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result == "hello"
    result = field.validate(123)
    assert result == 123
    result = field.validate(True)
    assert result == True

def test_if_then_else_from_json_schema_missing_then():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "else": {"type": "boolean"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result == "hello"
    result = field.validate(True)
    assert result == True

def test_if_then_else_from_json_schema_missing_else():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"type": "integer"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result == "hello"
    result = field.validate(123)
    assert result == 123

def test_if_then_else_from_json_schema_default_values():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}, "default": 42}
    field = if_then_else_from_json_schema(data, definitions)
    assert field.has_default()
    assert field.default == 42

def test_if_then_else_from_json_schema_with_ref():
    definitions = Definitions()
    definitions["#/components/schemas/StringType"] = from_json_schema({"type": "string"}, definitions)
    data = {"if": {"$ref": "#/components/schemas/StringType"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result == "hello"
    result = field.validate(123)
    assert result == 123
    result = field.validate(True)
    assert result == True

def test_if_then_else_from_json_schema_with_const():
    definitions = Definitions()
    data = {"if": {"const": "yes"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("yes")
    assert result == "yes"
    result = field.validate("no")
    assert result == True

def test_if_then_else_from_json_schema_with_enum():
    definitions = Definitions()
    data = {"if": {"enum": ["yes", "no"]}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("yes")
    assert result == "yes"
    result = field.validate("maybe")
    assert result == True

def test_if_then_else_from_json_schema_with_all_of():
    definitions = Definitions()
    data = {"if": {"allOf": [{"type": "string"}, {"maxLength": 5}]}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result == "hello"
    result = field.validate("toolong")
    assert result == True

def test_if_then_else_from_json_schema_with_any_of():
    definitions = Definitions()
    data = {"if": {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "then": {"type": "boolean"}, "else": {"type": "null"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result == True
    result = field.validate(123)
    assert result == True
    result = field.validate(True)
    assert result == True

def test_if_then_else_from_json_schema_with_one_of():
    definitions = Definitions()
    data = {"if": {"oneOf": [{"type": "string"}, {"type": "integer"}]}, "then": {"type": "boolean"}, "else": {"type": "null"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result == True
    result = field.validate(123)
    assert result == True
    result = field.validate(True)
    assert result == True

def test_if_then_else_from_json_schema_with_not():
    definitions = Definitions()
    data = {"if": {"not": {"type": "string"}}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate(123)
    assert result == 123
    result = field.validate("hello")
    assert result == True

def test_if_then_else_from_json_schema_nested_if():
    definitions = Definitions()
    data = {"if": {"type": "string"}, "then": {"if": {"maxLength": 5}, "then": {"type": "integer"}, "else": {"type": "boolean"}}, "else": {"type": "null"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result == "hello"
    result = field.validate("toolong")
    assert result == True
    result = field.validate(123)
    assert result == 123


# LLM-generated content at query #5
#--------------------------

def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5.0}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5.0
    assert not field.allow_null

def test_from_json_schema_type_integer():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "integer", True, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5
    assert field.allow_null

def test_from_json_schema_type_string():
    data = {"minLength": 1, "maxLength": 10, "format": "email", "pattern": "^a.*z$", "default": "test"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^a.*z$"
    assert field.default == "test"
    assert not field.allow_null

def test_from_json_schema_type_string_allow_blank():
    data = {"minLength": 0, "maxLength": 10, "format": "email", "pattern": "^a.*z$", "default": ""}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.allow_blank
    assert field.min_length is None
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^a.*z$"
    assert field.default == ""
    assert not field.allow_null

def test_from_json_schema_type_boolean():
    data = {"default": True}
    field = from_json_schema_type(data, "boolean", True, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True
    assert field.allow_null

def test_from_json_schema_type_array():
    items = {"type": "string"}
    data = {"items": items, "minItems": 1, "maxItems": 10, "uniqueItems": True, "default": ["a", "b"]}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items
    assert field.default == ["a", "b"]
    assert not field.allow_null

def test_from_json_schema_type_array_with_list_items():
    items = [{"type": "string"}, {"type": "integer"}]
    data = {"items": items, "minItems": 2, "maxItems": 2, "uniqueItems": False, "default": ["a", 1]}
    field = from_json_schema_type(data, "array", True, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, list)
    assert len(field.items) == 2
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Integer)
    assert field.min_items == 2
    assert field.max_items == 2
    assert not field.unique_items
    assert field.default == ["a", 1]
    assert field.allow_null

def test_from_json_schema_type_array_with_additional_items():
    items = {"type": "string"}
    additional_items = {"type": "integer"}
    data = {"items": items, "additionalItems": additional_items, "minItems": 1, "maxItems": 5, "uniqueItems": False, "default": ["a", 1, 2]}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert isinstance(field.additional_items, Integer)
    assert field.min_items == 1
    assert field.max_items == 5
    assert not field.unique_items
    assert field.default == ["a", 1, 2]
    assert not field.allow_null

def test_from_json_schema_type_array_with_additional_items_bool():
    items = {"type": "string"}
    data = {"items": items, "additionalItems": False, "minItems": 1, "maxItems": 5, "uniqueItems": False, "default": ["a", "b"]}
    field = from_json_schema_type(data, "array", True, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items == False
    assert field.min_items == 1
    assert field.max_items == 5
    assert not field.unique_items
    assert field.default == ["a", "b"]
    assert field.allow_null

def test_from_json_schema_type_object():
    properties = {"name": {"type": "string"}, "age": {"type": "integer"}}
    data = {"properties": properties, "minProperties": 1, "maxProperties": 2, "required": ["name"], "default": {"name": "John", "age": 30}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}
    assert not field.allow_null

def test_from_json_schema_type_object_with_pattern_properties():
    pattern_properties = {"^S_": {"type": "string"}}
    data = {"patternProperties": pattern_properties, "minProperties": 0, "maxProperties": 5, "default": {"S_key": "value"}}
    field = from_json_schema_type(data, "object", True, Definitions())
    assert isinstance(field, Object)
    assert "^S_" in field.pattern_properties
    assert isinstance(field.pattern_properties["^S_"], String)
    assert field.min_properties == 0
    assert field.max_properties == 5
    assert field.default == {"S_key": "value"}
    assert field.allow_null

def test_from_json_schema_type_object_with_additional_properties():
    additional_properties = {"type": "boolean"}
    data = {"additionalProperties": additional_properties, "minProperties": 0, "maxProperties": 5, "default": {"extra": True}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.additional_properties, Boolean)
    assert field.min_properties == 0
    assert field.max_properties == 5
    assert field.default == {"extra": True}
    assert not field.allow_null

def test_from_json_schema_type_object_with_additional_properties_bool():
    data = {"additionalProperties": False, "minProperties": 0, "maxProperties": 5, "default": {}}
    field = from_json_schema_type(data, "object", True, Definitions())
    assert isinstance(field, Object)
    assert field.additional_properties == False
    assert field.min_properties == 0
    assert field.max_properties == 5
    assert field.default == {}
    assert field.allow_null

def test_from_json_schema_type_object_with_property_names():
    property_names = {"type": "string", "pattern": "^[a-z]+$"}
    data = {"propertyNames": property_names, "minProperties": 0, "maxProperties": 5, "default": {"key": "value"}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.property_names, String)
    assert field.property_names.pattern == "^[a-z]+$"
    assert field.min_properties == 0
    assert field.max_properties == 5
    assert field.default == {"key": "value"}
    assert not field.allow_null


# LLM-generated content at query #6
#--------------------------

def test_ref_from_json_schema_creates_reference_with_correct_to():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is definitions


def test_ref_from_json_schema_raises_assertion_error_for_unsupported_ref():
    definitions = Definitions()
    data = {"$ref": "http://example.com/schema"}
    try:
        ref_from_json_schema(data, definitions)
        assert False
    except AssertionError as e:
        assert str(e) == "Unsupported $ref style in document."


def test_ref_from_json_schema_works_with_minimal_ref():
    definitions = Definitions()
    data = {"$ref": "#/User"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/User"
    assert result.definitions is definitions


# LLM-generated content at query #7
#--------------------------

def test_to_json_schema_with_definitions():
    definitions = Definitions({"User": String()})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "string"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_nested():
    inner_definitions = Definitions({"Inner": Integer()})
    outer_definitions = Definitions({"Outer": inner_definitions})
    result = to_json_schema(outer_definitions)
    expected = {"components": {"schemas": {"Outer": {"components": {"schemas": {"Inner": {"type": "integer"}}}}}}}
    assert result == expected

def test_to_json_schema_with_definitions_duplicate_key():
    definitions = Definitions({"User": String()})
    definitions["User"] = Integer()
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "integer"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_empty():
    definitions = Definitions()
    result = to_json_schema(definitions)
    expected = {}
    assert result == expected

def test_to_json_schema_with_definitions_and_field():
    definitions = Definitions({"User": String()})
    field = Integer()
    result = to_json_schema(field, _definitions=definitions)
    expected = {"type": "integer"}
    assert result == expected

def test_to_json_schema_with_definitions_root():
    definitions = Definitions({"User": String()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]

def test_to_json_schema_with_definitions_non_root():
    definitions = Definitions({"User": String()})
    result = to_json_schema(definitions, _definitions={})
    assert "components" not in result
    assert "schemas" not in result

def test_to_json_schema_with_definitions_multiple_keys():
    definitions = Definitions({"User": String(), "Post": Integer()})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "string"}, "Post": {"type": "integer"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_reference():
    user_field = String()
    definitions = Definitions({"User": user_field})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "string"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_complex_field():
    complex_field = Union(any_of=[String(), Integer()])
    definitions = Definitions({"Complex": complex_field})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Complex": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}}
    assert result == expected

def test_to_json_schema_with_definitions_and_standard_properties():
    field_with_default = String(default="test")
    definitions = Definitions({"Field": field_with_default})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Field": {"type": "string", "default": "test"}}}}
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test_predicate_at_line_160_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=String(), else_clause=None)
    result = to_json_schema(field)
    assert "else" not in result


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_120_evaluates_to_true():
    from your_module import to_json_schema, Schema, Field
    schema_instance = Schema(allow_null=True, fields={})
    result = to_json_schema(schema_instance)
    assert isinstance(result, dict)
    assert result["type"] == ["object", "null"]
    schema_instance_no_null = Schema(allow_null=False, fields={})
    result_no_null = to_json_schema(schema_instance_no_null)
    assert result_no_null["type"] == "object"


# LLM-generated content at query #10
#--------------------------

def test_boolean_field_with_null_allowed():
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_158_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=None)
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_89_evaluates_to_true():
    field = Object(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["object", "null"]


# LLM-generated content at query #13
#--------------------------

def test_predicate_at_line_89_evaluates_to_true():
    obj = Object(allow_null=True)
    result = to_json_schema(obj)
    assert result["type"] == ["object", "null"]


# LLM-generated content at query #14
#--------------------------

def test_to_json_schema_with_definitions():
    definitions = Definitions({"User": String()})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "string"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_nested():
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}}}}
    assert result == expected

def test_to_json_schema_with_definitions_multiple():
    definitions = Definitions({"User": String(), "Age": Integer()})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "string"}, "Age": {"type": "integer"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_empty():
    definitions = Definitions()
    result = to_json_schema(definitions)
    expected = {}
    assert result == expected

def test_to_json_schema_with_definitions_and_field():
    definitions = Definitions({"User": String()})
    field = Integer()
    result = to_json_schema(field, _definitions=definitions)
    expected = {"type": "integer"}
    assert result == expected

def test_to_json_schema_with_definitions_reference():
    user_schema = Object(properties={"name": String()})
    definitions = Definitions({"User": user_schema})
    field = Reference(to="User", target=user_schema)
    result = to_json_schema(field, _definitions=definitions)
    expected = {"$ref": "#/components/schemas/User"}
    assert result == expected

def test_to_json_schema_with_definitions_union():
    definitions = Definitions({"User": String()})
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field, _definitions=definitions)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_definitions_object():
    definitions = Definitions({"User": String()})
    field = Object(properties={"user": Reference(to="User", target=String())})
    result = to_json_schema(field, _definitions=definitions)
    expected = {"type": "object", "properties": {"user": {"$ref": "#/components/schemas/User"}}}
    assert result == expected

def test_to_json_schema_with_definitions_array():
    definitions = Definitions({"User": String()})
    field = Array(items=Reference(to="User", target=String()))
    result = to_json_schema(field, _definitions=definitions)
    expected = {"type": "array", "items": {"$ref": "#/components/schemas/User"}}
    assert result == expected

def test_to_json_schema_with_definitions_choice():
    definitions = Definitions({"User": String()})
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field, _definitions=definitions)
    expected = {"enum": ["a", "b"]}
    assert result == expected

def test_to_json_schema_with_definitions_const():
    definitions = Definitions({"User": String()})
    field = Const(const="fixed")
    result = to_json_schema(field, _definitions=definitions)
    expected = {"const": "fixed"}
    assert result == expected

def test_to_json_schema_with_definitions_one_of():
    definitions = Definitions({"User": String()})
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field, _definitions=definitions)
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_definitions_all_of():
    definitions = Definitions({"User": String()})
    field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(field, _definitions=definitions)
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert result == expected

def test_to_json_schema_with_definitions_if_then_else():
    definitions = Definitions({"User": String()})
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(field, _definitions=definitions)
    expected = {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    assert result == expected

def test_to_json_schema_with_definitions_not():
    definitions = Definitions({"User": String()})
    field = Not(negated=String())
    result = to_json_schema(field, _definitions=definitions)
    expected = {"not": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_definitions_schema():
    definitions = Definitions({"User": String()})
    field = Schema(fields={"name": String()})
    result = to_json_schema(field, _definitions=definitions)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert result == expected

def test_to_json_schema_with_definitions_allow_null():
    definitions = Definitions({"User": String(allow_null=True)})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": ["string", "null"]}}}}
    assert result == expected

def test_to_json_schema_with_definitions_default():
    definitions = Definitions({"User": String(default="guest")})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "string", "default": "guest"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_min_length():
    definitions = Definitions({"User": String(min_length=5)})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "string", "minLength": 5}}}}
    assert result == expected

def test_to_json_schema_with_definitions_max_length():
    definitions = Definitions({"User": String(max_length=10)})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "string", "maxLength": 10}}}}
    assert result == expected

def test_to_json_schema_with_definitions_pattern():
    definitions = Definitions({"User": String(pattern=r"^\d+$")})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "string", "pattern": r"^\d+$"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_format():
    definitions = Definitions({"User": String(format="email")})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"User": {"type": "string", "format": "email"}}}}
    assert result == expected

def test_to_json_schema_with_definitions_integer():
    definitions = Definitions({"Age": Integer(minimum=0, maximum=120)})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Age": {"type": "integer", "minimum": 0, "maximum": 120}}}}
    assert result == expected

def test_to_json_schema_with_definitions_float():
    definitions = Definitions({"Price": Float(minimum=0.0, exclusive_maximum=100.0)})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Price": {"type": "number", "minimum": 0.0, "exclusiveMaximum": 100.0}}}}
    assert result == expected

def test_to_json_schema_with_definitions_boolean():
    definitions = Definitions({"Active": Boolean(default=True)})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Active": {"type": "boolean", "default": True}}}}
    assert result == expected

def test_to_json_schema_with_definitions_array_min_max():
    definitions = Definitions({"Tags": Array(min_items=1, max_items=5)})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Tags": {"type": "array", "minItems": 1, "maxItems": 5}}}}
    assert result == expected

def test_to_json_schema_with_definitions_array_unique():
    definitions = Definitions({"Tags": Array(unique_items=True)})
    result = to_json_schema(definitions)
    expected = {"components": {"schemas": {"Tags": {"type": "array", "uniqueItems": True}}}}
    assert result == expected

def test_to_json_schema_with_definitions_object_additional_props():
    definitions = Definitions({"Meta": Object(additional_properties=True)})
    result = to_json_schema(definitions)
    expected = {"components":


# LLM-generated content at query #15
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

def test_to_json_schema_with_object_field_required():
    field = Object(properties={"name": String()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    assert result == expected

def test_to_json_schema_with_object_field_additional_properties_false():
    field = Object(additional_properties=False)
    result = to_json_schema(field)
    expected = {"type": "object", "additionalProperties": False}
    assert result == expected

def test_to_json_schema_with_object_field_min_properties():
    field = Object(min_properties=1)
    result = to_json_schema(field)
    expected = {"type": "object", "minProperties": 1}
    assert result == expected

def test_to_json_schema_with_object_field_max_properties():
    field = Object(max_properties=10)
    result = to_json_schema(field)
    expected = {"type": "object", "maxProperties": 10}
    assert result == expected

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    expected = {"enum": ["a", "b"]}
    assert result == expected

def test_to_json_schema_with_const_field():
    field = Const(const="fixed")
    result = to_json_schema(field)
    expected = {"const": "fixed"}
    assert result == expected

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_oneof_field():
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_allof_field():
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(field)
    expected = {"if": {"type": "string"}, "then": {"type": "integer"}}
    assert result == expected

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    result = to_json_schema(field)
    expected = {"not": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_default_value():
    field = String(default="default")
    result = to_json_schema(field)
    expected = {"type": "string", "default": "default"}
    assert result == expected

def test_to_json_schema_with_definitions():
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result


