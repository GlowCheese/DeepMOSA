####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

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
    result = from_json_schema(data, definitions=definitions)
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

def test_from_json_schema_type_null():
    data = {"type": "null"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const is None

def test_from_json_schema_type_multiple():
    data = {"type": ["string", "integer"]}
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
    data = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_anyOf():
    data = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)

def test_from_json_schema_oneOf():
    data = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)

def test_from_json_schema_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)

def test_from_json_schema_if_then_else():
    data = {"if": {"type": "string"}, "then": {"minLength": 5}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_multiple_constraints():
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_no_constraints():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_with_components():
    data = {"components": {"schemas": {"Test": {"type": "string"}}}}
    result = from_json_schema(data)
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "string"}, definitions=definitions)
    assert isinstance(result, Any)

def test_from_json_schema_allow_null():
    data = {"type": ["string", "null"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert result.allow_null == True

def test_from_json_schema_type_only_null():
    data = {"type": "null"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const is None

def test_from_json_schema_type_no_valid_types():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)


# LLM-generated content at query #2
#--------------------------

def test_if_then_else_from_json_schema_basic():
    data = {"if": {"type": "string"}, "then": {"type": "string", "minLength": 5}, "else": {"type": "integer"}}
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("hello")
    assert result == "hello"
    try:
        field.validate("hi")
        assert False
    except Exception as e:
        assert True
    result = field.validate(42)
    assert result == 42

def test_if_then_else_from_json_schema_without_then():
    data = {"if": {"type": "string"}, "else": {"type": "integer"}}
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("any_string")
    assert result == "any_string"
    result = field.validate(123)
    assert result == 123

def test_if_then_else_from_json_schema_without_else():
    data = {"if": {"type": "string"}, "then": {"type": "string", "maxLength": 10}}
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("short")
    assert result == "short"
    result = field.validate(3.14)
    assert result == 3.14

def test_if_then_else_from_json_schema_with_default():
    data = {"if": {"type": "boolean"}, "then": {"const": True}, "else": {"const": False}, "default": False}
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate(True)
    assert result == True
    result = field.validate(False)
    assert result == False

def test_if_then_else_from_json_schema_nested():
    data = {"if": {"type": "object", "properties": {"x": {"type": "integer"}}}, "then": {"required": ["x"]}, "else": {"type": "array"}}
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate({"x": 5})
    assert result == {"x": 5}
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

def test_if_then_else_from_json_schema_with_ref_in_if():
    definitions = Definitions()
    definitions["#/components/schemas/MyString"] = from_json_schema({"type": "string"}, definitions)
    data = {"if": {"$ref": "#/components/schemas/MyString"}, "then": {"minLength": 1}, "else": {"type": "number"}}
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("a")
    assert result == "a"
    result = field.validate(100)
    assert result == 100

def test_if_then_else_from_json_schema_complex_condition():
    data = {"if": {"allOf": [{"type": "string"}, {"pattern": "^[A-Z]+$"}]}, "then": {"type": "string", "maxLength": 10}, "else": {"type": "null"}}
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("HELLO")
    assert result == "HELLO"
    result = field.validate(None)
    assert result == None

def test_if_then_else_from_json_schema_boolean_schema():
    data = {"if": True, "then": {"type": "string"}, "else": False}
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("test")
    assert result == "test"
    try:
        field.validate(123)
        assert False
    except Exception as e:
        assert True

def test_if_then_else_from_json_schema_empty_then_and_else():
    data = {"if": {"type": "boolean"}}
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate(True)
    assert result == True
    result = field.validate("anything")
    assert result == "anything"

def test_if_then_else_from_json_schema_with_any_of_in_if():
    data = {"if": {"anyOf": [{"type": "string"}, {"type": "number"}]}, "then": {"type": "string"}, "else": {"type": "array"}}
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    result = field.validate("text")
    assert result == "text"
    result = field.validate([1, 2])
    assert result == [1, 2]


# LLM-generated content at query #3
#--------------------------

def test_to_json_schema_with_string_field():
    field = String(title="Name", description="A name", allow_null=True, default="John")
    result = to_json_schema(field)
    expected = {"type": ["string", "null"], "default": "John", "title": "Name", "description": "A name"}
    assert result == expected

def test_to_json_schema_with_integer_field():
    field = Integer(minimum=0, maximum=100, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "integer", "minimum": 0, "maximum": 100}
    assert result == expected

def test_to_json_schema_with_boolean_field():
    field = Boolean(default=False, allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["boolean", "null"], "default": False}
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
    field = Union(any_of=[String(), Integer()], allow_null=False)
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
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
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")], default="a")
    result = to_json_schema(field)
    expected = {"enum": ["a", "b"], "default": "a"}
    assert result == expected

def test_to_json_schema_with_const_field():
    field = Const(const=42, allow_null=False)
    result = to_json_schema(field)
    expected = {"const": 42}
    assert result == expected

def test_to_json_schema_with_allof_field():
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)], allow_null=False)
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string", "minLength": 1}, {"type": "string", "maxLength": 10}]}
    assert result == expected

def test_to_json_schema_with_oneof_field():
    field = OneOf(one_of=[String(), Integer()], allow_null=False)
    result = to_json_schema(field)
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean(), allow_null=False)
    result = to_json_schema(field)
    expected = {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    assert result == expected

def test_to_json_schema_with_not_field():
    field = Not(negated=String(), allow_null=False)
    result = to_json_schema(field)
    expected = {"not": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_any_field():
    field = Any()
    result = to_json_schema(field)
    assert result == True

def test_to_json_schema_with_nevermatch_field():
    field = NeverMatch()
    result = to_json_schema(field)
    assert result == False

def test_to_json_schema_with_decimal_field():
    field = Decimal(minimum=0.0, maximum=1.0, allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0}
    assert result == expected

def test_to_json_schema_with_float_field():
    field = Float(exclusive_minimum=0.0, exclusive_maximum=10.0, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 10.0}
    assert result == expected

def test_to_json_schema_with_array_field_with_additional_items():
    field = Array(items=String(), additional_items=Integer(), allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "array", "items": {"type": "string"}, "additionalItems": {"type": "integer"}}
    assert result == expected

def test_to_json_schema_with_object_field_with_additional_properties():
    field = Object(additional_properties=String(), allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "object", "additionalProperties": {"type": "string"}}
    assert result == expected

def test_to_json_schema_with_object_field_with_pattern_properties():
    field = Object(pattern_properties={"^[a-z]+$": String()}, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "object", "patternProperties": {"^[a-z]+$": {"type": "string"}}}
    assert result == expected

def test_to_json_schema_with_object_field_with_property_names():
    field = Object(property_names=String(pattern_regex=re.compile("^[a-z]+$")), allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "object", "propertyNames": {"type": "string", "pattern": "^[a-z]+$"}}
    assert result == expected

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String()}, required=["name"], allow_null=True)
    result = to_json_schema(field)
    expected = {"type": ["object", "null"], "properties": {"name": {"type": "string"}}, "required": ["name"]}
    assert result == expected

def test_to_json_schema_with_string_field_with_pattern():
    field = String(pattern_regex=re.compile("^[A-Z]+$"), allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "string", "pattern": "^[A-Z]+$"}
    assert result == expected

def test_to_json_schema_with_string_field_with_format():
    field = String(format="email", allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "string", "format": "email"}
    assert result == expected

def test_to_json_schema_with_string_field_with_min_max_length():
    field = String(min_length=5, max_length=10, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "string", "minLength": 5, "maxLength": 10}
    assert result == expected

def test_to_json_schema_with_integer_field_with_multiple_of():
    field = Integer(multiple_of=2, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "integer", "multipleOf": 2}
    assert result == expected

def test_to_json_schema_with_array_field_with_unique_items():
    field = Array(unique_items=True, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "array", "uniqueItems": True}
    assert result == expected

def test_to_json_schema_with_object_field_with_min_max_properties():
    field = Object(min_properties=1, max_properties=5, allow_null=False)
    result = to_json_schema(field)
    expected = {"type": "object", "minProperties": 1, "maxProperties": 5}
    assert result == expected

def test_to_json_schema_with_union_field_with_null_allowed():
    field = Union(any_of=[String(allow_null=True), Integer()], allow_null=False)
    result =


# LLM-generated content at query #4
#--------------------------

def test_array_field_with_allow_null_sets_type_to_array_and_null():
    field = Array(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["array", "null"]


# LLM-generated content at query #5
#--------------------------

def test_if_then_else_without_then_clause():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=String())
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #6
#--------------------------

def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    definitions = Definitions()
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5
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
    assert field.allow_blank == False

def test_from_json_schema_type_boolean():
    data = {"default": True}
    definitions = Definitions()
    field = from_json_schema_type(data, "boolean", True, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True
    assert field.allow_null == True
    assert field.coerce_types == False

def test_from_json_schema_type_array_with_items_list():
    definitions = Definitions()
    items_schema = {"type": "string"}
    data = {"items": [items_schema, items_schema], "additionalItems": False, "minItems": 2, "maxItems": 2, "uniqueItems": True, "default": ["a", "b"]}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert len(field.items) == 2
    assert isinstance(field.items[0], String)
    assert field.additional_items == False
    assert field.min_items == 2
    assert field.max_items == 2
    assert field.unique_items == True
    assert field.default == ["a", "b"]
    assert field.allow_null == False

def test_from_json_schema_type_array_with_items_single():
    definitions = Definitions()
    items_schema = {"type": "integer"}
    data = {"items": items_schema, "additionalItems": True, "minItems": 0, "maxItems": 10, "uniqueItems": False, "default": [1, 2]}
    field = from_json_schema_type(data, "array", True, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, Integer)
    assert field.additional_items == True
    assert field.min_items == 0
    assert field.max_items == 10
    assert field.unique_items == False
    assert field.default == [1, 2]
    assert field.allow_null == True

def test_from_json_schema_type_array_without_items():
    data = {"minItems": 0, "maxItems": None, "additionalItems": True, "uniqueItems": False, "default": []}
    definitions = Definitions()
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.items is None
    assert field.additional_items == True
    assert field.min_items == 0
    assert field.max_items is None
    assert field.unique_items == False
    assert field.default == []
    assert field.allow_null == False

def test_from_json_schema_type_object_with_properties():
    definitions = Definitions()
    prop_schema = {"type": "string"}
    data = {"properties": {"name": prop_schema}, "patternProperties": {"^test_": prop_schema}, "additionalProperties": False, "propertyNames": {"pattern": "^[a-z]+$"}, "minProperties": 1, "maxProperties": 5, "required": ["name"], "default": {"name": "john"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert "^test_" in field.pattern_properties
    assert isinstance(field.pattern_properties["^test_"], String)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.property_names.pattern == "^[a-z]+$"
    assert field.min_properties == 1
    assert field.max_properties == 5
    assert field.required == ["name"]
    assert field.default == {"name": "john"}
    assert field.allow_null == False

def test_from_json_schema_type_object_without_properties():
    data = {"minProperties": 0, "maxProperties": None, "additionalProperties": True, "required": [], "default": {}}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", True, definitions)
    assert isinstance(field, Object)
    assert field.properties == {}
    assert field.pattern_properties == {}
    assert field.additional_properties == True
    assert field.property_names is None
    assert field.min_properties == 0
    assert field.max_properties is None
    assert field.required == []
    assert field.default == {}
    assert field.allow_null == True

def test_from_json_schema_type_object_with_additional_properties_field():
    definitions = Definitions()
    additional_schema = {"type": "number"}
    data = {"additionalProperties": additional_schema, "default": {"extra": 42}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.additional_properties, Float)
    assert field.default == {"extra": 42}
    assert field.allow_null == False


# LLM-generated content at query #7
#--------------------------

```python
def test_from_json_schema_with_ref_returns_ref_field():
    data = {"$ref": "#/components/schemas/User"}
    definitions = Definitions()
    definitions["#/components/schemas/User"] = Any()
    result = from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"


# LLM-generated content at query #8
#--------------------------

def test_ref_from_json_schema_with_valid_ref():
    definitions = Definitions()
    definitions["#/components/schemas/User"] = "UserSchema"
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, definitions)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is definitions

def test_ref_from_json_schema_raises_assertion_for_non_hash_ref():
    definitions = Definitions()
    data = {"$ref": "http://example.com/schema"}
    try:
        ref_from_json_schema(data, definitions)
        assert False
    except AssertionError as e:
        assert str(e) == "Unsupported $ref style in document."

def test_ref_from_json_schema_creates_reference_with_correct_target():
    definitions = Definitions()
    mock_field = "MockField"
    definitions["#/definitions/Item"] = mock_field
    data = {"$ref": "#/definitions/Item"}
    reference = ref_from_json_schema(data, definitions)
    assert reference.target == mock_field

def test_ref_from_json_schema_passes_definitions_to_reference():
    definitions = Definitions({"#/definitions/Test": "TestSchema"})
    data = {"$ref": "#/definitions/Test"}
    reference = ref_from_json_schema(data, definitions)
    assert reference.definitions is definitions
    assert reference.to == "#/definitions/Test"


# LLM-generated content at query #9
#--------------------------

```python
def test_additional_properties_none_returns_none():
    from typesystem.json_schema import from_json_schema_type
    from typesystem.schemas import Definitions
    data = {"additionalProperties": None}
    definitions = Definitions()
    field = from_json_schema_type(data, "object", False, definitions)
    assert field.additional_properties is None


# LLM-generated content at query #10
#--------------------------

def test_array_field_with_list_items():
    from my_module import Array, String, to_json_schema
    field = Array(items=[String(), String(allow_null=True)])
    result = to_json_schema(field)
    assert isinstance(result, dict)
    assert "items" in result
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    assert result["items"][0] == {"type": "string"}
    assert result["items"][1] == {"type": ["string", "null"]}


# LLM-generated content at query #11
#--------------------------

def test_pattern_regex_flags_unicode():
    pattern = re.compile(r"^test$", re.UNICODE)
    field = String(pattern_regex=pattern)
    result = to_json_schema(field)
    assert "pattern" in result
    assert result["pattern"] == r"^test$"

def test_pattern_regex_flags_non_unicode_raises():
    pattern = re.compile(r"^test$", re.IGNORECASE)
    field = String(pattern_regex=pattern)
    try:
        to_json_schema(field)
        assert False
    except ValueError as e:
        assert "non-standard flags" in str(e)


# LLM-generated content at query #12
#--------------------------

def test_from_json_schema_type_number():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 4}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 4
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_number_nullable():
    data = {}
    field = from_json_schema_type(data, "number", True, Definitions())
    assert isinstance(field, Float)
    assert field.allow_null == True
    assert field.coerce_types == False

def test_from_json_schema_type_integer():
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 4}
    field = from_json_schema_type(data, "integer", False, Definitions())
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
    data = {"minLength": 5, "maxLength": 10, "format": "email", "pattern": "^a.*z$", "default": "abc"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^a.*z$"
    assert field.default == "abc"
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_string_allow_blank():
    data = {"minLength": 0}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.allow_blank == True
    assert field.min_length == None

def test_from_json_schema_type_boolean():
    data = {"default": True}
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_array_no_items():
    data = {"minItems": 0, "maxItems": 10, "uniqueItems": True, "default": []}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert field.min_items == 0
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == []
    assert field.allow_null == False
    assert field.items == None
    assert field.additional_items == True

def test_from_json_schema_type_array_with_items_single():
    definitions = Definitions()
    data = {"items": {"type": "string"}}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, Field)

def test_from_json_schema_type_array_with_items_list():
    definitions = Definitions()
    data = {"items": [{"type": "string"}, {"type": "number"}]}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, list)
    assert len(field.items) == 2

def test_from_json_schema_type_array_additional_items_bool():
    data = {"additionalItems": False}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert field.additional_items == False

def test_from_json_schema_type_array_additional_items_field():
    definitions = Definitions()
    data = {"additionalItems": {"type": "integer"}}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.additional_items, Field)

def test_from_json_schema_type_object_no_properties():
    data = {"minProperties": 0, "maxProperties": 10, "required": ["id"], "default": {}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert field.min_properties == 0
    assert field.max_properties == 10
    assert field.required == ["id"]
    assert field.default == {}
    assert field.allow_null == False
    assert field.properties == {}
    assert field.pattern_properties == {}
    assert field.additional_properties == None
    assert field.property_names == None

def test_from_json_schema_type_object_with_properties():
    definitions = Definitions()
    data = {"properties": {"name": {"type": "string"}}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties, dict)
    assert "name" in field.properties

def test_from_json_schema_type_object_with_pattern_properties():
    definitions = Definitions()
    data = {"patternProperties": {"^x_": {"type": "string"}}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.pattern_properties, dict)
    assert "^x_" in field.pattern_properties

def test_from_json_schema_type_object_additional_properties_bool():
    data = {"additionalProperties": False}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert field.additional_properties == False

def test_from_json_schema_type_object_additional_properties_field():
    definitions = Definitions()
    data = {"additionalProperties": {"type": "string"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.additional_properties, Field)

def test_from_json_schema_type_object_with_property_names():
    definitions = Definitions()
    data = {"propertyNames": {"pattern": "^[a-z]+$"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.property_names, Field)

def test_from_json_schema_type_invalid_type_string():
    try:
        from_json_schema_type({}, "invalid", False, Definitions())
        assert False
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid'"


# LLM-generated content at query #13
#--------------------------

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
    assert result.allow_null == False

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

def test_from_json_schema_enum():
    data = {"enum": [1, 2, 3]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)

def test_from_json_schema_const():
    data = {"const": "fixed"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == "fixed"

def test_from_json_schema_allOf():
    data = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_anyOf():
    data = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)

def test_from_json_schema_oneOf():
    data = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)

def test_from_json_schema_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)

def test_from_json_schema_if_then_else():
    data = {"if": {"type": "string"}, "then": {"minLength": 1}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_multiple_constraints():
    data = {"type": "string", "minLength": 5}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)

def test_from_json_schema_no_constraints():
    data = {}
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

def test_from_json_schema_components_definitions():
    data = {"components": {"schemas": {"Test": {"type": "string"}}}}
    result = from_json_schema(data)
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "string"}, definitions)
    assert isinstance(result, Any)

def test_from_json_schema_default_definitions():
    data = {"$ref": "#/components/schemas/Test", "components": {"schemas": {"Test": {"type": "string"}}}}
    result = from_json_schema(data)
    assert isinstance(result, Reference)


# LLM-generated content at query #14
#--------------------------

def test_predicate_at_line_160_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=String(), else_clause=None)
    result = to_json_schema(field)
    assert "else" not in result


# LLM-generated content at query #15
#--------------------------

```python
def test_from_json_schema_type_object_without_property_names():
    from typesystem.json_schema import from_json_schema_type
    from typesystem.schemas import Definitions
    definitions = Definitions()
    data = {}
    result = from_json_schema_type(data, "object", False, definitions)
    assert result.property_names is None


# LLM-generated content at query #16
#--------------------------

def test_additional_items_is_false():
    field = Array(additional_items=False)
    result = to_json_schema(field)
    assert "additionalItems" not in result


# LLM-generated content at query #17
#--------------------------

```python
def test_property_names_is_none():
    data = {}
    type_string = "object"
    allow_null = False
    definitions = Definitions()
    field = from_json_schema_type(data, type_string, allow_null, definitions)
    assert isinstance(field, Object)
    assert field.property_names is None


# LLM-generated content at query #18
#--------------------------

def test_to_json_schema_with_union_field():
    from typesystem.fields import Union, String, Integer
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

def test_to_json_schema_with_union_field_and_allow_null():
    from typesystem.fields import Union, String, Integer
    string_field = String(allow_null=True)
    union_field = Union(any_of=[string_field, Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == ["string", "null"]
    assert result["anyOf"][1]["type"] == "integer"

def test_to_json_schema_with_union_field_and_default():
    from typesystem.fields import Union, String, Integer
    union_field = Union(any_of=[String(default="test"), Integer(default=42)])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][0]["default"] == "test"
    assert result["anyOf"][1]["type"] == "integer"
    assert result["anyOf"][1]["default"] == 42

def test_to_json_schema_with_union_field_and_definitions():
    from typesystem.fields import Union, String, Integer
    from typesystem.schemas import Definitions
    union_field = Union(any_of=[String(), Integer()])
    definitions = Definitions({"MyUnion": union_field})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "MyUnion" in result["components"]["schemas"]
    assert "anyOf" in result["components"]["schemas"]["MyUnion"]
    assert len(result["components"]["schemas"]["MyUnion"]["anyOf"]) == 2

def test_to_json_schema_with_union_field_and_nested_definitions():
    from typesystem.fields import Union, String, Integer, Reference
    from typesystem.schemas import Definitions
    string_field = String()
    integer_field = Integer()
    union_field = Union(any_of=[string_field, integer_field])
    definitions = Definitions({"MyString": string_field, "MyUnion": union_field})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "MyString" in result["components"]["schemas"]
    assert "MyUnion" in result["components"]["schemas"]
    assert "anyOf" in result["components"]["schemas"]["MyUnion"]
    assert len(result["components"]["schemas"]["MyUnion"]["anyOf"]) == 2

def test_to_json_schema_with_union_field_and_reference():
    from typesystem.fields import Union, String, Integer, Reference
    from typesystem.schemas import Definitions
    string_field = String()
    integer_field = Integer()
    union_field = Union(any_of=[Reference(to="MyString"), integer_field])
    definitions = Definitions({"MyString": string_field, "MyUnion": union_field})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "MyString" in result["components"]["schemas"]
    assert "MyUnion" in result["components"]["schemas"]
    assert result["components"]["schemas"]["MyUnion"]["anyOf"][0]["$ref"] == "#/components/schemas/MyString"
    assert result["components"]["schemas"]["MyUnion"]["anyOf"][1]["type"] == "integer"

def test_to_json_schema_with_union_field_and_allow_null_on_child():
    from typesystem.fields import Union, String, Integer
    string_field = String(allow_null=True)
    integer_field = Integer()
    union_field = Union(any_of=[string_field, integer_field])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == ["string", "null"]
    assert result["anyOf"][1]["type"] == "integer"

def test_to_json_schema_with_union_field_and_allow_null_on_all_children():
    from typesystem.fields import Union, String, Integer
    string_field = String(allow_null=True)
    integer_field = Integer(allow_null=True)
    union_field = Union(any_of=[string_field, integer_field])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == ["string", "null"]
    assert result["anyOf"][1]["type"] == ["integer", "null"]

def test_to_json_schema_with_union_field_and_no_children():
    from typesystem.fields import Union
    union_field = Union(any_of=[])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert result["anyOf"] == []

def test_to_json_schema_with_union_field_and_single_child():
    from typesystem.fields import Union, String
    union_field = Union(any_of=[String()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 1
    assert result["anyOf"][0]["type"] == "string"

def test_to_json_schema_with_union_field_and_multiple_children():
    from typesystem.fields import Union, String, Integer, Boolean
    union_field = Union(any_of=[String(), Integer(), Boolean()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 3
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"
    assert result["anyOf"][2]["type"] == "boolean"

def test_to_json_schema_with_union_field_and_complex_children():
    from typesystem.fields import Union, String, Integer, Array
    array_field = Array(items=String())
    union_field = Union(any_of=[String(), Integer(), array_field])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 3
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"
    assert result["anyOf"][2]["type"] == "array"
    assert result["anyOf"][2]["items"]["type"] == "string"

def test_to_json_schema_with_union_field_and_object_child():
    from typesystem.fields import Union, String, Object
    object_field = Object(properties={"name": String()})
    union_field = Union(any_of=[String(), object_field])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "object"
    assert "properties" in result["anyOf"][1]
    assert "name" in result["anyOf"][1]["properties"]

def test_to_json_schema_with_union_field_and_schema_child():
    from typesystem.fields import Union, String, Schema
    schema_field = Schema(fields={"name": String()})
    union_field = Union(any_of=[String(), schema_field])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "object"
    assert "properties" in result["anyOf"][1]
    assert "name" in result["anyOf"][1]["properties"]

def test_to_json_schema_with_union_field_and_choice_child():
    from typesystem.fields import Union, String, Choice
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    union_field = Union(any_of=[String(), choice_field])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert "enum" in result["anyOf"][1]
    assert result["anyOf"][1]["enum"] == ["a", "b"]

def test_to_json_schema_with_union_field_and_const_child():
    from typesystem.fields import Union, String, Const



# LLM-generated content at query #19
#--------------------------

def test_to_json_schema_with_field_default():
    field = Field(default="test_default")
    result = to_json_schema(field)
    expected = {"default": "test_default"}
    assert result == expected

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

def test_to_json_schema_with_object_field_required():
    field = Object(properties={"name": String()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
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
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    expected = {"const": "fixed_value"}
    assert result == expected

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert result == expected

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(),


# LLM-generated content at query #20
#--------------------------

```python
def test_from_json_schema_object_without_property_names():
    data = {}
    type_string = "object"
    allow_null = False
    definitions = {}
    result = from_json_schema_type(data, type_string, allow_null, definitions)
    assert result.property_names is None


# LLM-generated content at query #21
#--------------------------

def test_pattern_regex_flags_unicode():
    import re
    from some_module import String, to_json_schema
    pattern = re.compile(r"^test$", re.UNICODE)
    field = String(pattern_regex=pattern)
    result = to_json_schema(field)
    assert "pattern" in result
    assert result["pattern"] == r"^test$"


# LLM-generated content at query #22
#--------------------------

def test_from_json_schema_type_array_with_items_list():
    from typesystem.json_schema import from_json_schema_type
    from typesystem.schemas import Definitions
    definitions = Definitions()
    data = {"items": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result.items, list)
    assert len(result.items) == 2


# LLM-generated content at query #23
#--------------------------

def test_predicate_at_line_120_evaluates_to_true():
    from your_module import Schema, to_json_schema
    schema_instance = Schema(allow_null=True)
    result = to_json_schema(schema_instance)
    assert result["type"] == ["object", "null"]
    schema_instance_not_null = Schema(allow_null=False)
    result_not_null = to_json_schema(schema_instance_not_null)
    assert result_not_null["type"] == "object"


# LLM-generated content at query #24
#--------------------------

def test_predicate_at_line_160_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=String(), else_clause=None)
    result = to_json_schema(field)
    assert "else" not in result


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_type_from_json_schema_with_single_type():
    definitions = Definitions()
    data = {"type": "string"}
    field = type_from_json_schema(data, definitions)
    validated = field.validate("hello")
    assert validated == "hello"

def test_type_from_json_schema_with_nullable_single_type():
    definitions = Definitions()
    data = {"type": ["string", "null"]}
    field = type_from_json_schema(data, definitions)
    validated = field.validate(None)
    assert validated is None

def test_type_from_json_schema_with_multiple_types():
    definitions = Definitions()
    data = {"type": ["string", "integer"]}
    field = type_from_json_schema(data, definitions)
    validated_string = field.validate("hello")
    validated_integer = field.validate(123)
    assert validated_string == "hello"
    assert validated_integer == 123

def test_type_from_json_schema_with_nullable_multiple_types():
    definitions = Definitions()
    data = {"type": ["string", "integer", "null"]}
    field = type_from_json_schema(data, definitions)
    validated_null = field.validate(None)
    validated_string = field.validate("hello")
    validated_integer = field.validate(123)
    assert validated_null is None
    assert validated_string == "hello"
    assert validated_integer == 123

def test_type_from_json_schema_with_number_and_integer():
    definitions = Definitions()
    data = {"type": ["number", "integer"]}
    field = type_from_json_schema(data, definitions)
    validated_number = field.validate(3.14)
    validated_integer = field.validate(42)
    assert validated_number == 3.14
    assert validated_integer == 42

def test_type_from_json_schema_with_no_type_specified():
    definitions = Definitions()
    data = {}
    field = type_from_json_schema(data, definitions)
    validated_null = field.validate(None)
    validated_bool = field.validate(True)
    validated_object = field.validate({})
    validated_array = field.validate([])
    validated_number = field.validate(3.14)
    validated_string = field.validate("hello")
    assert validated_null is None
    assert validated_bool is True
    assert validated_object == {}
    assert validated_array == []
    assert validated_number == 3.14
    assert validated_string == "hello"

def test_type_from_json_schema_with_only_null():
    definitions = Definitions()
    data = {"type": "null"}
    field = type_from_json_schema(data, definitions)
    validated = field.validate(None)
    assert validated is None

def test_type_from_json_schema_with_only_null_and_allow_null_false():
    definitions = Definitions()
    data = {"type": "null"}
    field = type_from_json_schema(data, definitions)
    try:
        field.validate("not null")
        assert False
    except ValidationError:
        pass

def test_type_from_json_schema_with_constraints():
    definitions = Definitions()
    data = {"type": "integer", "minimum": 0, "maximum": 10}
    field = type_from_json_schema(data, definitions)
    validated = field.validate(5)
    assert validated == 5

def test_type_from_json_schema_with_invalid_constraint():
    definitions = Definitions()
    data = {"type": "integer", "minimum": 0, "maximum": 10}
    field = type_from_json_schema(data, definitions)
    try:
        field.validate(15)
        assert False
    except ValidationError:
        pass

def test_type_from_json_schema_with_array_type():
    definitions = Definitions()
    data = {"type": "array", "items": {"type": "string"}}
    field = type_from_json_schema(data, definitions)
    validated = field.validate(["hello", "world"])
    assert validated == ["hello", "world"]

def test_type_from_json_schema_with_object_type():
    definitions = Definitions()
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    field = type_from_json_schema(data, definitions)
    validated = field.validate({"name": "Alice"})
    assert validated == {"name": "Alice"}

def test_type_from_json_schema_with_boolean_type():
    definitions = Definitions()
    data = {"type": "boolean"}
    field = type_from_json_schema(data, definitions)
    validated_true = field.validate(True)
    validated_false = field.validate(False)
    assert validated_true is True
    assert validated_false is False

def test_type_from_json_schema_with_number_type():
    definitions = Definitions()
    data = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    field = type_from_json_schema(data, definitions)
    validated = field.validate(0.5)
    assert validated == 0.5

def test_type_from_json_schema_with_string_type():
    definitions = Definitions()
    data = {"type": "string", "minLength": 1, "maxLength": 10}
    field = type_from_json_schema(data, definitions)
    validated = field.validate("hello")
    assert validated == "hello"

def test_type_from_json_schema_with_empty_type_strings():
    definitions = Definitions()
    data = {"type": []}
    field = type_from_json_schema(data, definitions)
    validated_null = field.validate(None)
    validated_bool = field.validate(True)
    validated_object = field.validate({})
    validated_array = field.validate([])
    validated_number = field.validate(3.14)
    validated_string = field.validate("hello")
    assert validated_null is None
    assert validated_bool is True
    assert validated_object == {}
    assert validated_array == []
    assert validated_number == 3.14
    assert validated_string == "hello"

def test_type_from_json_schema_with_type_null_in_union():
    definitions = Definitions()
    data = {"type": ["null", "string"]}
    field = type_from_json_schema(data, definitions)
    validated_null = field.validate(None)
    validated_string = field.validate("hello")
    assert validated_null is None
    assert validated_string == "hello"

def test_type_from_json_schema_with_type_removal_of_integer_when_number_present():
    definitions = Definitions()
    data = {"type": ["number", "integer"]}
    field = type_from_json_schema(data, definitions)
    validated_number = field.validate(3.14)
    validated_integer = field.validate(42)
    assert validated_number == 3.14
    assert validated_integer == 42

def test_type_from_json_schema_with_invalid_type_string():
    definitions = Definitions()
    data = {"type": "invalid"}
    try:
        type_from_json_schema(data, definitions)
        assert False
    except AssertionError:
        pass


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
    field = Object(properties={"name": String()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
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
    field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    field = IfThenElse(if_clause=String(), then_clause=Const(const="valid"))
    result = to_json_schema(field)
    expected = {"if":


# LLM-generated content at query #3
#--------------------------

def test_from_json_schema_type_number():
    definitions = Definitions()
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5.0}
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5.0
    assert field.coerce_types == False

def test_from_json_schema_type_integer():
    definitions = Definitions()
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5
    assert field.coerce_types == False

def test_from_json_schema_type_string():
    definitions = Definitions()
    data = {"minLength": 5, "maxLength": 10, "format": "email", "pattern": "^[a-z]+$", "default": "hello"}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-z]+$"
    assert field.default == "hello"
    assert field.coerce_types == False

def test_from_json_schema_type_boolean():
    definitions = Definitions()
    data = {"default": True}
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True
    assert field.coerce_types == False

def test_from_json_schema_type_object():
    definitions = Definitions()
    data = {"properties": {"name": {"type": "string"}}, "patternProperties": {"^x_": {"type": "integer"}}, "additionalProperties": False, "propertyNames": {"pattern": "^[a-z]+$"}, "minProperties": 1, "maxProperties": 5, "required": ["name"], "default": {"name": "test"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.pattern_properties["^x_"], Integer)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.property_names.pattern == "^[a-z]+$"
    assert field.min_properties == 1
    assert field.max_properties == 5
    assert field.required == ["name"]
    assert field.default == {"name": "test"}

def test_from_json_schema_type_array():
    definitions = Definitions()
    data = {"items": {"type": "string"}, "additionalItems": False, "minItems": 1, "maxItems": 5, "uniqueItems": True, "default": ["a", "b"]}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items == False
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items == True
    assert field.default == ["a", "b"]

def test_from_json_schema_type_with_allow_null():
    definitions = Definitions()
    data = {}
    field = from_json_schema_type(data, "string", True, definitions)
    assert isinstance(field, String)
    assert field.allow_null == True

def test_from_json_schema_type_with_no_default():
    definitions = Definitions()
    data = {}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.has_default() == False

def test_from_json_schema_type_with_min_length_zero():
    definitions = Definitions()
    data = {"minLength": 0}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.allow_blank == True
    assert field.min_length == None

def test_from_json_schema_type_with_min_length_one():
    definitions = Definitions()
    data = {"minLength": 1}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.allow_blank == False
    assert field.min_length == None

def test_from_json_schema_type_with_min_length_greater_than_one():
    definitions = Definitions()
    data = {"minLength": 5}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.allow_blank == False
    assert field.min_length == 5

def test_from_json_schema_type_with_additional_properties_field():
    definitions = Definitions()
    data = {"additionalProperties": {"type": "string"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.additional_properties, String)

def test_from_json_schema_type_with_additional_properties_bool():
    definitions = Definitions()
    data = {"additionalProperties": True}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert field.additional_properties == True

def test_from_json_schema_type_with_additional_properties_none():
    definitions = Definitions()
    data = {"additionalProperties": None}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert field.additional_properties == None

def test_from_json_schema_type_with_items_list():
    definitions = Definitions()
    data = {"items": [{"type": "string"}, {"type": "integer"}]}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, list)
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Integer)

def test_from_json_schema_type_with_items_single():
    definitions = Definitions()
    data = {"items": {"type": "string"}}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)

def test_from_json_schema_type_with_additional_items_field():
    definitions = Definitions()
    data = {"additionalItems": {"type": "string"}}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.additional_items, String)

def test_from_json_schema_type_with_additional_items_bool():
    definitions = Definitions()
    data = {"additionalItems": False}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.additional_items == False

def test_from_json_schema_type_with_no_items():
    definitions = Definitions()
    data = {}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.items == None

def test_from_json_schema_type_with_no_additional_items():
    definitions = Definitions()
    data = {}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.additional_items == True


# LLM-generated content at query #4
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
    result = from_json_schema(data, definitions=definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/Test"

def test_from_json_schema_with_const():
    data = {"const": 42}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == 42

def test_from_json_schema_with_enum():
    data = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]

def test_from_json_schema_with_allOf():
    data = {"allOf": [{"type": "string"}, {"type": "string", "maxLength": 5}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_from_json_schema_with_anyOf():
    data = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

def test_from_json_schema_with_oneOf():
    data = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2

def test_from_json_schema_with_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)

def test_from_json_schema_with_if_then_else():
    data = {"if": {"type": "string"}, "then": {"type": "string", "maxLength": 5}, "else": {"type": "number"}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)

def test_from_json_schema_with_type_string():
    data = {"type": "string"}
    result = from_json_schema(data)
    assert isinstance(result, String)

def test_from_json_schema_with_type_number():
    data = {"type": "number"}
    result = from_json_schema(data)
    assert isinstance(result, Number)

def test_from_json_schema_with_type_integer():
    data = {"type": "integer"}
    result = from_json_schema(data)
    assert isinstance(result, Integer)

def test_from_json_schema_with_type_boolean():
    data = {"type": "boolean"}
    result = from_json_schema(data)
    assert isinstance(result, Boolean)

def test_from_json_schema_with_type_array():
    data = {"type": "array", "items": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Array)

def test_from_json_schema_with_type_object():
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = from_json_schema(data)
    assert isinstance(result, Object)

def test_from_json_schema_with_multiple_types():
    data = {"type": ["string", "number"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

def test_from_json_schema_with_null_type():
    data = {"type": "null"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const is None

def test_from_json_schema_with_null_in_multiple_types():
    data = {"type": ["string", "null"]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert result.allow_null == True
    assert len(result.any_of) == 1

def test_from_json_schema_with_no_type():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_with_multiple_constraints():
    data = {"type": "string", "enum": ["a", "b"], "maxLength": 5}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_from_json_schema_with_components_schemas():
    data = {"components": {"schemas": {"Test": {"type": "string"}}}}
    result = from_json_schema(data)
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "string"}, definitions=definitions)
    assert "#/components/schemas/Test" in definitions

def test_from_json_schema_with_default_value():
    data = {"type": "string", "default": "hello"}
    result = from_json_schema(data)
    assert result.default == "hello"


# LLM-generated content at query #5
#--------------------------

def test_additional_items_is_bool_false():
    field = Array(additional_items=False)
    result = to_json_schema(field)
    assert result["additionalItems"] == False


# LLM-generated content at query #6
#--------------------------

def test_additional_items_is_none_so_additional_items_argument_is_true():
    data = {"items": {"type": "string"}, "additionalItems": None}
    definitions = Definitions()
    field = from_json_schema(data, definitions=definitions)
    assert field.additional_items is True


# LLM-generated content at query #7
#--------------------------

def test_pattern_properties_present():
    field = Object(pattern_properties={"^test.*": String()})
    result = to_json_schema(field)
    assert "patternProperties" in result
    assert "^test.*" in result["patternProperties"]
    assert result["patternProperties"]["^test.*"]["type"] == "string"


# LLM-generated content at query #8
#--------------------------

def test_predicate_at_line_22_evaluates_to_true():
    from typing import Union
    from my_module import to_json_schema, Reference, Field, Definitions
    field = Reference(to="test_ref", target=Field())
    result = to_json_schema(field)
    assert isinstance(field, Reference)


# LLM-generated content at query #9
#--------------------------

def test_from_json_schema_type_number():
    definitions = Definitions()
    data = {"minimum": 0.0, "maximum": 10.0, "exclusiveMinimum": 0.0, "exclusiveMaximum": 10.0, "multipleOf": 2.0, "default": 4.0}
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0.0
    assert field.maximum == 10.0
    assert field.exclusive_minimum == 0.0
    assert field.exclusive_maximum == 10.0
    assert field.multiple_of == 2.0
    assert field.default == 4.0
    assert field.allow_null == False
    assert field.coerce_types == False

def test_from_json_schema_type_integer():
    definitions = Definitions()
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 4}
    field = from_json_schema_type(data, "integer", True, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 4
    assert field.allow_null == True
    assert field.coerce_types == False

def test_from_json_schema_type_string():
    definitions = Definitions()
    data = {"minLength": 5, "maxLength": 10, "format": "email", "pattern": "^[a-z]+$", "default": "hello"}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-z]+$"
    assert field.default == "hello"
    assert field.allow_null == False
    assert field.coerce_types == False
    assert field.allow_blank == False

def test_from_json_schema_type_string_allow_blank():
    definitions = Definitions()
    data = {"minLength": 0}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.allow_blank == True

def test_from_json_schema_type_boolean():
    definitions = Definitions()
    data = {"default": True}
    field = from_json_schema_type(data, "boolean", True, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True
    assert field.allow_null == True
    assert field.coerce_types == False

def test_from_json_schema_type_object():
    definitions = Definitions()
    data = {"properties": {"name": {"type": "string"}}, "patternProperties": {"^x-": {"type": "string"}}, "additionalProperties": False, "propertyNames": {"pattern": "^[a-z]+$"}, "minProperties": 1, "maxProperties": 5, "required": ["name"], "default": {"name": "test"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties, dict)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.pattern_properties, dict)
    assert "^x-" in field.pattern_properties
    assert isinstance(field.pattern_properties["^x-"], String)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.property_names.pattern == "^[a-z]+$"
    assert field.min_properties == 1
    assert field.max_properties == 5
    assert field.required == ["name"]
    assert field.default == {"name": "test"}
    assert field.allow_null == False

def test_from_json_schema_type_object_no_properties():
    definitions = Definitions()
    data = {}
    field = from_json_schema_type(data, "object", True, definitions)
    assert isinstance(field, Object)
    assert field.properties == {}
    assert field.pattern_properties == {}
    assert field.additional_properties == None
    assert field.property_names == None
    assert field.min_properties == None
    assert field.max_properties == None
    assert field.required == []
    assert field.allow_null == True

def test_from_json_schema_type_array():
    definitions = Definitions()
    data = {"items": {"type": "string"}, "additionalItems": False, "minItems": 1, "maxItems": 10, "uniqueItems": True, "default": ["a", "b"]}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items == False
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["a", "b"]
    assert field.allow_null == False

def test_from_json_schema_type_array_items_list():
    definitions = Definitions()
    data = {"items": [{"type": "string"}, {"type": "integer"}]}
    field = from_json_schema_type(data, "array", True, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, list)
    assert len(field.items) == 2
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Integer)

def test_from_json_schema_type_array_no_items():
    definitions = Definitions()
    data = {}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.items == None
    assert field.additional_items == True

def test_from_json_schema_type_invalid_type_string():
    definitions = Definitions()
    data = {}
    try:
        from_json_schema_type(data, "invalid", False, definitions)
        assert False
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid'"


# LLM-generated content at query #10
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
    field = Object(properties={"name": String()}, required=["name"])
    result = to_json_schema(field)
    expected = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
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
    field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    field = IfThenElse(if_clause=String(), then_clause=Const(const="valid"))
    result = to_json_schema(field)
    expected = {"if":


# LLM-generated content at query #11
#--------------------------

def test_additional_items_is_not_bool():
    definitions = Definitions()
    data = {"additionalItems": {"type": "string"}}
    type_string = "array"
    allow_null = False
    field = from_json_schema_type(data, type_string, allow_null, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.additional_items, Field)


# LLM-generated content at query #12
#--------------------------

def test_from_json_schema_type_array_with_items_list():
    from typesystem.json_schema import from_json_schema_type
    from typesystem.schemas import Definitions
    data = {"items": [{"type": "string"}, {"type": "integer"}]}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result.items, list)
    assert len(result.items) == 2


# LLM-generated content at query #13
#--------------------------

def test_max_properties_included_when_set():
    from my_module import Object, to_json_schema
    field = Object(max_properties=5)
    result = to_json_schema(field)
    assert "maxProperties" in result
    assert result["maxProperties"] == 5


# LLM-generated content at query #14
#--------------------------

def test_additional_items_is_not_bool_and_not_none():
    field = Array(additional_items=String())
    result = to_json_schema(field)
    assert isinstance(result.get("additionalItems"), dict)


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
    field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(field)
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert result == expected

def test_to_json_schema_with_ifthenelse_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json


# LLM-generated content at query #16
#--------------------------

def test_predicate_at_line_158_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=String())
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #17
#--------------------------

```python
def test_from_json_schema_type_array_with_items_as_list():
    from typesystem.schemas import Definitions
    from typesystem.json_schema import from_json_schema_type
    data = {"items": [{"type": "string"}, {"type": "integer"}]}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result.items, list)
    assert len(result.items) == 2


# LLM-generated content at query #18
#--------------------------

```python
def test_additional_items_is_bool():
    data = {"additionalItems": True}
    definitions = {}
    result = from_json_schema(data, definitions=definitions)
    assert isinstance(result, Array)
    assert result.additional_items is True


