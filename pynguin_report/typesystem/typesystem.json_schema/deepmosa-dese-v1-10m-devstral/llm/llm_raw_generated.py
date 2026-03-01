####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    result = to_json_schema(field)
    assert result == {"type": "number"}

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

def test_to_json_schema_with_array_field_min_items():
    field = Array(min_items=2)
    result = to_json_schema(field)
    assert result == {"type": "array", "minItems": 2}

def test_to_json_schema_with_array_field_max_items():
    field = Array(max_items=5)
    result = to_json_schema(field)
    assert result == {"type": "array", "maxItems": 5}

def test_to_json_schema_with_array_field_items():
    field = Array(items=String())
    result = to_json_schema(field)
    assert result == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_array_field_additional_items():
    field = Array(additional_items=Boolean())
    result = to_json_schema(field)
    assert result == {"type": "array", "additionalItems": {"type": "boolean"}}

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

def test_to_json_schema_with_object_field_pattern_properties():
    field = Object(pattern_properties={"^S_": String(), "^I_": Integer()})
    result = to_json_schema(field)
    assert result == {"type": "object", "patternProperties": {"^S_": {"type": "string"}, "^I_": {"type": "integer"}}}

def test_to_json_schema_with_object_field_additional_properties():
    field = Object(additional_properties=Boolean())
    result = to_json_schema(field)
    assert result == {"type": "object", "additionalProperties": {"type": "boolean"}}

def test_to_json_schema_with_object_field_property_names():
    field = Object(property_names=String())
    result = to_json_schema(field)
    assert result == {"type": "object", "propertyNames": {"type": "string"}}

def test_to_json_schema_with_object_field_max_properties():
    field = Object(max_properties=5)
    result = to_json_schema(field)
    assert result == {"type": "object", "maxProperties": 5}

def test_to_json_schema_with_object_field_min_properties():
    field = Object(min_properties=2)
    result = to_json_schema(field)
    assert result == {"type": "object", "minProperties": 2}

def test_to_json_schema_with_object_field_required():
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

def test_to_json_schema_with_schema_field_required():
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert result == {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

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
    target_field = String()
    field = Reference(to="string_ref", target=target_field)
    result = to_json_schema(field)
    assert result == {"$ref": "#/components/schemas/string_ref", "components": {"schemas": {"string_ref": {"type": "string"}}}}


# LLM-generated content at query #2
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {"type": "number", "minimum": 0, "maximum": 100, "default": 50}
    definitions = Definitions()
    result = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.default == 50
    assert result.allow_null is False

def test_from_json_schema_type_integer():
    data = {"type": "integer", "minimum": 0, "maximum": 100, "default": 50}
    definitions = Definitions()
    result = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.default == 50
    assert result.allow_null is False

def test_from_json_schema_type_string():
    data = {"type": "string", "minLength": 5, "maxLength": 10, "default": "hello"}
    definitions = Definitions()
    result = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.default == "hello"
    assert result.allow_null is False
    assert result.allow_blank is False

def test_from_json_schema_type_boolean():
    data = {"type": "boolean", "default": True}
    definitions = Definitions()
    result = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(result, Boolean)
    assert result.default is True
    assert result.allow_null is False

def test_from_json_schema_type_array():
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5, "default": ["a", "b"]}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    assert result.min_items == 1
    assert result.max_items == 5
    assert result.default == ["a", "b"]
    assert result.allow_null is False

def test_from_json_schema_type_object():
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "default": {"name": "test"}
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert isinstance(result.properties["name"], String)
    assert result.required == ["name"]
    assert result.default == {"name": "test"}
    assert result.allow_null is False


# LLM-generated content at query #3
#--------------------------

```python
def test_schema_field_type_assignment():
    field = Schema(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["object", "null"]


# LLM-generated content at query #4
#--------------------------

```python
def test_if_then_else_from_json_schema_with_all_clauses():
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"}
    }
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, Field)
    assert isinstance(field.then_clause, Field)
    assert isinstance(field.else_clause, Field)

def test_if_then_else_from_json_schema_without_then_and_else():
    data = {
        "if": {"type": "string"}
    }
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, Field)
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Any)

def test_if_then_else_from_json_schema_without_else():
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"}
    }
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, Field)
    assert isinstance(field.then_clause, Field)
    assert isinstance(field.else_clause, Any)

def test_if_then_else_from_json_schema_without_then():
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"}
    }
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, Field)
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Field)

def test_if_then_else_from_json_schema_with_default():
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"},
        "default": "default_value"
    }
    definitions = Definitions()
    field = if_then_else_from_json_schema(data, definitions)
    assert field.default == "default_value"


# LLM-generated content at query #5
#--------------------------

```python
def test_multiple_of_predicate():
    field = Integer(multiple_of=5)
    result = to_json_schema(field)
    assert "multipleOf" in result
    assert result["multipleOf"] == 5


# LLM-generated content at query #6
#--------------------------

```python
def test_string_field_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]


# LLM-generated content at query #7
#--------------------------

```python
def test_field_items_is_list_or_tuple():
    field = Array(items=[String(), Integer()])
    assert isinstance(field.items, (list, tuple))


# LLM-generated content at query #8
#--------------------------

```python
def test_one_of_from_json_schema():
    definitions = Definitions()
    data = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Float)
    assert result.default == NO_DEFAULT

def test_one_of_from_json_schema_with_default():
    definitions = Definitions()
    data = {"oneOf": [{"type": "string"}], "default": "test"}
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert result.default == "test"


# LLM-generated content at query #9
#--------------------------

```python
def test_from_json_schema_type_additional_properties_none():
    data = {}
    type_string = "object"
    allow_null = False
    definitions = Definitions()
    result = from_json_schema_type(data, type_string, allow_null, definitions)
    assert result.additional_properties is None


# LLM-generated content at query #10
#--------------------------

```python
def test_from_json_schema_with_bool_true():
    result = from_json_schema(True)
    assert isinstance(result, Any)
    assert result.validate(None) is None

def test_from_json_schema_with_bool_false():
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)
    with pytest.raises(ValidationError):
        result.validate(None)

def test_from_json_schema_with_ref():
    data = {"$ref": "#/components/schemas/test"}
    definitions = Definitions()
    definitions["#/components/schemas/test"] = String()
    result = from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.validate("test") == "test"

def test_from_json_schema_with_type_constraint():
    data = {"type": "string"}
    result = from_json_schema(data)
    assert isinstance(result, String)
    assert result.validate("test") == "test"

def test_from_json_schema_with_enum():
    data = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)
    assert result.validate("a") == "a"
    with pytest.raises(ValidationError):
        result.validate("d")

def test_from_json_schema_with_const():
    data = {"const": "test"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.validate("test") == "test"
    with pytest.raises(ValidationError):
        result.validate("other")

def test_from_json_schema_with_all_of():
    data = {"allOf": [{"type": "string"}, {"type": "string", "minLength": 5}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert result.validate("test") == "test"
    with pytest.raises(ValidationError):
        result.validate("a")

def test_from_json_schema_with_any_of():
    data = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert result.validate("test") == "test"
    assert result.validate(1) == 1
    with pytest.raises(ValidationError):
        result.validate(None)

def test_from_json_schema_with_one_of():
    data = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)
    assert result.validate("test") == "test"
    with pytest.raises(ValidationError):
        result.validate(1)

def test_from_json_schema_with_not():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)
    assert result.validate(1) == 1
    with pytest.raises(ValidationError):
        result.validate("test")

def test_from_json_schema_with_if_then_else():
    data = {"if": {"type": "string"}, "then": {"type": "string", "minLength": 5}, "else": {"type": "number"}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)
    assert result.validate("test") == "test"
    assert result.validate(1) == 1
    with pytest.raises(ValidationError):
        result.validate("a")

def test_from_json_schema_with_multiple_constraints():
    data = {"type": "string", "enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert result.validate("a") == "a"
    with pytest.raises(ValidationError):
        result.validate("d")

def test_from_json_schema_with_no_constraints():
    result = from_json_schema({})
    assert isinstance(result, Any)
    assert result.validate(None) is None


# LLM-generated content at query #11
#--------------------------

```python
def test_array_field_type_assignment():
    field = Array(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["array", "null"]


# LLM-generated content at query #12
#--------------------------

```python
def test_if_then_else_with_none_else_clause():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=None)
    result = to_json_schema(field)
    assert "else" not in result


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_from_json_schema_type_pattern_properties_none():
    data = {}
    type_string = "object"
    allow_null = False
    definitions = Definitions()
    result = from_json_schema_type(data, type_string, allow_null, definitions)
    assert result.pattern_properties is None


# LLM-generated content at query #15
#--------------------------

```python
def test_from_json_schema_type_array_additional_items_none():
    data = {"type": "array"}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert result.additional_items is True


# LLM-generated content at query #16
#--------------------------

```python
def test_from_json_schema_type_property_names_not_none():
    data = {"type": "object", "propertyNames": {"type": "string"}}
    definitions = {}
    result = from_json_schema_type(data, "object", False, definitions)
    assert result.property_names is not None


# LLM-generated content at query #17
#--------------------------

```python
def test_from_json_schema_type_with_list_items():
    data = {"items": [{"type": "string"}]}
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result.items, list)
    assert len(result.items) == 1
    assert isinstance(result.items[0], String)


# LLM-generated content at query #18
#--------------------------

```python
def test_to_json_schema_with_never_match_field():
    assert to_json_schema(NeverMatch()) is False


# LLM-generated content at query #19
#--------------------------

```python
def test_to_json_schema_with_any_field():
    result = to_json_schema(Any())
    assert result is True

def test_to_json_schema_with_never_match_field():
    result = to_json_schema(NeverMatch())
    assert result is False

def test_to_json_schema_with_string_field():
    field = String(min_length=1, max_length=10, pattern_regex=re.compile("^[a-z]+$"), format="email")
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
    field = Choice(choices=[("a", "A"), ("b", "B")])
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
    field = Reference(to="test", target=String())
    result = to_json_schema(field)
    assert result == {
        "$ref": "#/components/schemas/test",
        "components": {"schemas": {"test": {"type": "string"}}}
    }

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

def test_to_json_schema_with_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["string", "null"]}

def test_to_json_schema_with_default_value():
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_unknown_field_type():
    class UnknownField(Field):
        pass
    with pytest.raises(ValueError, match="Cannot convert field type 'UnknownField' to JSON Schema"):
        to_json_schema(UnknownField())


# LLM-generated content at query #20
#--------------------------

```python
def test_to_json_schema_with_any_field():
    schema = to_json_schema(Any())
    assert schema is True

def test_to_json_schema_with_never_match_field():
    schema = to_json_schema(NeverMatch())
    assert schema is False

def test_to_json_schema_with_string_field():
    field = String(min_length=1, max_length=10, pattern_regex=re.compile(r"^[a-z]+$"), format="email")
    schema = to_json_schema(field)
    assert schema == {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "format": "email"
    }

def test_to_json_schema_with_integer_field():
    field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=True, multiple_of=2)
    schema = to_json_schema(field)
    assert schema == {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True,
        "multipleOf": 2
    }

def test_to_json_schema_with_float_field():
    field = Float(minimum=0.0, maximum=100.0, exclusive_minimum=True, exclusive_maximum=True, multiple_of=0.5)
    schema = to_json_schema(field)
    assert schema == {
        "type": "number",
        "minimum": 0.0,
        "maximum": 100.0,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True,
        "multipleOf": 0.5
    }

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    schema = to_json_schema(field)
    assert schema == {"type": "boolean"}

def test_to_json_schema_with_array_field():
    field = Array(min_items=1, max_items=10, items=String(), additional_items=False, unique_items=True)
    schema = to_json_schema(field)
    assert schema == {
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
    schema = to_json_schema(field)
    assert schema == {
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
    schema = to_json_schema(field)
    assert schema == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    schema = to_json_schema(field)
    assert schema == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="fixed_value")
    schema = to_json_schema(field)
    assert schema == {"const": "fixed_value"}

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
    assert schema == {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    schema = to_json_schema(field)
    assert schema == {"not": {"type": "string"}}

def test_to_json_schema_with_reference_field():
    target = String()
    field = Reference(to="test", target=target)
    schema = to_json_schema(field)
    assert schema == {"$ref": "#/components/schemas/test"}

def test_to_json_schema_with_definitions():
    definitions = Definitions()
    definitions["test"] = String()
    schema = to_json_schema(definitions)
    assert schema == {"components": {"schemas": {"test": {"type": "string"}}}}

def test_to_json_schema_with_allow_null():
    field = String(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["string", "null"]}

def test_to_json_schema_with_default_value():
    field = String(default="default_value")
    schema = to_json_schema(field)
    assert schema == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_invalid_field_type():
    with pytest.raises(ValueError):
        to_json_schema("invalid_field_type")


# LLM-generated content at query #21
#--------------------------

```python
def test_definitions_iteration():
    arg = {"key1": "value1", "key2": "value2"}
    _definitions = None
    result = to_json_schema(arg, _definitions)
    assert isinstance(result, dict)


# LLM-generated content at query #22
#--------------------------

```python
def test_ref_from_json_schema_with_valid_reference():
    data = {"$ref": "#/definitions/valid"}
    definitions = Definitions({"valid": "dummy"})
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/valid"
    assert result.definitions == definitions

def test_ref_from_json_schema_with_invalid_reference():
    data = {"$ref": "invalid"}
    definitions = Definitions({"valid": "dummy"})
    try:
        ref_from_json_schema(data, definitions)
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "Unsupported $ref style in document."


# LLM-generated content at query #23
#--------------------------

```python
def test_type_from_json_schema_with_single_type():
    data = {"type": "string"}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, String)
    assert field.allow_null is False

def test_type_from_json_schema_with_nullable_single_type():
    data = {"type": ["string", "null"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, String)
    assert field.allow_null is True

def test_type_from_json_schema_with_multiple_types():
    data = {"type": ["string", "number"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2
    assert isinstance(field.any_of[0], String)
    assert isinstance(field.any_of[1], Float)
    assert field.allow_null is False

def test_type_from_json_schema_with_nullable_multiple_types():
    data = {"type": ["string", "number", "null"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2
    assert isinstance(field.any_of[0], String)
    assert isinstance(field.any_of[1], Float)
    assert field.allow_null is True

def test_type_from_json_schema_with_no_type():
    data = {}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, NeverMatch)

def test_type_from_json_schema_with_nullable_no_type():
    data = {"type": ["null"]}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Const)
    assert field.const is None

def test_type_from_json_schema_with_object_type():
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)

def test_type_from_json_schema_with_array_type():
    data = {"type": "array", "items": {"type": "string"}}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)

def test_type_from_json_schema_with_integer_type():
    data = {"type": "integer"}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Integer)
    assert field.allow_null is False

def test_type_from_json_schema_with_boolean_type():
    data = {"type": "boolean"}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Boolean)
    assert field.allow_null is False

def test_type_from_json_schema_with_number_type():
    data = {"type": "number"}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Float)
    assert field.allow_null is False

def test_type_from_json_schema_with_pattern_properties():
    data = {"type": "object", "patternProperties": {"^S_": {"type": "string"}}}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Object)
    assert "patternProperties" in data
    assert "^S_" in field.pattern_properties
    assert isinstance(field.pattern_properties["^S_"], String)

def test_type_from_json_schema_with_additional_properties():
    data = {"type": "object", "additionalProperties": {"type": "string"}}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.additional_properties, String)

def test_type_from_json_schema_with_property_names():
    data = {"type": "object", "propertyNames": {"type": "string"}}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.property_names, String)


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_at_line_158_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=Integer())
    result = to_json_schema(field)
    assert "then" not in result


# LLM-generated content at query #25
#--------------------------

```python
def test_to_json_schema_with_any_field():
    assert to_json_schema(Any()) == True

def test_to_json_schema_with_never_match_field():
    assert to_json_schema(NeverMatch()) == False

def test_to_json_schema_with_string_field():
    field = String()
    assert to_json_schema(field) == {"type": "string"}

def test_to_json_schema_with_nullable_string_field():
    field = String(allow_null=True)
    assert to_json_schema(field) == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_with_default():
    field = String(default="default_value")
    assert to_json_schema(field) == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_string_field_with_min_length():
    field = String(min_length=5)
    assert to_json_schema(field) == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_with_max_length():
    field = String(max_length=10)
    assert to_json_schema(field) == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_with_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    assert to_json_schema(field) == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_with_format():
    field = String(format="email")
    assert to_json_schema(field) == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    assert to_json_schema(field) == {"type": "integer"}

def test_to_json_schema_with_nullable_integer_field():
    field = Integer(allow_null=True)
    assert to_json_schema(field) == {"type": ["integer", "null"]}

def test_to_json_schema_with_integer_field_with_minimum():
    field = Integer(minimum=0)
    assert to_json_schema(field) == {"type": "integer", "minimum": 0}

def test_to_json_schema_with_integer_field_with_maximum():
    field = Integer(maximum=100)
    assert to_json_schema(field) == {"type": "integer", "maximum": 100}

def test_to_json_schema_with_integer_field_with_exclusive_minimum():
    field = Integer(exclusive_minimum=0)
    assert to_json_schema(field) == {"type": "integer", "exclusiveMinimum": 0}

def test_to_json_schema_with_integer_field_with_exclusive_maximum():
    field = Integer(exclusive_maximum=100)
    assert to_json_schema(field) == {"type": "integer", "exclusiveMaximum": 100}

def test_to_json_schema_with_integer_field_with_multiple_of():
    field = Integer(multiple_of=2)
    assert to_json_schema(field) == {"type": "integer", "multipleOf": 2}

def test_to_json_schema_with_float_field():
    field = Float()
    assert to_json_schema(field) == {"type": "number"}

def test_to_json_schema_with_nullable_float_field():
    field = Float(allow_null=True)
    assert to_json_schema(field) == {"type": ["number", "null"]}

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    assert to_json_schema(field) == {"type": "number"}

def test_to_json_schema_with_nullable_decimal_field():
    field = Decimal(allow_null=True)
    assert to_json_schema(field) == {"type": ["number", "null"]}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    assert to_json_schema(field) == {"type": "boolean"}

def test_to_json_schema_with_nullable_boolean_field():
    field = Boolean(allow_null=True)
    assert to_json_schema(field) == {"type": ["boolean", "null"]}

def test_to_json_schema_with_array_field():
    field = Array()
    assert to_json_schema(field) == {"type": "array"}

def test_to_json_schema_with_nullable_array_field():
    field = Array(allow_null=True)
    assert to_json_schema(field) == {"type": ["array", "null"]}

def test_to_json_schema_with_array_field_with_min_items():
    field = Array(min_items=1)
    assert to_json_schema(field) == {"type": "array", "minItems": 1}

def test_to_json_schema_with_array_field_with_max_items():
    field = Array(max_items=10)
    assert to_json_schema(field) == {"type": "array", "maxItems": 10}

def test_to_json_schema_with_array_field_with_items():
    field = Array(items=String())
    assert to_json_schema(field) == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_array_field_with_additional_items():
    field = Array(additional_items=String())
    assert to_json_schema(field) == {"type": "array", "additionalItems": {"type": "string"}}

def test_to_json_schema_with_array_field_with_unique_items():
    field = Array(unique_items=True)
    assert to_json_schema(field) == {"type": "array", "uniqueItems": True}

def test_to_json_schema_with_object_field():
    field = Object()
    assert to_json_schema(field) == {"type": "object"}

def test_to_json_schema_with_nullable_object_field():
    field = Object(allow_null=True)
    assert to_json_schema(field) == {"type": ["object", "null"]}

def test_to_json_schema_with_object_field_with_properties():
    field = Object(properties={"name": String()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_object_field_with_pattern_properties():
    field = Object(pattern_properties={r"^[a-z]+$": String()})
    assert to_json_schema(field) == {"type": "object", "patternProperties": {"^[a-z]+$": {"type": "string"}}}

def test_to_json_schema_with_object_field_with_additional_properties():
    field = Object(additional_properties=String())
    assert to_json_schema(field) == {"type": "object", "additionalProperties": {"type": "string"}}

def test_to_json_schema_with_object_field_with_property_names():
    field = Object(property_names=String())
    assert to_json_schema(field) == {"type": "object", "propertyNames": {"type": "string"}}

def test_to_json_schema_with_object_field_with_max_properties():
    field = Object(max_properties=10)
    assert to_json_schema(field) == {"type": "object", "maxProperties": 10}

def test_to_json_schema_with_object_field_with_min_properties():
    field = Object(min_properties=1)
    assert to_json_schema(field) == {"type": "object", "minProperties": 1}

def test_to_json_schema_with_object_field_with_required():
    field = Object(required=["name"])
    assert to_json_schema(field) == {"type": "object", "required": ["name"]}

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_nullable_schema_field():
    field = Schema(allow_null=True, fields={"name": String()})
    assert to_json_schema(field) == {"type": ["object", "null"], "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_schema_field_with_required():
    field = Schema(fields={"name": String()}, required=["name"])
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert to_json_schema(field) == {"enum": ["a", "b"]}

def test_to_json_schema_with_choice_field_with_default():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")], default="a")
    assert to_json_schema(field) == {"enum": ["a", "b"], "default": "a"}

def test_to_json_schema_with_const_field():
    field = Const(const="constant_value")
    assert to_json_schema(field) == {"const": "constant_value"}

def test_to_json_schema_with_const_field_with_default():
    field = Const(const="constant_value", default="constant_value")
    assert to_json_schema(field) == {"const": "constant_value", "default": "constant_value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    assert to_json_schema(field) == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_union_field_with_default():
    field = Union(any_of=[String(), Integer()], default="default")
    assert to_json_schema(field) == {"anyOf": [{"type": "string"}, {"type": "integer"}], "default": "default"}

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    assert to_json_schema(field) == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

def


# LLM-generated content at query #26
#--------------------------

```python
def test_if_clause_without_else_clause():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=None)
    result = to_json_schema(field)
    assert "else" not in result


# LLM-generated content at query #27
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

def test_to_json_schema_with_string_field_allow_null():
    field = String(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_min_length():
    field = String(min_length=5)
    schema = to_json_schema(field)
    assert schema == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_max_length():
    field = String(max_length=10)
    schema = to_json_schema(field)
    assert schema == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    schema = to_json_schema(field)
    assert schema == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_format():
    field = String(format="email")
    schema = to_json_schema(field)
    assert schema == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    schema = to_json_schema(field)
    assert schema == {"type": "integer"}

def test_to_json_schema_with_integer_field_allow_null():
    field = Integer(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["integer", "null"]}

def test_to_json_schema_with_integer_field_minimum():
    field = Integer(minimum=0)
    schema = to_json_schema(field)
    assert schema == {"type": "integer", "minimum": 0}

def test_to_json_schema_with_integer_field_maximum():
    field = Integer(maximum=100)
    schema = to_json_schema(field)
    assert schema == {"type": "integer", "maximum": 100}

def test_to_json_schema_with_float_field():
    field = Float()
    schema = to_json_schema(field)
    assert schema == {"type": "number"}

def test_to_json_schema_with_float_field_allow_null():
    field = Float(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["number", "null"]}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    schema = to_json_schema(field)
    assert schema == {"type": "boolean"}

def test_to_json_schema_with_boolean_field_allow_null():
    field = Boolean(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["boolean", "null"]}

def test_to_json_schema_with_array_field():
    field = Array()
    schema = to_json_schema(field)
    assert schema == {"type": "array"}

def test_to_json_schema_with_array_field_allow_null():
    field = Array(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["array", "null"]}

def test_to_json_schema_with_array_field_items():
    field = Array(items=String())
    schema = to_json_schema(field)
    assert schema == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_object_field():
    field = Object()
    schema = to_json_schema(field)
    assert schema == {"type": "object"}

def test_to_json_schema_with_object_field_allow_null():
    field = Object(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["object", "null"]}

def test_to_json_schema_with_object_field_properties():
    field = Object(properties={"name": String(), "age": Integer()})
    schema = to_json_schema(field)
    assert schema == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String(), "age": Integer()})
    schema = to_json_schema(field)
    assert schema == {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
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
    definitions = Definitions()
    definitions["string_field"] = String()
    schema = to_json_schema(definitions)
    assert schema == {"components": {"schemas": {"string_field": {"type": "string"}}}}


# LLM-generated content at query #28
#--------------------------

```python
def test_root_definitions_present():
    field = String()
    definitions = {"test": {"type": "string"}}
    result = to_json_schema(field, definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert result["components"]["schemas"] == definitions


# LLM-generated content at query #29
#--------------------------

```python
def test_array_field_type_predicate():
    field = Array(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["array", "null"]


# LLM-generated content at query #30
#--------------------------

```python
def test_from_json_schema_type_number():
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True,
        "multipleOf": 2,
        "default": 50
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(result, Float)
    assert result.allow_null is False
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.exclusive_minimum is True
    assert result.exclusive_maximum is True
    assert result.multiple_of == 2
    assert result.default == 50
    assert result.coerce_types is False

def test_from_json_schema_type_integer():
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True,
        "multipleOf": 2,
        "default": 50
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(result, Integer)
    assert result.allow_null is False
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.exclusive_minimum is True
    assert result.exclusive_maximum is True
    assert result.multiple_of == 2
    assert result.default == 50
    assert result.coerce_types is False

def test_from_json_schema_type_string():
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(result, String)
    assert result.allow_null is False
    assert result.allow_blank is False
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.format == "email"
    assert result.pattern == "^[a-zA-Z0-9]+$"
    assert result.default == "test"
    assert result.coerce_types is False

def test_from_json_schema_type_boolean():
    data = {
        "type": "boolean",
        "default": True
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(result, Boolean)
    assert result.allow_null is False
    assert result.default is True
    assert result.coerce_types is False

def test_from_json_schema_type_array():
    data = {
        "type": "array",
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(result, Array)
    assert result.allow_null is False
    assert result.min_items == 1
    assert result.max_items == 10
    assert result.additional_items is False
    assert isinstance(result.items, String)
    assert result.unique_items is True
    assert result.default == ["test"]

def test_from_json_schema_type_object():
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "patternProperties": {"^S_": {"type": "string"}},
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test"}
    }
    definitions = Definitions()
    result = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(result, Object)
    assert result.allow_null is False
    assert "name" in result.properties
    assert isinstance(result.properties["name"], String)
    assert "^S_" in result.pattern_properties
    assert isinstance(result.pattern_properties["^S_"], String)
    assert result.additional_properties is False
    assert isinstance(result.property_names, String)
    assert result.min_properties == 1
    assert result.max_properties == 10
    assert result.required == ["name"]
    assert result.default == {"name": "test"}

def test_from_json_schema_type_invalid_type_string():
    data = {}
    definitions = Definitions()
    try:
        from_json_schema_type(data, "invalid", False, definitions)
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid'"


# LLM-generated content at query #31
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

def test_to_json_schema_with_object_field():
    field = Object()
    result = to_json_schema(field)
    assert result == {"type": "object"}

def test_to_json_schema_with_object_field_allow_null():
    field = Object(allow_null=True)
    result = to_json_schema(field)
    assert result == {"type": ["object", "null"]}

def test_to_json_schema_with_schema_field():
    field = Schema()
    result = to_json_schema(field)
    assert result == {"type": "object"}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5, "uniqueItems": True, "default": ["test"]}
    result = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    assert result.min_items == 1
    assert result.max_items == 5
    assert result.unique_items is True
    assert result.default == ["test"]
    assert result.allow_null is False

def test_from_json_schema_type_object():
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "additionalProperties": False,
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    result = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert "age" in result.properties
    assert result.additional_properties is False
    assert result.required == ["name"]
    assert result.default == {"name": "John", "age": 30}
    assert result.allow_null is False

def test_from_json_schema_type_allow_null():
    data = {"type": "string", "minLength": 0}
    result = from_json_schema_type(data, "string", True, Definitions())
    assert isinstance(result, String)
    assert result.allow_null is True
    assert result.allow_blank is True

def test_from_json_schema_type_array_with_additional_items():
    data = {"type": "array", "items": [{"type": "string"}, {"type": "integer"}], "additionalItems": {"type": "boolean"}}
    result = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, list)
    assert len(result.items) == 2
    assert isinstance(result.additional_items, Boolean)

def test_from_json_schema_type_object_with_pattern_properties():
    data = {
        "type": "object",
        "patternProperties": {"^S_": {"type": "string"}, "^I_": {"type": "integer"}},
        "additionalProperties": False
    }
    result = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(result, Object)
    assert "S_" in result.pattern_properties
    assert "I_" in result.pattern_properties
    assert result.additional_properties is False

def test_from_json_schema_type_object_with_property_names():
    data = {
        "type": "object",
        "propertyNames": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
        "additionalProperties": True
    }
    result = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(result, Object)
    assert isinstance(result.property_names, String)
    assert result.property_names.pattern == "^[A-Za-z_][A-Za-z0-9_]*$"
    assert result.additional_properties is True


# LLM-generated content at query #2
#--------------------------

```python
def test_all_of_from_json_schema_basic():
    data = {"allOf": [{"type": "string"}, {"type": "number"}]}
    definitions = Definitions()
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], String)
    assert isinstance(result.all_of[1], Float)

def test_all_of_from_json_schema_with_default():
    data = {"allOf": [{"type": "integer"}], "default": 42}
    definitions = Definitions()
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert result.default == 42

def test_all_of_from_json_schema_empty():
    data = {"allOf": []}
    definitions = Definitions()
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 0

def test_all_of_from_json_schema_with_definitions():
    data = {"allOf": [{"$ref": "#/components/schemas/test"}]}
    definitions = Definitions()
    definitions["#/components/schemas/test"] = String()
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 1
    assert isinstance(result.all_of[0], Reference)


# LLM-generated content at query #3
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
    field = Array(additional_items=Integer())
    assert to_json_schema(field) == {"type": "array", "additionalItems": {"type": "integer"}}

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
    field = Object(pattern_properties={"^S_": String()})
    assert to_json_schema(field) == {"type": "object", "patternProperties": {"^S_": {"type": "string"}}}

def test_to_json_schema_with_object_field_additional_properties():
    field = Object(additional_properties=Integer())
    assert to_json_schema(field) == {"type": "object", "additionalProperties": {"type": "integer"}}

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
    field = Schema(fields={"name": String()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_schema_field_allow_null():
    field = Schema(allow_null=True, fields={"name": String()})
    assert to_json_schema(field) == {"type": ["object", "null"], "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_schema_field_required():
    field = Schema(fields={"name": String()}, required=["name"])
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    assert to_json_schema(field) == {"enum": ["a", "b"]}

def test_to_json_schema_with_choice_field_default():
    field = Choice(choices=[("a", "A"), ("b", "B")], default="a")
    assert to_json_schema(field) == {"enum": ["a", "b"], "default": "a"}

def test_to_json_schema_with_const_field():
    field = Const(const="value")
    assert to_json_schema(field) == {"const": "value"}

def test_to_json_schema_with_const_field_default():
    field = Const(const="value", default="value")
    assert to_json_schema(field) == {"const": "value", "default": "value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    assert to_json_schema(field) == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_union_field_default():
    field = Union(any_of=[String(), Integer()], default="default")
    assert to_json_schema(field) == {"anyOf": [{"type": "string"}, {"type": "integer"}], "default": "default"}

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

def test_to_json_schema_with_reference_field():
    target = String()
    field = Reference(to="ref", target=target)
    assert to_json_schema(field) == {"$ref": "#/components/schemas/ref", "components": {"schemas": {"ref": {"type": "string"}}}}


# LLM-generated content at query #4
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

def test_to_json_schema_with_float_field():
    field = Float()
    result = to_json_schema(field)
    assert result == {"type": "number"}

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

def test_to_json_schema_with_array_field_with_items():
    field = Array(items=String())
    result = to_json_schema(field)
    assert result == {"type": "array", "items": {"type": "string"}}

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


# LLM-generated content at query #5
#--------------------------

```python
def test_to_json_schema_with_any_field():
    assert to_json_schema(Any()) == True

def test_to_json_schema_with_never_match_field():
    assert to_json_schema(NeverMatch()) == False

def test_to_json_schema_with_string_field():
    field = String()
    assert to_json_schema(field) == {"type": "string"}

def test_to_json_schema_with_nullable_string_field():
    field = String(allow_null=True)
    assert to_json_schema(field) == {"type": ["string", "null"]}

def test_to_json_schema_with_string_field_with_default():
    field = String(default="default_value")
    assert to_json_schema(field) == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_string_field_with_min_length():
    field = String(min_length=5)
    assert to_json_schema(field) == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_with_max_length():
    field = String(max_length=10)
    assert to_json_schema(field) == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_with_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    assert to_json_schema(field) == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_with_format():
    field = String(format="email")
    assert to_json_schema(field) == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    assert to_json_schema(field) == {"type": "integer"}

def test_to_json_schema_with_nullable_integer_field():
    field = Integer(allow_null=True)
    assert to_json_schema(field) == {"type": ["integer", "null"]}

def test_to_json_schema_with_integer_field_with_minimum():
    field = Integer(minimum=0)
    assert to_json_schema(field) == {"type": "integer", "minimum": 0}

def test_to_json_schema_with_integer_field_with_maximum():
    field = Integer(maximum=100)
    assert to_json_schema(field) == {"type": "integer", "maximum": 100}

def test_to_json_schema_with_integer_field_with_exclusive_minimum():
    field = Integer(exclusive_minimum=0)
    assert to_json_schema(field) == {"type": "integer", "exclusiveMinimum": 0}

def test_to_json_schema_with_integer_field_with_exclusive_maximum():
    field = Integer(exclusive_maximum=100)
    assert to_json_schema(field) == {"type": "integer", "exclusiveMaximum": 100}

def test_to_json_schema_with_integer_field_with_multiple_of():
    field = Integer(multiple_of=2)
    assert to_json_schema(field) == {"type": "integer", "multipleOf": 2}

def test_to_json_schema_with_float_field():
    field = Float()
    assert to_json_schema(field) == {"type": "number"}

def test_to_json_schema_with_decimal_field():
    field = Decimal()
    assert to_json_schema(field) == {"type": "number"}

def test_to_json_schema_with_boolean_field():
    field = Boolean()
    assert to_json_schema(field) == {"type": "boolean"}

def test_to_json_schema_with_nullable_boolean_field():
    field = Boolean(allow_null=True)
    assert to_json_schema(field) == {"type": ["boolean", "null"]}

def test_to_json_schema_with_array_field():
    field = Array()
    assert to_json_schema(field) == {"type": "array"}

def test_to_json_schema_with_nullable_array_field():
    field = Array(allow_null=True)
    assert to_json_schema(field) == {"type": ["array", "null"]}

def test_to_json_schema_with_array_field_with_min_items():
    field = Array(min_items=1)
    assert to_json_schema(field) == {"type": "array", "minItems": 1}

def test_to_json_schema_with_array_field_with_max_items():
    field = Array(max_items=10)
    assert to_json_schema(field) == {"type": "array", "maxItems": 10}

def test_to_json_schema_with_array_field_with_items():
    field = Array(items=String())
    assert to_json_schema(field) == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_array_field_with_additional_items():
    field = Array(additional_items=String())
    assert to_json_schema(field) == {"type": "array", "additionalItems": {"type": "string"}}

def test_to_json_schema_with_array_field_with_unique_items():
    field = Array(unique_items=True)
    assert to_json_schema(field) == {"type": "array", "uniqueItems": True}

def test_to_json_schema_with_object_field():
    field = Object()
    assert to_json_schema(field) == {"type": "object"}

def test_to_json_schema_with_nullable_object_field():
    field = Object(allow_null=True)
    assert to_json_schema(field) == {"type": ["object", "null"]}

def test_to_json_schema_with_object_field_with_properties():
    field = Object(properties={"name": String()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_object_field_with_pattern_properties():
    field = Object(pattern_properties={"^S_": String()})
    assert to_json_schema(field) == {"type": "object", "patternProperties": {"^S_": {"type": "string"}}}

def test_to_json_schema_with_object_field_with_additional_properties():
    field = Object(additional_properties=String())
    assert to_json_schema(field) == {"type": "object", "additionalProperties": {"type": "string"}}

def test_to_json_schema_with_object_field_with_property_names():
    field = Object(property_names=String())
    assert to_json_schema(field) == {"type": "object", "propertyNames": {"type": "string"}}

def test_to_json_schema_with_object_field_with_max_properties():
    field = Object(max_properties=10)
    assert to_json_schema(field) == {"type": "object", "maxProperties": 10}

def test_to_json_schema_with_object_field_with_min_properties():
    field = Object(min_properties=1)
    assert to_json_schema(field) == {"type": "object", "minProperties": 1}

def test_to_json_schema_with_object_field_with_required():
    field = Object(required=["name"])
    assert to_json_schema(field) == {"type": "object", "required": ["name"]}

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_nullable_schema_field():
    field = Schema(allow_null=True, fields={"name": String()})
    assert to_json_schema(field) == {"type": ["object", "null"], "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_schema_field_with_required():
    field = Schema(fields={"name": String()}, required=["name"])
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert to_json_schema(field) == {"enum": ["a", "b"]}

def test_to_json_schema_with_choice_field_with_default():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")], default="a")
    assert to_json_schema(field) == {"enum": ["a", "b"], "default": "a"}

def test_to_json_schema_with_const_field():
    field = Const(const="constant_value")
    assert to_json_schema(field) == {"const": "constant_value"}

def test_to_json_schema_with_const_field_with_default():
    field = Const(const="constant_value", default="constant_value")
    assert to_json_schema(field) == {"const": "constant_value", "default": "constant_value"}

def test_to_json_schema_with_union_field():
    field = Union(any_of=[String(), Integer()])
    assert to_json_schema(field) == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_union_field_with_default():
    field = Union(any_of=[String(), Integer()], default="default")
    assert to_json_schema(field) == {"anyOf": [{"type": "string"}, {"type": "integer"}], "default": "default"}

def test_to_json_schema_with_one_of_field():
    field = OneOf(one_of=[String(), Integer()])
    assert to_json_schema(field) == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_all_of_field():
    field = AllOf(all_of=[String(), Integer()])
    assert to_json_schema(field) == {"allOf": [{"type": "string"}, {"type": "integer"}]}

def test_to_json_schema_with_if_then_else_field():
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_cl


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_46():
    field = Integer(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["integer", "null"]


# LLM-generated content at query #7
#--------------------------

```python
def test_string_field_allow_null():
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]


# LLM-generated content at query #8
#--------------------------

```python
def test_if_then_else_from_json_schema():
    definitions = Definitions()
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"},
        "default": 42
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Union)
    assert isinstance(result.then_clause, Union)
    assert isinstance(result.else_clause, Union)
    assert result.default == 42


# LLM-generated content at query #9
#--------------------------

```python
def test_schema_field_type_assignment():
    schema_field = Schema(allow_null=False)
    result = to_json_schema(schema_field)
    assert result["type"] == "object"


# LLM-generated content at query #10
#--------------------------

```python
def test_enum_from_json_schema_with_valid_data():
    data = {"enum": ["a", "b", "c"], "default": "a"}
    definitions = Definitions()
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b"), ("c", "c")]
    assert field.default == "a"

def test_enum_from_json_schema_without_default():
    data = {"enum": ["x", "y"]}
    definitions = Definitions()
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [("x", "x"), ("y", "y")]
    assert field.default == NO_DEFAULT

def test_enum_from_json_schema_empty_enum():
    data = {"enum": [], "default": "z"}
    definitions = Definitions()
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == []
    assert field.default == "z"


# LLM-generated content at query #11
#--------------------------

```python
def test_type_from_json_schema_with_single_type():
    data = {"type": "string"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    assert result.allow_null is False

def test_type_from_json_schema_with_multiple_types():
    data = {"type": ["string", "number"]}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Float)
    assert result.allow_null is False

def test_type_from_json_schema_with_null_type():
    data = {"type": "null"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)
    assert result.const is None

def test_type_from_json_schema_with_allow_null():
    data = {"type": "string", "default": None}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    assert result.allow_null is True

def test_type_from_json_schema_with_integer_type():
    data = {"type": "integer", "minimum": 0, "maximum": 100}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.maximum == 100

def test_type_from_json_schema_with_boolean_type():
    data = {"type": "boolean", "default": True}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Boolean)
    assert result.default is True

def test_type_from_json_schema_with_array_type():
    data = {"type": "array", "items": {"type": "string"}}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

def test_type_from_json_schema_with_object_type():
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Object)
    assert isinstance(result.properties["name"], String)

def test_type_from_json_schema_with_no_type():
    data = {}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Any)

def test_type_from_json_schema_with_null_in_multiple_types():
    data = {"type": ["string", "null"]}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True


# LLM-generated content at query #12
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

def test_to_json_schema_with_string_field_with_default():
    field = String(default="default_value")
    assert to_json_schema(field) == {"type": "string", "default": "default_value"}

def test_to_json_schema_with_string_field_with_min_length():
    field = String(min_length=5)
    assert to_json_schema(field) == {"type": "string", "minLength": 5}

def test_to_json_schema_with_string_field_with_max_length():
    field = String(max_length=10)
    assert to_json_schema(field) == {"type": "string", "maxLength": 10}

def test_to_json_schema_with_string_field_with_pattern():
    field = String(pattern_regex=re.compile(r"^[a-z]+$"))
    assert to_json_schema(field) == {"type": "string", "pattern": "^[a-z]+$"}

def test_to_json_schema_with_string_field_with_format():
    field = String(format="email")
    assert to_json_schema(field) == {"type": "string", "format": "email"}

def test_to_json_schema_with_integer_field():
    field = Integer()
    assert to_json_schema(field) == {"type": "integer"}

def test_to_json_schema_with_integer_field_allow_null():
    field = Integer(allow_null=True)
    assert to_json_schema(field) == {"type": ["integer", "null"]}

def test_to_json_schema_with_integer_field_with_minimum():
    field = Integer(minimum=0)
    assert to_json_schema(field) == {"type": "integer", "minimum": 0}

def test_to_json_schema_with_integer_field_with_maximum():
    field = Integer(maximum=100)
    assert to_json_schema(field) == {"type": "integer", "maximum": 100}

def test_to_json_schema_with_float_field():
    field = Float()
    assert to_json_schema(field) == {"type": "number"}

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

def test_to_json_schema_with_array_field_with_items():
    field = Array(items=String())
    assert to_json_schema(field) == {"type": "array", "items": {"type": "string"}}

def test_to_json_schema_with_array_field_with_min_items():
    field = Array(min_items=1)
    assert to_json_schema(field) == {"type": "array", "minItems": 1}

def test_to_json_schema_with_array_field_with_max_items():
    field = Array(max_items=10)
    assert to_json_schema(field) == {"type": "array", "maxItems": 10}

def test_to_json_schema_with_object_field():
    field = Object()
    assert to_json_schema(field) == {"type": "object"}

def test_to_json_schema_with_object_field_with_properties():
    field = Object(properties={"name": String()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_object_field_with_required():
    field = Object(properties={"name": String()}, required=["name"])
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

def test_to_json_schema_with_schema_field():
    field = Schema(fields={"name": String()})
    assert to_json_schema(field) == {"type": "object", "properties": {"name": {"type": "string"}}}

def test_to_json_schema_with_choice_field():
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert to_json_schema(field) == {"enum": ["a", "b"]}

def test_to_json_schema_with_const_field():
    field = Const(const="constant_value")
    assert to_json_schema(field) == {"const": "constant_value"}

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
    field = IfThenElse(if_clause=String(), then_clause=Integer())
    assert to_json_schema(field) == {"if": {"type": "string"}, "then": {"type": "integer"}}

def test_to_json_schema_with_not_field():
    field = Not(negated=String())
    assert to_json_schema(field) == {"not": {"type": "string"}}

def test_to_json_schema_with_definitions():
    definitions = Definitions({"string_field": String()})
    assert to_json_schema(definitions) == {"components": {"schemas": {"string_field": {"type": "string"}}}}


# LLM-generated content at query #13
#--------------------------

```python
def test_array_field_type_predicate():
    field = Array(allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "array"


# LLM-generated content at query #14
#--------------------------

```python
def test_to_json_schema_integer_type_without_null():
    field = Integer(allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "integer"


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_160_evaluates_to_false():
    field = IfThenElse(if_clause=String(), then_clause=None, else_clause=None)
    assert to_json_schema(field) == {
        "if": {"type": "string"},
        "title": None,
        "description": None,
        "default": None,
        "examples": None,
    }


# LLM-generated content at query #16
#--------------------------

```python
def test_array_field_items_is_list_or_tuple():
    field = Array(items=[String(), Integer()])
    assert isinstance(field.items, (list, tuple))


# LLM-generated content at query #17
#--------------------------

```python
def test_from_json_schema_with_bool_true():
    result = from_json_schema(True)
    assert isinstance(result, Any)

def test_from_json_schema_with_bool_false():
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

def test_from_json_schema_with_ref():
    data = {"$ref": "#/components/schemas/test"}
    definitions = Definitions()
    definitions["#/components/schemas/test"] = String()
    result = from_json_schema(data, definitions=definitions)
    assert isinstance(result, Reference)

def test_from_json_schema_with_type_constraint():
    data = {"type": "string"}
    result = from_json_schema(data)
    assert isinstance(result, String)

def test_from_json_schema_with_enum_constraint():
    data = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]

def test_from_json_schema_with_const_constraint():
    data = {"const": "test"}
    result = from_json_schema(data)
    assert isinstance(result, Const)
    assert result.const == "test"

def test_from_json_schema_with_all_of_constraint():
    data = {"allOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_from_json_schema_with_any_of_constraint():
    data = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

def test_from_json_schema_with_one_of_constraint():
    data = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2

def test_from_json_schema_with_not_constraint():
    data = {"not": {"type": "string"}}
    result = from_json_schema(data)
    assert isinstance(result, Not)
    assert isinstance(result.negated, String)

def test_from_json_schema_with_if_then_else_constraint():
    data = {"if": {"type": "string"}, "then": {"type": "number"}, "else": {"type": "boolean"}}
    result = from_json_schema(data)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, Integer)
    assert isinstance(result.else_clause, Boolean)

def test_from_json_schema_with_multiple_constraints():
    data = {"type": "string", "enum": ["a", "b", "c"]}
    result = from_json_schema(data)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

def test_from_json_schema_with_no_constraints():
    data = {}
    result = from_json_schema(data)
    assert isinstance(result, Any)

def test_from_json_schema_with_components_schemas():
    data = {
        "components": {
            "schemas": {
                "test": {"type": "string"}
            }
        }
    }
    result = from_json_schema(data)
    assert isinstance(result, Any)


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_172():
    field = Field()
    definitions = {}
    is_root = True
    assert is_root and definitions


