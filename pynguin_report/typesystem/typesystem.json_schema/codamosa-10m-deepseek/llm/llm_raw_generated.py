####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function one_of_from_json_schema
def test_one_of_from_json_schema():
    definitions = Definitions()
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    field = one_of_from_json_schema(data, definitions)
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2
    assert isinstance(field.one_of[0], String)
    assert isinstance(field.one_of[1], Integer)


# LLM-generated content at query #2
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():
    definitions = Definitions()
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"},
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Float)
    assert isinstance(field.else_clause, Boolean)

    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Float)
    assert field.else_clause is None

    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert field.then_clause is None
    assert isinstance(field.else_clause, Boolean)


# LLM-generated content at query #3
#--------------------------

# Unit test for function one_of_from_json_schema
def test_one_of_from_json_schema():
    from typesystem.fields import Integer, String
    from typesystem.composites import OneOf

    # Test case 1: Valid input with multiple items
    data = {"oneOf": [{"type": "integer"}, {"type": "string"}]}
    definitions = Definitions()
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Integer)
    assert isinstance(result.one_of[1], String)

    # Test case 2: Valid input with default value
    data = {"oneOf": [{"type": "integer"}, {"type": "string"}], "default": 42}
    definitions = Definitions()
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert result.default == 42

    # Test case 3: Empty oneOf list
    data = {"oneOf": []}
    definitions = Definitions()
    try:
        one_of_from_json_schema(data, definitions)
        assert False, "Expected ValueError for empty oneOf list"
    except ValueError:
        pass

    # Test case 4: Invalid input (non-dict)
    data = "invalid"
    definitions = Definitions()
    try:
        one_of_from_json_schema(data, definitions)
        assert False, "Expected TypeError for non-dict input"
    except TypeError:
        pass



# LLM-generated content at query #4
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    definitions = Definitions()
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "format": "email",
        "pattern": "^[A-Za-z0-9]+$",
        "default": "example@example.com"
    }
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[A-Za-z0-9]+$"
    assert field.default == "example@example.com"

    data = {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 101,
        "multipleOf": 2,
        "default": 2
    }
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 101
    assert field.multiple_of == 2
    assert field.default == 2

    data = {
        "type": "number",
        "minimum": 1.0,
        "maximum": 100.0,
        "exclusiveMinimum": 0.0,
        "exclusiveMaximum": 101.0,
        "multipleOf": 2.0,
        "default": 2.0
    }
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 1.0
    assert field.maximum == 100.0
    assert field.exclusive_minimum == 0.0
    assert field.exclusive_maximum == 101.0
    assert field.multiple_of == 2.0
    assert field.default == 2.0

    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True

    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["example"]
    }
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["example"]

    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "minProperties": 1,
        "maxProperties": 2,
        "required": ["name"],
        "default": {"name": "example"}
    }
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.required == ["name"]
    assert field.default == {"name": "example"}


# LLM-generated content at query #5
#--------------------------

# Unit test for function all_of_from_json_schema
def test_all_of_from_json_schema():
    definitions = Definitions()
    data = {
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    }
    field = all_of_from_json_schema(data, definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2
    assert isinstance(field.all_of[0], String)
    assert isinstance(field.all_of[1], String)
    assert field.all_of[0].min_length == 5
    assert field.all_of[1].max_length == 10



# LLM-generated content at query #6
#--------------------------

# Unit test for function from_json_schema
def test_from_json_schema():
    # Test with boolean input
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test with dictionary input
    schema = {"type": "string"}
    assert isinstance(from_json_schema(schema), String)

    schema = {"type": "integer"}
    assert isinstance(from_json_schema(schema), Integer)

    schema = {"type": "number"}
    assert isinstance(from_json_schema(schema), Number)

    schema = {"type": "boolean"}
    assert isinstance(from_json_schema(schema), Boolean)

    schema = {"type": "array", "items": {"type": "string"}}
    assert isinstance(from_json_schema(schema), Array)

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert isinstance(from_json_schema(schema), Object)

    schema = {"enum": ["red", "green", "blue"]}
    assert isinstance(from_json_schema(schema), Choice)

    schema = {"const": "constant_value"}
    assert isinstance(from_json_schema(schema), Const)

    schema = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    assert isinstance(from_json_schema(schema), AllOf)

    schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    assert isinstance(from_json_schema(schema), OneOf)

    schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    assert isinstance(from_json_schema(schema), OneOf)

    schema = {"not": {"type": "string"}}
    assert isinstance(from_json_schema(schema), Not)

    schema = {"if": {"type": "string"}, "then": {"minLength": 5}}
    assert isinstance(from_json_schema(schema), IfThenElse)

    # Test with $ref
    definitions = Definitions()
    definitions["#/components/schemas/Example"] = String()
    schema = {"$ref": "#/components/schemas/Example"}
    assert isinstance(from_json_schema(schema, definitions=definitions), Reference)

    # Test with nested definitions
    schema = {"type": "object", "properties": {"name": {"$ref": "#/components/schemas/Example"}}}
    assert isinstance(from_json_schema(schema, definitions=definitions), Object)

    # Test with complex schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0}
        },
        "required": ["name"]
    }
    assert isinstance(from_json_schema(schema), Object)

    # Test with additional properties
    schema = {"type": "object", "additionalProperties": {"type": "string"}}
    assert isinstance(from_json_schema(schema), Object)

    # Test with pattern properties
    schema = {"type": "object", "patternProperties": {"^[a-z]+$": {"type": "string"}}}
    assert isinstance(from_json_schema(schema), Object)

    # Test with array items
    schema = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    assert isinstance(from_json_schema(schema), Array)

    # Test with unique items
    schema = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
    assert isinstance(from_json_schema(schema), Array)

    # Test with format
    schema = {"type": "string", "format": "email"}
    assert isinstance(from_json_schema(schema), String)

    # Test with multiple constraints
    schema = {"type": "string", "minLength": 5, "maxLength": 10}
    assert isinstance(from_json_schema(schema), String)

    # Test with multiple types
    schema = {"type": ["string", "number"]}
    assert isinstance(from_json_schema(schema), Union)

    # Test with default definitions
    schema = {"type": "string"}
    assert isinstance(from_json_schema(schema), String)

    # Test with custom definitions
    custom_definitions = Definitions()
    custom_definitions["#/components/schemas/Example"] = String()
    schema = {"$ref": "#/components/schemas/Example"}
    assert isinstance(from_json_schema(schema, definitions=custom_definitions), Reference)

    # Test with no constraints
    schema = {}
    assert isinstance(from_json_schema(schema), Any)


# LLM-generated content at query #7
#--------------------------

# Unit test for function get_standard_properties
def test_get_standard_properties():
    # Test case 1: Field with default value
    field_with_default = Field(default="test_default")
    assert get_standard_properties(field_with_default) == {"default": "test_default"}

    # Test case 2: Field without default value
    field_without_default = Field()
    assert get_standard_properties(field_without_default) == {}

    # Test case 3: Field with None as default value
    field_with_none_default = Field(default=None)
    assert get_standard_properties(field_with_none_default) == {"default": None}

    # Test case 4: Field with a complex default value
    complex_default = {"key": "value"}
    field_with_complex_default = Field(default=complex_default)
    assert get_standard_properties(field_with_complex_default) == {"default": complex_default}

    # Test case 5: Field with default value and other properties
    field_with_properties = Field(default="test_default", description="Test Field")
    assert get_standard_properties(field_with_properties) == {"default": "test_default"}


# LLM-generated content at query #8
#--------------------------

# Unit test for function enum_from_json_schema
def test_enum_from_json_schema():
    # Test with simple enum values
    data = {"enum": ["a", "b", "c"]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test with enum values and default
    data = {"enum": [1, 2, 3], "default": 2}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [(1, 1), (2, 2), (3, 3)]
    assert field.default == 2

    # Test with empty enum (should raise an error)
    data = {"enum": []}
    try:
        field = enum_from_json_schema(data, definitions=Definitions())
        assert False, "Expected ValueError for empty enum"
    except ValueError:
        pass

    # Test with enum containing None
    data = {"enum": [None, "a", "b"]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [(None, None), ("a", "a"), ("b", "b")]

    # Test with enum containing mixed types
    data = {"enum": [1, "a", True]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [(1, 1), ("a", "a"), (True, True)]


# LLM-generated content at query #9
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    # Test with Any field
    any_field = Any()
    assert to_json_schema(any_field) == True

    # Test with NeverMatch field
    never_match_field = NeverMatch()
    assert to_json_schema(never_match_field) == False

    # Test with String field
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    expected_string_schema = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected_string_schema

    # Test with Integer field
    integer_field = Integer(allow_null=True, minimum=0, maximum=100, multiple_of=5)
    expected_integer_schema = {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 5
    }
    assert to_json_schema(integer_field) == expected_integer_schema

    # Test with Boolean field
    boolean_field = Boolean(allow_null=True)
    expected_boolean_schema = {
        "type": ["boolean", "null"]
    }
    assert to_json_schema(boolean_field) == expected_boolean_schema

    # Test with Array field
    item_field = String()
    array_field = Array(allow_null=True, min_items=1, max_items=10, items=item_field, unique_items=True)
    expected_array_schema = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 10,
        "items": {"type": "string"},
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected_array_schema

    # Test with Object field
    properties = {"name": String()}
    object_field = Object(allow_null=True, properties=properties, required=["name"])
    expected_object_schema = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(object_field) == expected_object_schema

    # Test with Reference field
    definitions = {"Person": Object(properties={"name": String()})}
    reference_field = Reference(to="Person", definitions=definitions)
    expected_reference_schema = {
        "$ref": "#/components/schemas/Person",
        "components": {
            "schemas": {
                "Person": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        }
    }
    assert to_json_schema(reference_field) == expected_reference_schema

    # Test with Union field
    string_or_int = Union(any_of=[String(), Integer()])
    expected_union_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    assert to_json_schema(string_or_int) == expected_union_schema

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    expected_all_of_schema = {
        "allOf": [
            {"type": "string", "minLength": 1},
            {"type": "string", "maxLength": 10}
        ]
    }
    assert to_json_schema(all_of_field) == expected_all_of_schema

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(min_length=5),
        then_clause=String(max_length=10),
        else_clause=Integer()
    )
    expected_if_then_else_schema = {
        "if": {"type": "string", "minLength": 5},
        "then": {"type": "string", "maxLength": 10},
        "else": {"type": "integer"}
    }
    assert to_json_schema(if_then_else_field) == expected_if_then_else_schema

    print("All tests passed!")

test_to_json_schema()


# LLM-generated content at query #10
#--------------------------

# Unit test for function from_json_schema
def test_from_json_schema():
    # Test with a boolean schema
    assert isinstance(from_json_schema(True), Field)
    assert isinstance(from_json_schema(False), Field)
    
    # Test with a valid JSON schema dictionary
    schema = {"type": "string", "minLength": 5}
    assert isinstance(from_json_schema(schema), Field)
    
    # Test with a schema containing a reference
    schema = {"$ref": "#/definitions/Example"}
    assert isinstance(from_json_schema(schema), Field)
    
    # Test with a schema containing enum
    schema = {"enum": ["value1", "value2"]}
    assert isinstance(from_json_schema(schema), Field)
    
    # Test with a schema containing const
    schema = {"const": "value"}
    assert isinstance(from_json_schema(schema), Field)
    
    # Test with a schema containing allOf
    schema = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    assert isinstance(from_json_schema(schema), Field)
    
    # Test with a schema containing anyOf
    schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    assert isinstance(from_json_schema(schema), Field)
    
    # Test with a schema containing oneOf
    schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    assert isinstance(from_json_schema(schema), Field)
    
    # Test with a schema containing not
    schema = {"not": {"type": "string"}}
    assert isinstance(from_json_schema(schema), Field)
    
    # Test with a schema containing if/then/else
    schema = {"if": {"type": "string"}, "then": {"minLength": 5}, "else": {"type": "number"}}
    assert isinstance(from_json_schema(schema), Field)
    
    # Test with a schema containing multiple constraints
    schema = {"type": "string", "minLength": 5, "maxLength": 10}
    assert isinstance(from_json_schema(schema), Field)


# LLM-generated content at query #11
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():
    definitions = {
        "#/components/schemas/MySchema": String(min_length=1)
    }
    data = {"$ref": "#/components/schemas/MySchema"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/MySchema"
    assert result.definitions == definitions




# LLM-generated content at query #12
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    field = ref_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/User"
    assert field.definitions is definitions

    try:
        ref_from_json_schema({"$ref": "http://example.com"}, definitions=definitions)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError for unsupported $ref style"


# LLM-generated content at query #13
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    # Test case 1: Any field
    assert to_json_schema(Any()) == True

    # Test case 2: NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test case 3: String field
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern="^[a-z]+$", format="email", default="test@example.com")
    expected_string_schema = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "format": "email",
        "default": "test@example.com"
    }
    assert to_json_schema(string_field) == expected_string_schema

    # Test case 4: Integer field
    integer_field = Integer(allow_null=True, minimum=1, maximum=100, exclusive_minimum=0, exclusive_maximum=101, multiple_of=5, default=50)
    expected_integer_schema = {
        "type": ["integer", "null"],
        "minimum": 1,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 101,
        "multipleOf": 5,
        "default": 50
    }
    assert to_json_schema(integer_field) == expected_integer_schema

    # Test case 5: Boolean field
    boolean_field = Boolean(allow_null=True, default=True)
    expected_boolean_schema = {
        "type": ["boolean", "null"],
        "default": True
    }
    assert to_json_schema(boolean_field) == expected_boolean_schema

    # Test case 6: Array field
    array_field = Array(allow_null=True, min_items=1, max_items=10, items=Integer(), additional_items=True, unique_items=True, default=[1, 2, 3])
    expected_array_schema = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 10,
        "items": {"type": "integer"},
        "additionalItems": True,
        "uniqueItems": True,
        "default": [1, 2, 3]
    }
    assert to_json_schema(array_field) == expected_array_schema

    # Test case 7: Object field
    object_field = Object(allow_null=True, properties={"name": String()}, pattern_properties={"^[a-z]+$": Integer()}, additional_properties=True, property_names=String(), min_properties=1, max_properties=10, required=["name"], default={"name": "John"})
    expected_object_schema = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}},
        "patternProperties": {"^[a-z]+$": {"type": "integer"}},
        "additionalProperties": True,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "John"}
    }
    assert to_json_schema(object_field) == expected_object_schema

    # Test case 8: Schema field
    schema_field = Schema(allow_null=True, fields={"name": String(), "age": Integer()}, required=["name"], default={"name": "John", "age": 30})
    expected_schema_schema = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    assert to_json_schema(schema_field) == expected_schema_schema

    # Test case 9: Choice field
    choice_field = Choice(choices=[("male", "Male"), ("female", "Female")], default="male")
    expected_choice_schema = {
        "enum": ["male", "female"],
        "default": "male"
    }
    assert to_json_schema(choice_field) == expected_choice_schema

    # Test case 10: Const field
    const_field = Const(const=42, default=42)
    expected_const_schema = {
        "const": 42,
        "default": 42
    }
    assert to_json_schema(const_field) == expected_const_schema

    # Test case 11: Union field
    union_field = Union(any_of=[Integer(), String()], default=10)
    expected_union_schema = {
        "anyOf": [{"type": "integer"}, {"type": "string"}],
        "default": 10
    }
    assert to_json_schema(union_field) == expected_union_schema

    # Test case 12: OneOf field
    one_of_field = OneOf(one_of=[Integer(), String()], default=10)
    expected_one_of_schema = {
        "oneOf": [{"type": "integer"}, {"type": "string"}],
        "default": 10
    }
    assert to_json_schema(one_of_field) == expected_one_of_schema

    # Test case 13: AllOf field
    all_of_field = AllOf(all_of=[Integer(), Integer(minimum=10)], default=10)
    expected_all_of_schema = {
        "allOf": [{"type": "integer"}, {"type": "integer", "minimum": 10}],
        "default": 10
    }
    assert to_json_schema(all_of_field) == expected_all_of_schema

    # Test case 14: IfThenElse field
    if_then_else_field = IfThenElse(if_clause=Integer(minimum=10), then_clause=String(), else_clause=Integer(), default=10)
    expected_if_then_else_schema = {
        "if": {"type": "integer", "minimum": 10},
        "then": {"type": "string"},
        "else": {"type": "integer"},
        "default": 10
    }
    assert to_json_schema(if_then_else_field) == expected_if_then_else_schema

    # Test case 15: Not field
    not_field = Not(negated=Integer(minimum=10), default=10)
    expected_not_schema = {
        "not": {"type": "integer", "minimum": 10},
        "default": 10
    }
    assert to_json_schema(not_field) == expected_not_schema

    # Test case 16: Definitions
    definitions = Definitions({"Person": Schema(fields={"name": String(), "age": Integer()})})
    expected_definitions_schema = {
        "components": {
            "schemas": {
                "Person": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}
                }
            }
        }
    }
    assert to_json_schema(definitions) == expected_definitions_schema

    # Test case 17: Reference field
    reference_field = Reference(to="Person", definitions=Definitions({"Person": Schema(fields={"name": String()})}))
    expected_reference_schema = {"$ref": "#/components/schemas/Person"}
    assert to_json_schema(reference_field) == expected_reference_schema

    # Test case 18: Invalid field type
    try:
        to_json_schema("invalid_field")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #14
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    assert to_json_schema(Any()) == True
    assert to_json_schema(NeverMatch()) == False
    assert to_json_schema(String()) == {"type": "string"}
    assert to_json_schema(Integer()) == {"type": "integer"}
    assert to_json_schema(Float()) == {"type": "number"}
    assert to_json_schema(Boolean()) == {"type": "boolean"}
    assert to_json_schema(Array()) == {"type": "array"}
    assert to_json_schema(Object()) == {"type": "object"}
    assert to_json_schema(Schema()) == {"type": "object"}
    assert to_json_schema(Choice(choices=[("a", "A")])) == {"enum": ["a"]}
    assert to_json_schema(Const(const=True)) == {"const": True}
    assert to_json_schema(Union(any_of=[String(), Integer()])) == {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(OneOf(one_of=[String(), Integer()])) == {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(AllOf(all_of=[String(), Integer()])) == {"allOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())) == {"if": {"type": "string"}, "then": {"type": "integer"}, "else": {"type": "boolean"}}
    assert to_json_schema(Not(negated=String())) == {"not": {"type": "string"}}


# LLM-generated content at query #15
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():
    definitions = Definitions()
    ref_data = {"$ref": "#/definitions/Example"}
    field = ref_from_json_schema(ref_data, definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/definitions/Example"


# LLM-generated content at query #16
#--------------------------

# Unit test for function type_from_json_schema
def test_type_from_json_schema():
    # Test with single type string
    data = {"type": "string"}
    definitions = Definitions()
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, String)
    assert field.allow_null is False

    # Test with multiple type strings
    data = {"type": ["string", "number"]}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2
    assert isinstance(field.any_of[0], String)
    assert isinstance(field.any_of[1], Number)
    assert field.allow_null is False

    # Test with allow null
    data = {"type": "string", "nullable": True}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, String)
    assert field.allow_null is True

    # Test with no type strings
    data = {"nullable": True}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Const)
    assert field.const is None

    # Test with no type strings and not nullable
    data = {}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, NeverMatch)


# LLM-generated content at query #17
#--------------------------

# Unit test for function type_from_json_schema
def test_type_from_json_schema():
    definitions = Definitions()
    data = {"type": "string", "minLength": 1}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, String)
    assert field.min_length == 1

    data = {"type": "number", "minimum": 0}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Number)
    assert field.minimum == 0

    data = {"type": "integer", "multipleOf": 2}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Integer)
    assert field.multiple_of == 2

    data = {"type": "boolean"}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Boolean)

    data = {"type": "null"}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Const)
    assert field.const is None

    data = {"type": ["string", "null"]}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, String)
    assert field.allow_null

    data = {"type": ["string", "number"]}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2
    assert isinstance(field.any_of[0], String)
    assert isinstance(field.any_of[1], Number)

    data = {"type": []}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, NeverMatch)

    data = {"type": "null", "allow_null": True}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Const)
    assert field.const is None


# LLM-generated content at query #18
#--------------------------

# Unit test for function type_from_json_schema
def test_type_from_json_schema():
    # Test with a single type string
    data = {"type": "string"}
    field = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, String)
    assert not field.allow_null

    # Test with multiple type strings
    data = {"type": ["string", "number"]}
    field = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Union)
    assert len(field.any_of) == 2
    assert isinstance(field.any_of[0], String)
    assert isinstance(field.any_of[1], Number)
    assert not field.allow_null

    # Test with null type
    data = {"type": "null"}
    field = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Const)
    assert field.const is None
    assert not field.allow_null

    # Test with allow_null
    data = {"type": ["string", "null"]}
    field = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, String)
    assert field.allow_null

    # Test with no type strings (only null)
    data = {}
    field = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Any)
    assert not field.allow_null

    # Test with additional constraints
    data = {"type": "string", "minLength": 5}
    field = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert not field.allow_null


# LLM-generated content at query #19
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    # Test number field
    data = {
        "type": "number",
        "minimum": 1,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 11,
        "multipleOf": 2,
        "default": 5.5
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 5.5

    # Test integer field
    data = {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 11,
        "multipleOf": 2,
        "default": 5
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 5

    # Test string field
    data = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9]+$"
    assert field.default == "test"

    # Test boolean field
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array field
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test object field
    data = {
        "type": "object",
        "properties": {"test": {"type": "string"}},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["test"],
        "default": {"test": "value"}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["test"]
    assert field.default == {"test": "value"}

    # Test invalid type string
    try:
        from_json_schema_type({}, "invalid", False, Definitions())
        assert False, "Expected assertion error"
    except AssertionError:
        pass


# LLM-generated content at query #20
#--------------------------

# Unit test for function type_from_json_schema
def test_type_from_json_schema():
    definitions = Definitions()

    # Test case: Single type string, no null
    data = {"type": "string"}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, String)
    assert not field.allow_null

    # Test case: Single type string, allow null
    data = {"type": ["string", "null"]}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, String)
    assert field.allow_null

    # Test case: Multiple type strings, no null
    data = {"type": ["string", "integer"]}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2
    assert isinstance(field.any_of[0], String)
    assert isinstance(field.any_of[1], Integer)
    assert not field.allow_null

    # Test case: Multiple type strings, allow null
    data = {"type": ["string", "integer", "null"]}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2
    assert isinstance(field.any_of[0], String)
    assert isinstance(field.any_of[1], Integer)
    assert field.allow_null

    # Test case: No type strings, allow null
    data = {"type": ["null"]}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Const)
    assert field.value is None

    # Test case: No type strings, no null
    data = {"type": []}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, NeverMatch)


# LLM-generated content at query #21
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    # Test with type "number"
    data = {
        "type": "number",
        "minimum": 1,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 11,
        "multipleOf": 2,
        "default": 5.0
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 5.0

    # Test with type "integer"
    data = {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 11,
        "multipleOf": 2,
        "default": 5
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 5

    # Test with type "string"
    data = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "format": "email",
        "pattern": "^[a-z]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == re.compile("^[a-z]+$")
    assert field.default == "test"

    # Test with type "boolean"
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test with type "array"
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test with type "object"
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "minProperties": 1,
        "maxProperties": 2,
        "required": ["name"],
        "default": {"name": "test"}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.required == ["name"]
    assert field.default == {"name": "test"}

    # Test with allow_null=True
    data = {
        "type": "string",
        "default": None
    }
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True
    assert field.default == None

    # Test with invalid type_string
    try:
        from_json_schema_type({}, "invalid", False, Definitions())
        assert False, "Expected assertion error"
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid'"


# LLM-generated content at query #22
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    definitions = Definitions()
    data = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "format": "email",
        "default": "test@example.com"
    }
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.pattern == re.compile("^[a-z]+$")
    assert field.format == "email"
    assert field.default == "test@example.com"

    data = {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 11,
        "multipleOf": 2,
        "default": 2
    }
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 2

    data = {
        "type": "number",
        "minimum": 1.0,
        "maximum": 10.0,
        "exclusiveMinimum": 0.0,
        "exclusiveMaximum": 11.0,
        "multipleOf": 2.0,
        "default": 2.0
    }
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 1.0
    assert field.maximum == 10.0
    assert field.exclusive_minimum == 0.0
    assert field.exclusive_maximum == 11.0
    assert field.multiple_of == 2.0
    assert field.default == 2.0

    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default is True

    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items is True
    assert field.default == ["test"]

    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 2,
        "default": {"name": "test"}
    }
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.default == {"name": "test"}


# LLM-generated content at query #23
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    # Test with number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 10,
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 10

    # Test with integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 10,
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 10

    # Test with string type
    data = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "format": "email",
        "pattern": "^[a-z]+$",
        "default": "test",
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == re.compile("^[a-z]+$")
    assert field.default == "test"

    # Test with boolean type
    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test with array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"],
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test with object type
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test"},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}


# LLM-generated content at query #24
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    # Test case for 'number' type
    data = {
        "type": "number",
        "minimum": 1,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 11,
        "multipleOf": 2,
        "default": 3,
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 3

    # Test case for 'integer' type
    data = {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 11,
        "multipleOf": 2,
        "default": 3,
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 3

    # Test case for 'string' type
    data = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "format": "email",
        "pattern": "^[a-z]+$",
        "default": "test",
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-z]+$"
    assert field.default == "test"

    # Test case for 'boolean' type
    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test case for 'array' type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"],
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test case for 'object' type
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test"},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}


# LLM-generated content at query #25
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    definitions = Definitions()
    data = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.pattern == re.compile("^[a-z]+$")
    assert field.default == "test"

    data = {
        "type": "number",
        "minimum": 1,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 11,
        "multipleOf": 2,
        "default": 4
    }
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 4

    data = {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 11,
        "multipleOf": 2,
        "default": 4
    }
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 4

    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True

    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "default": {"name": "test"}
    }
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert field.required == ["name"]
    assert field.default == {"name": "test"}


# LLM-generated content at query #26
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    definitions = Definitions()
    
    # Test number type
    data = {"type": "number", "minimum": 1, "maximum": 10, "exclusiveMinimum": 0.5, "exclusiveMaximum": 10.5, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0.5
    assert field.exclusive_maximum == 10.5
    assert field.multiple_of == 2
    assert field.default == 5
    
    # Test integer type
    data = {"type": "integer", "minimum": 1, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 11, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 11
    assert field.multiple_of == 2
    assert field.default == 5
    
    # Test string type
    data = {"type": "string", "minLength": 1, "maxLength": 10, "format": "email", "pattern": "^[a-z]+$", "default": "test"}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == re.compile("^[a-z]+$")
    assert field.default == "test"
    
    # Test boolean type
    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True
    
    # Test array type
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10, "uniqueItems": True, "default": ["test"]}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]
    
    # Test object type
    data = {"type": "object", "properties": {"name": {"type": "string"}}, "minProperties": 1, "maxProperties": 10, "required": ["name"], "default": {"name": "test"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}


# LLM-generated content at query #27
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    from typesystem.fields import String, Integer, Boolean, Array, Object, Reference, Schema, Choice, Const, Union, OneOf, AllOf, IfThenElse, Not, Any, NeverMatch
    from typesystem.schemas import Definitions

    # Test Any
    assert to_json_schema(Any()) == True

    # Test NeverMatch
    assert to_json_schema(NeverMatch()) == False

    # Test String
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern="^[a-zA-Z]+$", format="email")
    assert to_json_schema(string_field) == {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-zA-Z]+$",
        "format": "email"
    }

    # Test Integer
    integer_field = Integer(allow_null=True, minimum=1, maximum=10, exclusive_minimum=0, exclusive_maximum=11, multiple_of=2)
    assert to_json_schema(integer_field) == {
        "type": ["integer", "null"],
        "minimum": 1,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 11,
        "multipleOf": 2
    }

    # Test Boolean
    boolean_field = Boolean(allow_null=True)
    assert to_json_schema(boolean_field) == {
        "type": ["boolean", "null"]
    }

    # Test Array
    array_field = Array(allow_null=True, min_items=1, max_items=10, items=String(), additional_items=True, unique_items=True)
    assert to_json_schema(array_field) == {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 10,
        "items": {
            "type": "string"
        },
        "additionalItems": True,
        "uniqueItems": True
    }

    # Test Object
    object_field = Object(allow_null=True, properties={"name": String()}, pattern_properties={"^[a-z]+$": Integer()}, additional_properties=True, property_names=String(), min_properties=1, max_properties=10, required=["name"])
    assert to_json_schema(object_field) == {
        "type": ["object", "null"],
        "properties": {
            "name": {
                "type": "string"
            }
        },
        "patternProperties": {
            "^[a-z]+$": {
                "type": "integer"
            }
        },
        "additionalProperties": True,
        "propertyNames": {
            "type": "string"
        },
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"]
    }

    # Test Schema
    schema_field = Schema(allow_null=True, fields={"name": String()}, required=["name"])
    assert to_json_schema(schema_field) == {
        "type": ["object", "null"],
        "properties": {
            "name": {
                "type": "string"
            }
        },
        "required": ["name"]
    }

    # Test Choice
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    assert to_json_schema(choice_field) == {
        "enum": ["a", "b"]
    }

    # Test Const
    const_field = Const(const="constant")
    assert to_json_schema(const_field) == {
        "const": "constant"
    }

    # Test Union
    union_field = Union(any_of=[String(), Integer()])
    assert to_json_schema(union_field) == {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }

    # Test OneOf
    one_of_field = OneOf(one_of=[String(), Integer()])
    assert to_json_schema(one_of_field) == {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }

    # Test AllOf
    all_of_field = AllOf(all_of=[String(), Integer()])
    assert to_json_schema(all_of_field) == {
        "allOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }

    # Test IfThenElse
    if_then_else_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    assert to_json_schema(if_then_else_field) == {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }

    # Test Not
    not_field = Not(negated=String())
    assert to_json_schema(not_field) == {
        "not": {"type": "string"}
    }

    # Test Reference
    definitions = Definitions({"MySchema": Schema(fields={"name": String()})})
    reference_field = Reference(to="MySchema", definitions=definitions)
    assert to_json_schema(reference_field) == {
        "$ref": "#/components/schemas/MySchema",
        "components": {
            "schemas": {
                "MySchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string"
                        }
                    }
                }
            }
        }
    }


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    assert to_json_schema(Any()) is True
    assert to_json_schema(NeverMatch()) is False
    assert to_json_schema(String(allow_null=True)) == {
        "type": ["string", "null"],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Integer(allow_null=True)) == {
        "type": ["integer", "null"],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Float(allow_null=True)) == {
        "type": ["number", "null"],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Boolean(allow_null=True)) == {
        "type": ["boolean", "null"],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Array(allow_null=True)) == {
        "type": ["array", "null"],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Object(allow_null=True)) == {
        "type": ["object", "null"],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Schema(allow_null=True)) == {
        "type": ["object", "null"],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Choice(choices=[("a", "a")])) == {
        "enum": ["a"],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Const(const="a")) == {
        "const": "a",
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Union(any_of=[String(), Integer()])) == {
        "anyOf": [{"type": "string", "default": NO_DEFAULT}, {"type": "integer", "default": NO_DEFAULT}],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(OneOf(one_of=[String(), Integer()])) == {
        "oneOf": [{"type": "string", "default": NO_DEFAULT}, {"type": "integer", "default": NO_DEFAULT}],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(AllOf(all_of=[String(), Integer()])) == {
        "allOf": [{"type": "string", "default": NO_DEFAULT}, {"type": "integer", "default": NO_DEFAULT}],
        "default": NO_DEFAULT,
    }
    assert to_json_schema(IfThenElse(if_clause=String(), then_clause=Integer())) == {
        "if": {"type": "string", "default": NO_DEFAULT},
        "then": {"type": "integer", "default": NO_DEFAULT},
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Not(negated=String())) == {
        "not": {"type": "string", "default": NO_DEFAULT},
        "default": NO_DEFAULT,
    }
    assert to_json_schema(Reference(to="a", definitions={"a": String()})) == {
        "$ref": "#/components/schemas/a",
        "components": {"schemas": {"a": {"type": "string", "default": NO_DEFAULT}}},
    }
    assert to_json_schema(Definitions({"a": String()})) == {
        "components": {"schemas": {"a": {"type": "string", "default": NO_DEFAULT}}},
    }


# LLM-generated content at query #2
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    field = ref_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/User"
    assert field.definitions is definitions


# LLM-generated content at query #3
#--------------------------

# Unit test for function one_of_from_json_schema
def test_one_of_from_json_schema():
    definitions = Definitions()
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    field = one_of_from_json_schema(data, definitions)
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2
    assert isinstance(field.one_of[0], String)
    assert isinstance(field.one_of[1], Float)


# LLM-generated content at query #4
#--------------------------

# Unit test for function type_from_json_schema
def test_type_from_json_schema():
    definitions = Definitions()
    data_single_type = {"type": "string"}
    field = type_from_json_schema(data_single_type, definitions=definitions)
    assert isinstance(field, String)
    assert not field.allow_null

    data_multiple_types = {"type": ["string", "null"]}
    field = type_from_json_schema(data_multiple_types, definitions=definitions)
    assert isinstance(field, String)
    assert field.allow_null

    data_empty_type = {"type": []}
    field = type_from_json_schema(data_empty_type, definitions=definitions)
    assert isinstance(field, NeverMatch)

    data_null_type = {"type": ["null"]}
    field = type_from_json_schema(data_null_type, definitions=definitions)
    assert isinstance(field, Const)
    assert field.const is None

    data_invalid_type = {"type": "invalid"}
    try:
        type_from_json_schema(data_invalid_type, definitions=definitions)
        assert False, "Expected a ValueError"
    except ValueError:
        pass


# LLM-generated content at query #5
#--------------------------

# Unit test for function from_json_schema
def test_from_json_schema():
    # Test with boolean input
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test with dictionary input
    schema = {"type": "string"}
    assert isinstance(from_json_schema(schema), String)

    schema = {"type": "integer"}
    assert isinstance(from_json_schema(schema), Integer)

    schema = {"type": "number"}
    assert isinstance(from_json_schema(schema), Number)

    schema = {"type": "boolean"}
    assert isinstance(from_json_schema(schema), Boolean)

    schema = {"type": "array", "items": {"type": "string"}}
    assert isinstance(from_json_schema(schema), Array)

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert isinstance(from_json_schema(schema), Object)

    # Test with $ref
    schema = {"$ref": "#/components/schemas/Person"}
    assert isinstance(from_json_schema(schema), Reference)

    # Test with enum
    schema = {"enum": ["red", "green", "blue"]}
    assert isinstance(from_json_schema(schema), Choice)

    # Test with const
    schema = {"const": "red"}
    assert isinstance(from_json_schema(schema), Const)

    # Test with allOf
    schema = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    assert isinstance(from_json_schema(schema), AllOf)

    # Test with anyOf
    schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    assert isinstance(from_json_schema(schema), OneOf)

    # Test with oneOf
    schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    assert isinstance(from_json_schema(schema), OneOf)

    # Test with not
    schema = {"not": {"type": "string"}}
    assert isinstance(from_json_schema(schema), Not)

    # Test with if-then-else
    schema = {"if": {"type": "string"}, "then": {"minLength": 5}, "else": {"type": "number"}}
    assert isinstance(from_json_schema(schema), IfThenElse)


# LLM-generated content at query #6
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():
    definitions = Definitions()
    reference_string = "#/components/schemas/test_schema"
    data = {"$ref": reference_string}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == reference_string
    assert result.definitions == definitions


# LLM-generated content at query #7
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"},
    }
    definitions = Definitions()
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Field)
    assert isinstance(result.then_clause, Field)
    assert isinstance(result.else_clause, Field)
    assert result.default == NO_DEFAULT

    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
    }
    definitions = Definitions()
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Field)
    assert isinstance(result.then_clause, Field)
    assert result.else_clause is None
    assert result.default == NO_DEFAULT

    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
    }
    definitions = Definitions()
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Field)
    assert result.then_clause is None
    assert isinstance(result.else_clause, Field)
    assert result.default == NO_DEFAULT


# LLM-generated content at query #8
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():
    definitions = Definitions()
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"},
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Float)
    assert isinstance(field.else_clause, Boolean)


# LLM-generated content at query #9
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    field = ref_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/User"
    assert field.definitions is definitions

    # Test with a different reference path
    data = {"$ref": "#/definitions/User"}
    field = ref_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/definitions/User"
    assert field.definitions is definitions

    # Test with an invalid reference path (should still work but may not resolve properly)
    data = {"$ref": "#/invalid/path"}
    field = ref_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/invalid/path"
    assert field.definitions is definitions


# LLM-generated content at query #10
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    # Test with Any field
    any_field = Any()
    assert to_json_schema(any_field) == True

    # Test with NeverMatch field
    never_match_field = NeverMatch()
    assert to_json_schema(never_match_field) == False

    # Test with String field
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    expected_string_schema = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected_string_schema

    # Test with Integer field
    integer_field = Integer(allow_null=True, minimum=0, maximum=100, multiple_of=5)
    expected_integer_schema = {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 5
    }
    assert to_json_schema(integer_field) == expected_integer_schema

    # Test with Boolean field
    boolean_field = Boolean(allow_null=True)
    expected_boolean_schema = {
        "type": ["boolean", "null"]
    }
    assert to_json_schema(boolean_field) == expected_boolean_schema

    # Test with Array field
    array_field = Array(
        allow_null=True,
        min_items=1,
        max_items=10,
        items=String(),
        additional_items=False,
        unique_items=True
    )
    expected_array_schema = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 10,
        "items": {"type": "string"},
        "additionalItems": False,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected_array_schema

    # Test with Object field
    object_field = Object(
        allow_null=True,
        properties={"name": String()},
        required=["name"],
        additional_properties=False
    )
    expected_object_schema = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False
    }
    assert to_json_schema(object_field) == expected_object_schema

    # Test with Reference field
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    reference_field = Reference(to="Person", definitions=definitions)
    expected_reference_schema = {
        "$ref": "#/components/schemas/Person",
        "components": {
            "schemas": {
                "Person": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        }
    }
    assert to_json_schema(reference_field) == expected_reference_schema

    print("All tests passed!")

test_to_json_schema()


# LLM-generated content at query #11
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    field = ref_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/User"
    assert field.definitions is definitions


# LLM-generated content at query #12
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    definitions = Definitions()
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert not field.allow_null

    data = {"type": "integer", "minimum": 1, "maximum": 100}
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.maximum == 100
    assert not field.allow_null

    data = {"type": "number", "minimum": 1.0, "maximum": 100.0}
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 1.0
    assert field.maximum == 100.0
    assert not field.allow_null

    data = {"type": "boolean"}
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert not field.allow_null

    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert not field.allow_null

    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert not field.allow_null


# LLM-generated content at query #13
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():
    definitions = Definitions()
    definitions["#/components/schemas/User"] = String()
    
    # Test with a valid reference
    data = {"$ref": "#/components/schemas/User"}
    field = ref_from_json_schema(data, definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/User"
    
    # Test with an unsupported reference format (should raise an assertion error)
    try:
        data = {"$ref": "http://example.com/schema#/User"}
        ref_from_json_schema(data, definitions)
        assert False, "Expected assertion error for unsupported $ref style"
    except AssertionError as e:
        assert str(e) == "Unsupported $ref style in document."


# LLM-generated content at query #14
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    # Test case 1: Test with Any field
    field = Any()
    result = to_json_schema(field)
    assert result == True

    # Test case 2: Test with NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result == False

    # Test case 3: Test with String field
    field = String()
    result = to_json_schema(field)
    assert result == {'type': 'string'}

    # Test case 4: Test with Integer field
    field = Integer()
    result = to_json_schema(field)
    assert result == {'type': 'integer'}

    # Test case 5: Test with Float field
    field = Float()
    result = to_json_schema(field)
    assert result == {'type': 'number'}

    # Test case 6: Test with Boolean field
    field = Boolean()
    result = to_json_schema(field)
    assert result == {'type': 'boolean'}

    # Test case 7: Test with Array field
    field = Array(items=String())
    result = to_json_schema(field)
    assert result == {'type': 'array', 'items': {'type': 'string'}}

    # Test case 8: Test with Object field
    field = Object(properties={'name': String()})
    result = to_json_schema(field)
    assert result == {'type': 'object', 'properties': {'name': {'type': 'string'}}}

    # Test case 9: Test with Choice field
    field = Choice(choices=[('a', 'a')])
    result = to_json_schema(field)
    assert result == {'enum': ['a']}

    # Test case 10: Test with Const field
    field = Const(const='a')
    result = to_json_schema(field)
    assert result == {'const': 'a'}

    # Test case 11: Test with Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}

    # Test case 12: Test with OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]}

    # Test case 13: Test with AllOf field
    field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result == {'allOf': [{'type': 'string'}, {'type': 'integer'}]}

    # Test case 14: Test with IfThenElse field
    field = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(field)
    assert result == {'if': {'type': 'string'}, 'then': {'type': 'integer'}}

    # Test case 15: Test with Not field
    field = Not(negated=String())
    result = to_json_schema(field)
    assert result == {'not': {'type': 'string'}}


# LLM-generated content at query #15
#--------------------------

# Unit test for function from_json_schema
def test_from_json_schema():
    # Test case 1: Boolean input (True)
    result = from_json_schema(True)
    assert isinstance(result, Any)

    # Test case 2: Boolean input (False)
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

    # Test case 3: Dictionary input with $ref
    schema = {"$ref": "#/components/schemas/Example"}
    result = from_json_schema(schema)
    assert isinstance(result, Reference)

    # Test case 4: Dictionary input with type constraints
    schema = {"type": "string", "minLength": 5}
    result = from_json_schema(schema)
    assert isinstance(result, String)
    assert result.min_length == 5

    # Test case 5: Dictionary input with enum
    schema = {"enum": ["red", "green", "blue"]}
    result = from_json_schema(schema)
    assert isinstance(result, Choice)
    assert result.choices == [("red", "red"), ("green", "green"), ("blue", "blue")]

    # Test case 6: Dictionary input with const
    schema = {"const": "example"}
    result = from_json_schema(schema)
    assert isinstance(result, Const)
    assert result.const == "example"

    # Test case 7: Dictionary input with allOf
    schema = {"allOf": [{"type": "string"}, {"minLength": 3}]}
    result = from_json_schema(schema)
    assert isinstance(result, AllOf)

    # Test case 8: Dictionary input with anyOf
    schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(schema)
    assert isinstance(result, OneOf)

    # Test case 9: Dictionary input with oneOf
    schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(schema)
    assert isinstance(result, OneOf)

    # Test case 10: Dictionary input with not
    schema = {"not": {"type": "string"}}
    result = from_json_schema(schema)
    assert isinstance(result, Not)

    # Test case 11: Dictionary input with if-then-else
    schema = {"if": {"type": "string"}, "then": {"minLength": 5}, "else": {"type": "number"}}
    result = from_json_schema(schema)
    assert isinstance(result, IfThenElse)

    # Test case 12: Dictionary input with multiple constraints
    schema = {"type": "string", "minLength": 5, "enum": ["hello", "world"]}
    result = from_json_schema(schema)
    assert isinstance(result, AllOf)


# LLM-generated content at query #16
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    def assert_json_schema_equal(expected, actual):
        assert json.dumps(expected, sort_keys=True) == json.dumps(
            actual, sort_keys=True
        )

    assert_json_schema_equal(True, to_json_schema(Any()))
    assert_json_schema_equal(False, to_json_schema(NeverMatch()))

    assert_json_schema_equal(
        {"type": "string", "minLength": 1},
        to_json_schema(String(min_length=1)),
    )

    assert_json_schema_equal(
        {"type": "number", "minimum": 0, "maximum": 100},
        to_json_schema(Float(minimum=0, maximum=100)),
    )

    assert_json_schema_equal(
        {"type": "integer", "minimum": 0, "maximum": 100},
        to_json_schema(Integer(minimum=0, maximum=100)),
    )

    assert_json_schema_equal(
        {"type": "boolean"}, to_json_schema(Boolean())
    )

    assert_json_schema_equal(
        {"type": "array", "items": {"type": "string"}},
        to_json_schema(Array(items=String())),
    )

    assert_json_schema_equal(
        {"type": "object", "properties": {"name": {"type": "string"}}},
        to_json_schema(Object(properties={"name": String()})),
    )

    assert_json_schema_equal(
        {"enum": ["red", "green", "blue"]},
        to_json_schema(Choice(choices=[("red", "red"), ("green", "green"), ("blue", "blue")])),
    )

    assert_json_schema_equal(
        {"const": "red"},
        to_json_schema(Const(const="red")),
    )

    assert_json_schema_equal(
        {"anyOf": [{"type": "string"}, {"type": "number"}]},
        to_json_schema(Union(any_of=[String(), Float()])),
    )

    assert_json_schema_equal(
        {"oneOf": [{"type": "string"}, {"type": "number"}]},
        to_json_schema(OneOf(one_of=[String(), Float()])),
    )

    assert_json_schema_equal(
        {"allOf": [{"type": "string"}, {"type": "number"}]},
        to_json_schema(AllOf(all_of=[String(), Float()])),
    )

    assert_json_schema_equal(
        {"if": {"type": "string"}, "then": {"type": "number"}},
        to_json_schema(IfThenElse(if_clause=String(), then_clause=Float())),
    )

    assert_json_schema_equal(
        {"not": {"type": "string"}},
        to_json_schema(Not(negated=String())),
    )

    assert_json_schema_equal(
        {"components": {"schemas": {"Person": {"type": "object", "properties": {"name": {"type": "string"}}}}}},
        to_json_schema(Definitions({"Person": Object(properties={"name": String()})})),
    )

    assert_json_schema_equal(
        {"$ref": "#/components/schemas/Person", "components": {"schemas": {"Person": {"type": "object", "properties": {"name": {"type": "string"}}}}}},
        to_json_schema(Reference(to="Person", target=Object(properties={"name": String()}))),
    )


# LLM-generated content at query #17
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    # Test with Any field
    any_field = Any()
    assert to_json_schema(any_field) == True

    # Test with NeverMatch field
    never_match_field = NeverMatch()
    assert to_json_schema(never_match_field) == False

    # Test with String field
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern="^[a-z]*$", format="email")
    expected_string_schema = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]*$",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected_string_schema

    # Test with Integer field
    integer_field = Integer(allow_null=True, minimum=1, maximum=100, multiple_of=2)
    expected_integer_schema = {
        "type": ["integer", "null"],
        "minimum": 1,
        "maximum": 100,
        "multipleOf": 2
    }
    assert to_json_schema(integer_field) == expected_integer_schema

    # Test with Boolean field
    boolean_field = Boolean(allow_null=True)
    expected_boolean_schema = {
        "type": ["boolean", "null"]
    }
    assert to_json_schema(boolean_field) == expected_boolean_schema

    # Test with Array field
    array_field = Array(
        allow_null=True,
        min_items=1,
        max_items=10,
        items=String(),
        additional_items=False,
        unique_items=True
    )
    expected_array_schema = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 10,
        "items": {"type": "string"},
        "additionalItems": False,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected_array_schema

    # Test with Object field
    object_field = Object(
        allow_null=True,
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=2
    )
    expected_object_schema = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 2
    }
    assert to_json_schema(object_field) == expected_object_schema

    # Test with Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    expected_choice_schema = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected_choice_schema

    # Test with Const field
    const_field = Const(const="test")
    expected_const_schema = {
        "const": "test"
    }
    assert to_json_schema(const_field) == expected_const_schema

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    expected_union_schema = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected_union_schema

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    expected_all_of_schema = {
        "allOf": [{"type": "string", "minLength": 1}, {"type": "string", "maxLength": 10}]
    }
    assert to_json_schema(all_of_field) == expected_all_of_schema

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected_if_then_else_schema = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_then_else_field) == expected_if_then_else_schema

    # Test with Not field
    not_field = Not(negated=String())
    expected_not_schema = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected_not_schema

    # Test with Reference field
    definitions = {"Test": String()}
    reference_field = Reference(to="Test", definitions=definitions)
    expected_reference_schema = {
        "$ref": "#/components/schemas/Test",
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(reference_field, _definitions=definitions) == expected_reference_schema

    print("All tests passed!")

test_to_json_schema()


# LLM-generated content at query #18
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    definitions = Definitions()
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-zA-Z]+$",
        "format": "email",
        "default": "example@example.com"
    }
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^[a-zA-Z]+$"
    assert field.format == "email"
    assert field.default == "example@example.com"

    data = {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 101,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 1
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 101
    assert field.multiple_of == 2
    assert field.default == 50

    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True

    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "default": ["item1"]
    }
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items == True
    assert field.default == ["item1"]

    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "minProperties": 1,
        "maxProperties": 2,
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}


# LLM-generated content at query #19
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    # Test with type "number"
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 10,
        "multipleOf": 2,
        "default": 4,
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 4

    # Test with type "integer"
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 10,
        "multipleOf": 2,
        "default": 4,
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 4

    # Test with type "string"
    data = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "format": "email",
        "pattern": "^[a-z]+$",
        "default": "test",
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == re.compile("^[a-z]+$")
    assert field.default == "test"

    # Test with type "boolean"
    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test with type "array"
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"],
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test with type "object"
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test"},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}


# LLM-generated content at query #20
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    field = String(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["string", "null"]}


# LLM-generated content at query #21
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    definitions = Definitions()
    data = {"type": "number", "minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5

    data = {"type": "integer", "minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5

    data = {"type": "string", "minLength": 1, "maxLength": 10, "format": "email", "pattern": "^[a-zA-Z0-9]+$", "default": "test"}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9]+$"
    assert field.default == "test"

    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True

    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10, "uniqueItems": True, "default": ["test"]}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, Field)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    data = {"type": "object", "properties": {"name": {"type": "string"}}, "minProperties": 1, "maxProperties": 10, "required": ["name"], "default": {"name": "test"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert field.properties is not None
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}


# LLM-generated content at query #22
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():
    # Test with a simple string field
    field = String()
    schema = to_json_schema(field)
    assert schema == {"type": "string"}

    # Test with a string field with null allowed
    field = String(allow_null=True)
    schema = to_json_schema(field)
    assert schema == {"type": ["string", "null"]}

    # Test with an integer field
    field = Integer()
    schema = to_json_schema(field)
    assert schema == {"type": "integer"}

    # Test with a float field
    field = Float()
    schema = to_json_schema(field)
    assert schema == {"type": "number"}

    # Test with a boolean field
    field = Boolean()
    schema = to_json_schema(field)
    assert schema == {"type": "boolean"}

    # Test with an array field
    field = Array(items=String())
    schema = to_json_schema(field)
    assert schema == {"type": "array", "items": {"type": "string"}}

    # Test with an object field
    field = Object(properties={"name": String()})
    schema = to_json_schema(field)
    assert schema == {"type": "object", "properties": {"name": {"type": "string"}}}

    # Test with a choice field
    field = Choice(choices=[("a", "A"), ("b", "B")])
    schema = to_json_schema(field)
    assert schema == {"enum": ["a", "b"]}

    # Test with a const field
    field = Const(const="test")
    schema = to_json_schema(field)
    assert schema == {"const": "test"}

    # Test with a union field
    field = Union(any_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert schema == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

    # Test with a oneOf field
    field = OneOf(one_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert schema == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

    # Test with an allOf field
    field = AllOf(all_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert schema == {"allOf": [{"type": "string"}, {"type": "integer"}]}

    # Test with an if-then-else field
    field = IfThenElse(if_clause=String(), then_clause=Integer())
    schema = to_json_schema(field)
    assert schema == {
        "if": {"type": "string"},
        "then": {"type": "integer"},
    }

    # Test with a not field
    field = Not(negated=String())
    schema = to_json_schema(field)
    assert schema == {"not": {"type": "string"}}

    # Test with a reference field
    definitions = {"Person": Object(properties={"name": String()})}
    field = Reference(to="Person", definitions=definitions)
    schema = to_json_schema(field)
    assert schema == {
        "$ref": "#/components/schemas/Person",
        "components": {
            "schemas": {
                "Person": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            }
        },
    }

    # Test with a schema field
    field = Schema(fields={"name": String()}, required=["name"])
    schema = to_json_schema(field)
    assert schema == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    # Test with a never match field
    field = NeverMatch()
    schema = to_json_schema(field)
    assert schema is False

    # Test with an any field
    field = Any()
    schema = to_json_schema(field)
    assert schema is True


# LLM-generated content at query #23
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    data = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-zA-Z0-9_]+$",
        "format": "email",
        "default": "test@example.com"
    }
    expected = String(
        allow_null=False,
        allow_blank=False,
        min_length=1,
        max_length=10,
        pattern="^[a-zA-Z0-9_]+$",
        format="email",
        default="test@example.com",
        coerce_types=False
    )
    assert from_json_schema_type(data, "string", False, Definitions()) == expected

    data = {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 101,
        "multipleOf": 2,
        "default": 50
    }
    expected = Integer(
        allow_null=False,
        minimum=1,
        maximum=100,
        exclusive_minimum=0,
        exclusive_maximum=101,
        multiple_of=2,
        default=50,
        coerce_types=False
    )
    assert from_json_schema_type(data, "integer", False, Definitions()) == expected

    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "additionalItems": True,
        "uniqueItems": True,
        "default": ["item1"]
    }
    expected = Array(
        allow_null=False,
        min_items=1,
        max_items=10,
        additional_items=True,
        items=String(allow_null=False, coerce_types=False),
        unique_items=True,
        default=["item1"]
    )
    assert from_json_schema_type(data, "array", False, Definitions()) == expected

    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "minProperties": 1,
        "maxProperties": 2,
        "additionalProperties": False,
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    expected = Object(
        allow_null=False,
        properties={
            "name": String(allow_null=False, coerce_types=False),
            "age": Integer(allow_null=False, coerce_types=False)
        },
        min_properties=1,
        max_properties=2,
        additional_properties=False,
        required=["name"],
        default={"name": "John", "age": 30}
    )
    assert from_json_schema_type(data, "object", False, Definitions()) == expected


# LLM-generated content at query #24
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type():
    definitions = Definitions()
    # Test with type "number"
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 10,
        "multipleOf": 2,
        "default": 4,
    }
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 4

    # Test with type "integer"
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 10,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 10,
        "multipleOf": 2,
        "default": 4,
    }
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 4

    # Test with type "string"
    data = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "format": "email",
        "default": "test",
    }
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.pattern == re.compile("^[a-z]+$")
    assert field.format == "email"
    assert field.default == "test"

    # Test with type "boolean"
    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test with type "array"
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"],
    }
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test with type "object"
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test"},
    }
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}

    # Test with allow_null=True
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, definitions)
    assert isinstance(field, String)
    assert field.allow_null == True


