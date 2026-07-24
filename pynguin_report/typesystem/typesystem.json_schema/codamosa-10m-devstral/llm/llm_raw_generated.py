####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_all_of_from_json_schema():
    definitions = Definitions()
    data = {
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ],
        "default": "test"
    }
    field = all_of_from_json_schema(data, definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2
    assert isinstance(field.all_of[0], String)
    assert isinstance(field.all_of[1], String)
    assert field.default == "test"

    # Test with nested allOf
    nested_data = {
        "allOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"allOf": [{"type": "object", "properties": {"age": {"type": "integer"}}}]}
        ]
    }
    nested_field = all_of_from_json_schema(nested_data, definitions)
    assert isinstance(nested_field, AllOf)
    assert len(nested_field.all_of) == 2
    assert isinstance(nested_field.all_of[0], Object)
    assert isinstance(nested_field.all_of[1], AllOf)


# LLM-generated content at query #2
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test simple type
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)

    # Test enum
    enum_schema = {"enum": ["a", "b", "c"]}
    enum_field = from_json_schema(enum_schema)
    assert isinstance(enum_field, Choice)
    assert enum_field.choices == ["a", "b", "c"]

    # Test const
    const_schema = {"const": "fixed_value"}
    const_field = from_json_schema(const_schema)
    assert isinstance(const_field, Const)
    assert const_field.value == "fixed_value"

    # Test allOf
    all_of_schema = {
        "allOf": [
            {"type": "string"},
            {"minLength": 5}
        ]
    }
    all_of_field = from_json_schema(all_of_schema)
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.schemas) == 2

    # Test anyOf
    any_of_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    any_of_field = from_json_schema(any_of_schema)
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.schemas) == 2

    # Test oneOf
    one_of_schema = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    one_of_field = from_json_schema(one_of_schema)
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.schemas) == 2

    # Test not
    not_schema = {
        "not": {"type": "string"}
    }
    not_field = from_json_schema(not_schema)
    assert isinstance(not_field, Not)
    assert isinstance(not_field.schema, String)

    # Test if-then-else
    if_then_else_schema = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"}
    }
    if_then_else_field = from_json_schema(if_then_else_schema)
    assert isinstance(if_then_else_field, IfThenElse)
    assert isinstance(if_then_else_field.if_schema, String)
    assert isinstance(if_then_else_field.then_schema, String)
    assert isinstance(if_then_else_field.else_schema, Number)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    ref_schema = {"$ref": "#/components/schemas/Test"}
    ref_field = from_json_schema(ref_schema, definitions=definitions)
    assert isinstance(ref_field, Reference)
    assert ref_field.ref == "#/components/schemas/Test"

    # Test combined constraints
    combined_schema = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-zA-Z]+$"
    }
    combined_field = from_json_schema(combined_schema)
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.schemas) == 4


# LLM-generated content at query #3
#--------------------------

```python
def test_ref_from_json_schema():
    # Test with a valid reference string
    data = {"$ref": "#/components/schemas/User"}
    definitions = Definitions()
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"
    assert result.definitions == definitions

    # Test with an invalid reference string (not starting with "#/")
    data = {"$ref": "components/schemas/User"}
    definitions = Definitions()
    with pytest.raises(AssertionError):
        ref_from_json_schema(data, definitions)


# LLM-generated content at query #4
#--------------------------

```python
def test_if_then_else_from_json_schema():
    definitions = Definitions()

    # Test with if, then, and else clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, Number)
    assert isinstance(result.else_clause, Boolean)

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, Number)
    assert result.else_clause is None

    # Test with only if clause
    data = {
        "if": {"type": "string"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"},
        "default": 42
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert result.default == 42


# LLM-generated content at query #5
#--------------------------

```python
def test_one_of_from_json_schema():
    definitions = Definitions()
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "default_value"
    }
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Number)
    assert result.default == "default_value"


# LLM-generated content at query #6
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9]+$"
    assert field.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
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

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 10,
        "default": {"name": "test", "age": 25}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.default == {"name": "test", "age": 25}


# LLM-generated content at query #7
#--------------------------

```python
def test_type_from_json_schema():
    # Test with a single type
    data = {"type": "string"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)

    # Test with multiple types (Union)
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert any(isinstance(field, String) for field in result.any_of)
    assert any(isinstance(field, Number) for field in result.any_of)

    # Test with allow_null
    data = {"type": "string", "nullable": True}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)
    assert result.allow_null is True

    # Test with no type specified (allow_null)
    data = {"nullable": True}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test with no type specified (no allow_null)
    data = {}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, NeverMatch)

    # Test with additional constraints
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10

    # Test with array type
    data = {"type": "array", "items": {"type": "string"}}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

    # Test with object type
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert isinstance(result.properties["name"], String)

    # Test with number type and constraints
    data = {"type": "number", "minimum": 0, "maximum": 100}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Number)
    assert result.minimum == 0
    assert result.maximum == 100

    # Test with boolean type
    data = {"type": "boolean"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Boolean)

    # Test with integer type
    data = {"type": "integer"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Integer)


# LLM-generated content at query #8
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_maximum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMaximum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean()
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    obj_field = Object(
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=5
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 5
    }
    assert to_json_schema(obj_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {"enum": ["a", "b"]}
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="fixed_value")
    expected = {"const": "fixed_value"}
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(union_field) == expected

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(one_of_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Integer()])
    expected = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(all_of_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected

    # Test Reference field
    ref_field = Reference(to="test", target=String(), definitions={"test": String()})
    expected = {
        "$ref": "#/components/schemas/test",
        "components": {"schemas": {"test": {"type": "string"}}}
    }
    assert to_json_schema(ref_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test Definitions
    definitions = Definitions({"field1": String(), "field2": Integer()})
    expected = {
        "components": {
            "schemas": {
                "field1": {"type": "string"},
                "field2": {"type": "integer"}
            }
        }
    }
    assert to_json_schema(definitions) == expected

    # Test nullable fields
    nullable_string = String(allow_null=True)
    expected = {"type": ["string", "null"]}
    assert to_json_schema(nullable_string) == expected

    # Test default values
    string_with_default = String(default="default_value")
    expected = {"type": "string", "default": "default_value"}
    assert to_json_schema(string_with_default) == expected


# LLM-generated content at query #9
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "[a-z]+"
    assert result["format"] == "email"

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100

    # Test Float field
    float_field = Float(minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test Boolean field
    bool_field = Boolean()
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=10)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["items"]["type"] == "string"

    # Test Object field
    object_field = Object(properties={"name": String()}, required=["name"])
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert result["properties"]["name"]["type"] == "string"
    assert result["required"] == ["name"]

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="fixed")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert result["anyOf"] == [{"type": "string"}, {"type": "integer"}]

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert result["oneOf"] == [{"type": "string"}, {"type": "integer"}]

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Integer()])
    result = to_json_schema(all_of_field)
    assert result["allOf"] == [{"type": "string"}, {"type": "integer"}]

    # Test IfThenElse field
    if_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(if_field)
    assert result["if"]["type"] == "string"
    assert result["then"]["type"] == "integer"
    assert result["else"]["type"] == "boolean"

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert result["not"]["type"] == "string"

    # Test Reference field
    definitions = Definitions()
    ref_field = Reference(to="test", definitions=definitions)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/test"

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert result["properties"]["name"]["type"] == "string"
    assert result["required"] == ["name"]

    # Test Definitions
    definitions = Definitions()
    definitions["test"] = String()
    result = to_json_schema(definitions)
    assert result["components"]["schemas"]["test"]["type"] == "string"


# LLM-generated content at query #10
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type
    data = {"type": "string"}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, String)

    # Test union of types
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert any(isinstance(field, String) for field in result.any_of)
    assert any(isinstance(field, Number) for field in result.any_of)

    # Test nullable type
    data = {"type": "string", "nullable": True}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, String)
    assert result.allow_null is True

    # Test no type specified (allow_null)
    data = {"nullable": True}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test no type specified (not nullable)
    data = {}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, NeverMatch)

    # Test with definitions
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = Object(properties={"name": String()})
    data = {"$ref": "#/components/schemas/Test"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.ref == "#/components/schemas/Test"


# LLM-generated content at query #11
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected_string_schema = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected_string_schema

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True)
    expected_int_schema = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True
    }
    assert to_json_schema(int_field) == expected_int_schema

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected_float_schema = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected_float_schema

    # Test Boolean field
    bool_field = Boolean()
    expected_bool_schema = {
        "type": "boolean"
    }
    assert to_json_schema(bool_field) == expected_bool_schema

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected_array_schema = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected_array_schema

    # Test Object field
    object_field = Object(
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=5
    )
    expected_object_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 5
    }
    assert to_json_schema(object_field) == expected_object_schema

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected_choice_schema = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected_choice_schema

    # Test Const field
    const_field = Const(const="fixed_value")
    expected_const_schema = {
        "const": "fixed_value"
    }
    assert to_json_schema(const_field) == expected_const_schema

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected_union_schema = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected_union_schema

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Integer()])
    expected_all_of_schema = {
        "allOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(all_of_field) == expected_all_of_schema

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected_one_of_schema = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(one_of_field) == expected_one_of_schema

    # Test Not field
    not_field = Not(negated=String())
    expected_not_schema = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected_not_schema

    # Test IfThenElse field
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

    # Test Reference field
    definitions = Definitions()
    ref_field = Reference(to="test_ref", definitions=definitions)
    expected_ref_schema = {
        "$ref": "#/components/schemas/test_ref",
        "components": {
            "schemas": {}
        }
    }
    assert to_json_schema(ref_field) == expected_ref_schema

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected_schema_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected_schema_schema

    # Test allow_null fields
    string_field_with_null = String(allow_null=True)
    expected_string_with_null_schema = {
        "type": ["string", "null"]
    }
    assert to_json_schema(string_field_with_null) == expected_string_with_null_schema

    # Test default values
    string_field_with_default = String(default="default_value")
    expected_string_with_default_schema = {
        "type": "string",
        "default": "default_value"
    }
    assert to_json_schema(string_field_with_default) == expected_string_with_default_schema


# LLM-generated content at query #12
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(allow_null=True, multiple_of=0.5)
    expected = {
        "type": ["number", "null"],
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(
        allow_null=True,
        items=String(),
        additional_items=False,
        min_items=1,
        max_items=5,
        unique_items=True
    )
    expected = {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String()},
        additional_properties=False,
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "additionalProperties": False,
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"]
    }
    assert to_json_schema(object_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")], default="a")
    expected = {
        "enum": ["a", "b"],
        "default": "a"
    }
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const=42, default=42)
    expected = {
        "const": 42,
        "default": 42
    }
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    assert to_json_schema(union_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const("test")])
    expected = {
        "allOf": [
            {"type": "string"},
            {"const": "test"}
        ]
    }
    assert to_json_schema(all_of_field) == expected

    # Test Reference field
    definitions = Definitions({"Test": String()})
    ref_field = Reference(to="Test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(ref_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=Const(True),
        then_clause=String(),
        else_clause=Integer()
    )
    expected = {
        "if": {"const": True},
        "then": {"type": "string"},
        "else": {"type": "integer"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected


# LLM-generated content at query #13
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test type constraints
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)

    # Test enum constraint
    enum_schema = {"enum": ["a", "b", "c"]}
    assert isinstance(from_json_schema(enum_schema), Choice)
    assert from_json_schema(enum_schema).choices == ["a", "b", "c"]

    # Test const constraint
    const_schema = {"const": "value"}
    assert isinstance(from_json_schema(const_schema), Const)
    assert from_json_schema(const_schema).value == "value"

    # Test allOf constraint
    all_of_schema = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    result = from_json_schema(all_of_schema)
    assert isinstance(result, AllOf)
    assert len(result.schemas) == 2

    # Test anyOf constraint
    any_of_schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(any_of_schema)
    assert isinstance(result, OneOf)
    assert len(result.schemas) == 2

    # Test oneOf constraint
    one_of_schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(one_of_schema)
    assert isinstance(result, OneOf)
    assert len(result.schemas) == 2

    # Test not constraint
    not_schema = {"not": {"type": "string"}}
    result = from_json_schema(not_schema)
    assert isinstance(result, Not)
    assert isinstance(result.schema, String)

    # Test if-then-else constraint
    if_then_else_schema = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    }
    result = from_json_schema(if_then_else_schema)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_schema, String)
    assert isinstance(result.then_schema, String)
    assert isinstance(result.else_schema, Integer)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    ref_schema = {"$ref": "#/components/schemas/Test"}
    result = from_json_schema(ref_schema, definitions=definitions)
    assert isinstance(result, Reference)
    assert result.reference == "#/components/schemas/Test"

    # Test combined constraints
    combined_schema = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-zA-Z]+$"
    }
    result = from_json_schema(combined_schema)
    assert isinstance(result, AllOf)
    assert len(result.schemas) == 2  # type and pattern constraints

    # Test empty schema
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #14
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "format": "email",
        "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$",
        "default": "test@example.com"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
    assert field.default == "test@example.com"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
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

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #15
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9]+$"
    assert field.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items == False
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "patternProperties": {
            "^S_": {"type": "string"},
            "^I_": {"type": "integer"}
        },
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test", "age": 25}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert isinstance(field.pattern_properties["^S_"], String)
    assert isinstance(field.pattern_properties["^I_"], Integer)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 25}


# LLM-generated content at query #16
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "pattern": "^[a-zA-Z]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.pattern == "^[a-zA-Z]+$"
    assert field.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
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

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "default": {"name": "test", "age": 25}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 25}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #17
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type
    data = {"type": "string"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)

    # Test union of types
    data = {"type": ["string", "integer"]}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Integer)

    # Test nullable type
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test empty type with allow_null
    data = {"type": []}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, NeverMatch)

    # Test empty type with allow_null and null in type
    data = {"type": ["null"]}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test constraints on string type
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10

    # Test constraints on number type
    data = {"type": "number", "minimum": 0, "maximum": 100}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Number)
    assert result.minimum == 0
    assert result.maximum == 100

    # Test constraints on array type
    data = {"type": "array", "minItems": 1, "maxItems": 10}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Array)
    assert result.min_items == 1
    assert result.max_items == 10

    # Test constraints on object type
    data = {"type": "object", "minProperties": 1, "maxProperties": 10}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Object)
    assert result.min_properties == 1
    assert result.max_properties == 10


# LLM-generated content at query #18
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9]+$"
    assert field.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
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

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 2,
        "default": {"name": "test", "age": 25}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.default == {"name": "test", "age": 25}

    # Test invalid type_string
    try:
        from_json_schema_type({}, "invalid", False, Definitions())
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid'"


# LLM-generated content at query #19
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_maximum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMaximum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    obj_field = Object(
        properties={"name": String()},
        required=["name"],
        additional_properties=False
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False
    }
    assert to_json_schema(obj_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="fixed_value")
    expected = {
        "const": "fixed_value"
    }
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const("test")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "test"}]
    }
    assert to_json_schema(all_of_field) == expected

    # Test Reference field
    definitions = Definitions({"TestSchema": String()})
    ref_field = Reference(to="TestSchema", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/TestSchema",
        "components": {
            "schemas": {
                "TestSchema": {"type": "string"}
            }
        }
    }
    assert to_json_schema(ref_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected


# LLM-generated content at query #20
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9]+$"
    assert field.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items == False
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "patternProperties": {
            "^S_": {"type": "string"},
            "^I_": {"type": "integer"}
        },
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert isinstance(field.pattern_properties["^S_"], String)
    assert isinstance(field.pattern_properties["^I_"], Integer)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #21
#--------------------------

```python
def test_to_json_schema():
    # Test Any
    assert to_json_schema(Any()) == True

    # Test NeverMatch
    assert to_json_schema(NeverMatch()) == False

    # Test String
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float
    float_field = Float(allow_null=True, minimum=0.0, maximum=1.0, multiple_of=0.1)
    expected = {
        "type": ["number", "null"],
        "minimum": 0.0,
        "maximum": 1.0,
        "multipleOf": 0.1
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean
    bool_field = Boolean(allow_null=True)
    expected = {"type": ["boolean", "null"]}
    assert to_json_schema(bool_field) == expected

    # Test Array
    array_field = Array(allow_null=False, items=String(), additional_items=False, min_items=1, max_items=10)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 10
    }
    assert to_json_schema(array_field) == expected

    # Test Object
    object_field = Object(
        allow_null=True,
        properties={"name": String()},
        additional_properties=False,
        required=["name"]
    )
    expected = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}},
        "additionalProperties": False,
        "required": ["name"]
    }
    assert to_json_schema(object_field) == expected

    # Test Choice
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {"enum": ["a", "b"]}
    assert to_json_schema(choice_field) == expected

    # Test Const
    const_field = Const(const="fixed_value")
    expected = {"const": "fixed_value"}
    assert to_json_schema(const_field) == expected

    # Test Union
    union_field = Union(any_of=[String(), Integer()])
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(union_field) == expected

    # Test AllOf
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(all_of_field) == expected

    # Test OneOf
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(one_of_field) == expected

    # Test IfThenElse
    if_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not
    not_field = Not(negated=String())
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected

    # Test Reference
    definitions = Definitions({"Test": String()})
    ref_field = Reference(to="Test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {"schemas": {"Test": {"type": "string"}}}
    }
    assert to_json_schema(ref_field) == expected


# LLM-generated content at query #22
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "default": 50.5,
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50.5

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 50,
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "pattern": "^[A-Za-z]+$",
        "default": "hello",
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.pattern == "^[A-Za-z]+$"
    assert field.default == "hello"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True,
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default is True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["item1", "item2"],
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items is True
    assert field.default == ["item1", "item2"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
        "default": {"name": "John", "age": 30},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #23
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "default": 50.5
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50.5

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "format": "email",
        "default": "test@example.com"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.format == "email"
    assert field.default == "test@example.com"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default is True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "default": ["item1", "item2"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.default == ["item1", "item2"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #24
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test simple type schema
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)

    # Test enum schema
    enum_schema = {"enum": ["a", "b", "c"]}
    result = from_json_schema(enum_schema)
    assert isinstance(result, Choice)
    assert result.choices == ["a", "b", "c"]

    # Test const schema
    const_schema = {"const": "fixed_value"}
    result = from_json_schema(const_schema)
    assert isinstance(result, Const)
    assert result.value == "fixed_value"

    # Test allOf schema
    all_of_schema = {
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    }
    result = from_json_schema(all_of_schema)
    assert isinstance(result, AllOf)
    assert len(result.schemas) == 2

    # Test anyOf schema
    any_of_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result = from_json_schema(any_of_schema)
    assert isinstance(result, OneOf)
    assert len(result.schemas) == 2

    # Test oneOf schema
    one_of_schema = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result = from_json_schema(one_of_schema)
    assert isinstance(result, OneOf)
    assert len(result.schemas) == 2

    # Test not schema
    not_schema = {"not": {"type": "string"}}
    result = from_json_schema(not_schema)
    assert isinstance(result, Not)

    # Test if-then-else schema
    if_then_else_schema = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"minLength": 10}
    }
    result = from_json_schema(if_then_else_schema)
    assert isinstance(result, IfThenElse)

    # Test reference schema
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = {"type": "string"}
    ref_schema = {"$ref": "#/components/schemas/Test"}
    result = from_json_schema(ref_schema, definitions=definitions)
    assert isinstance(result, Reference)

    # Test combined constraints
    combined_schema = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    }
    result = from_json_schema(combined_schema)
    assert isinstance(result, AllOf)
    assert len(result.schemas) == 2

    # Test empty schema
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #25
#--------------------------

```python
def test_to_json_schema():
    # Test Any
    assert to_json_schema(Any()) == True

    # Test NeverMatch
    assert to_json_schema(NeverMatch()) == False

    # Test String
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float
    float_field = Float(allow_null=True, multiple_of=0.5)
    expected = {
        "type": ["number", "null"],
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean
    bool_field = Boolean(allow_null=False)
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array
    array_field = Array(allow_null=True, items=String(), min_items=1, max_items=5)
    expected = {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5
    }
    assert to_json_schema(array_field) == expected

    # Test Object
    obj_field = Object(
        allow_null=False,
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=5
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 5
    }
    assert to_json_schema(obj_field) == expected

    # Test Choice
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {"enum": ["a", "b"]}
    assert to_json_schema(choice_field) == expected

    # Test Const
    const_field = Const(const="fixed_value")
    expected = {"const": "fixed_value"}
    assert to_json_schema(const_field) == expected

    # Test Union
    union_field = Union(any_of=[String(), Integer()])
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(union_field) == expected

    # Test AllOf
    all_of_field = AllOf(all_of=[String(), Const("test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(all_of_field) == expected

    # Test OneOf
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(one_of_field) == expected

    # Test Not
    not_field = Not(negated=String())
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected

    # Test IfThenElse
    if_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Reference
    ref_field = Reference(to="Test", target=String())
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {"schemas": {"Test": {"type": "string"}}}
    }
    assert to_json_schema(ref_field) == expected

    # Test Schema
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test Definitions
    definitions = Definitions({"StringField": String(), "IntField": Integer()})
    expected = {
        "components": {
            "schemas": {
                "StringField": {"type": "string"},
                "IntField": {"type": "integer"}
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #26
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[A-Za-z]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^[A-Za-z]+$"
    assert field.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default is True

    # Test array type
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
    assert field.unique_items is True
    assert field.default == ["test"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "default": {"name": "test", "age": 25}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 25}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #27
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test with definitions
    data_with_definitions = {
        "components": {
            "schemas": {
                "test_schema": {"type": "string"}
            }
        }
    }
    definitions = Definitions()
    result = from_json_schema(data_with_definitions, definitions=definitions)
    assert isinstance(result, Any)

    # Test $ref
    data_with_ref = {"$ref": "#/components/schemas/test_schema"}
    definitions["#/components/schemas/test_schema"] = String()
    result = from_json_schema(data_with_ref, definitions=definitions)
    assert isinstance(result, Reference)

    # Test type constraints
    data_with_type = {"type": "string", "minLength": 1}
    result = from_json_schema(data_with_type)
    assert isinstance(result, String)

    # Test enum
    data_with_enum = {"enum": ["a", "b", "c"]}
    result = from_json_schema(data_with_enum)
    assert isinstance(result, Choice)

    # Test const
    data_with_const = {"const": "test"}
    result = from_json_schema(data_with_const)
    assert isinstance(result, Const)

    # Test allOf
    data_with_all_of = {"allOf": [{"type": "string"}, {"minLength": 1}]}
    result = from_json_schema(data_with_all_of)
    assert isinstance(result, AllOf)

    # Test anyOf
    data_with_any_of = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data_with_any_of)
    assert isinstance(result, OneOf)

    # Test oneOf
    data_with_one_of = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(data_with_one_of)
    assert isinstance(result, OneOf)

    # Test not
    data_with_not = {"not": {"type": "string"}}
    result = from_json_schema(data_with_not)
    assert isinstance(result, Not)

    # Test if-then-else
    data_with_if_then_else = {
        "if": {"type": "string"},
        "then": {"minLength": 1},
        "else": {"minLength": 0}
    }
    result = from_json_schema(data_with_if_then_else)
    assert isinstance(result, IfThenElse)

    # Test multiple constraints
    data_with_multiple_constraints = {
        "type": "string",
        "minLength": 1,
        "enum": ["a", "b", "c"]
    }
    result = from_json_schema(data_with_multiple_constraints)
    assert isinstance(result, AllOf)

    # Test no constraints
    data_with_no_constraints = {}
    result = from_json_schema(data_with_no_constraints)
    assert isinstance(result, Any)


# LLM-generated content at query #28
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "format": "email",
        "pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "default": "test@example.com"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    assert field.default == "test@example.com"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["item1", "item2"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items == False
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["item1", "item2"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "patternProperties": {
            "^S_": {"type": "string"},
            "^I_": {"type": "integer"}
        },
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert isinstance(field.pattern_properties["^S_"], String)
    assert isinstance(field.pattern_properties["^I_"], Integer)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}

    # Test invalid type
    with pytest.raises(AssertionError):
        from_json_schema_type({"type": "invalid"}, "invalid", False, Definitions())


# LLM-generated content at query #29
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[A-Za-z]+$",
        "default": "hello"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^[A-Za-z]+$"
    assert field.default == "hello"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "default": ["hello"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items == True
    assert field.default == ["hello"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}


# LLM-generated content at query #30
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean()
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(
        items=String(),
        min_items=1,
        max_items=5,
        unique_items=True
    )
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=3
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 3
    }
    assert to_json_schema(object_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {"enum": ["a", "b"]}
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="fixed")
    expected = {"const": "fixed"}
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(union_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Integer()])
    expected = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(all_of_field) == expected

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(one_of_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Reference field
    ref_field = Reference(to="Test", target=String())
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {"schemas": {"Test": {"type": "string"}}}
    }
    assert to_json_schema(ref_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test with definitions
    definitions = Definitions({"Test": String()})
    expected = {
        "components": {
            "schemas": {"Test": {"type": "string"}}
        }
    }
    assert to_json_schema(definitions) == expected


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_enum_from_json_schema():
    data = {"enum": ["a", "b", "c"]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b"), ("c", "c")]
    assert field.default == NO_DEFAULT

    data_with_default = {"enum": [1, 2, 3], "default": 2}
    field_with_default = enum_from_json_schema(data_with_default, definitions=Definitions())
    assert field_with_default.choices == [(1, 1), (2, 2), (3, 3)]
    assert field_with_default.default == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_to_json_schema():
    # Test Any type
    assert to_json_schema(Any()) == True

    # Test NeverMatch type
    assert to_json_schema(NeverMatch()) == False

    # Test String type
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer type
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float type
    float_field = Float(minimum=0.0, maximum=1.0, multiple_of=0.1)
    expected = {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "multipleOf": 0.1
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean type
    bool_field = Boolean()
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array type
    array_field = Array(items=String(), min_items=1, max_items=10, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object type
    object_field = Object(
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 10
    }
    assert to_json_schema(object_field) == expected

    # Test Choice type
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {"enum": ["a", "b"]}
    assert to_json_schema(choice_field) == expected

    # Test Const type
    const_field = Const(const="fixed_value")
    expected = {"const": "fixed_value"}
    assert to_json_schema(const_field) == expected

    # Test Union type
    union_field = Union(any_of=[String(), Integer()])
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(union_field) == expected

    # Test OneOf type
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(one_of_field) == expected

    # Test AllOf type
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(all_of_field) == expected

    # Test IfThenElse type
    if_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not type
    not_field = Not(negated=String())
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected

    # Test Reference type
    definitions = Definitions()
    ref_field = Reference(to="test", definitions=definitions)
    definitions["test"] = String()
    expected = {"$ref": "#/components/schemas/test"}
    assert to_json_schema(ref_field) == expected

    # Test Schema type
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test with definitions
    definitions = Definitions()
    definitions["string_field"] = String()
    definitions["int_field"] = Integer()
    expected = {
        "components": {
            "schemas": {
                "string_field": {"type": "string"},
                "int_field": {"type": "integer"}
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5)
    expected = {
        "type": "array",
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"}
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
        properties={"name": String()},
        required=["name"]
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(object_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="fixed")
    expected = {
        "const": "fixed"
    }
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "test"}]
    }
    assert to_json_schema(all_of_field) == expected

    # Test Reference field
    definitions = Definitions({"Test": String()})
    ref_field = Reference(to="Test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(ref_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()})
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}}
    }
    assert to_json_schema(schema_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected


# LLM-generated content at query #4
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test type constraints
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)

    # Test enum constraint
    enum_field = from_json_schema({"enum": ["a", "b", "c"]})
    assert isinstance(enum_field, Choice)
    assert enum_field.choices == ["a", "b", "c"]

    # Test const constraint
    const_field = from_json_schema({"const": "fixed_value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "fixed_value"

    # Test allOf constraint
    all_of_field = from_json_schema({
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    })
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.schemas) == 2

    # Test anyOf constraint
    any_of_field = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    })
    assert isinstance(any_of_field, Union)
    assert len(any_of_field.schemas) == 2

    # Test oneOf constraint
    one_of_field = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    })
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.schemas) == 2

    # Test not constraint
    not_field = from_json_schema({
        "not": {"type": "string"}
    })
    assert isinstance(not_field, Not)
    assert isinstance(not_field.schema, String)

    # Test if-then-else constraint
    if_then_else_field = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"}
    })
    assert isinstance(if_then_else_field, IfThenElse)
    assert isinstance(if_then_else_field.if_schema, String)
    assert isinstance(if_then_else_field.then_schema, String)
    assert isinstance(if_then_else_field.else_schema, Number)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    ref_field = from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=definitions)
    assert isinstance(ref_field, Reference)
    assert ref_field.reference == "#/components/schemas/Test"

    # Test combined constraints
    combined_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    })
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.schemas) == 4

    # Test empty schema
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #5
#--------------------------

```python
def test_one_of_from_json_schema():
    definitions = Definitions()

    # Test with simple oneOf schema
    schema = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    field = one_of_from_json_schema(schema, definitions)
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2
    assert isinstance(field.one_of[0], String)
    assert isinstance(field.one_of[1], Number)

    # Test with nested oneOf schema
    schema = {
        "oneOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "array", "items": {"type": "integer"}}
        ]
    }
    field = one_of_from_json_schema(schema, definitions)
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2
    assert isinstance(field.one_of[0], Object)
    assert isinstance(field.one_of[1], Array)

    # Test with default value
    schema = {
        "oneOf": [
            {"type": "boolean"},
            {"type": "null"}
        ],
        "default": True
    }
    field = one_of_from_json_schema(schema, definitions)
    assert field.default == True

    # Test with reference in oneOf
    definitions["#/components/schemas/Test"] = String()
    schema = {
        "oneOf": [
            {"$ref": "#/components/schemas/Test"},
            {"type": "integer"}
        ]
    }
    field = one_of_from_json_schema(schema, definitions)
    assert isinstance(field, OneOf)
    assert isinstance(field.one_of[0], Reference)
    assert isinstance(field.one_of[1], Integer)


# LLM-generated content at query #6
#--------------------------

```python
def test_if_then_else_from_json_schema():
    definitions = Definitions()

    # Test with if, then, and else clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "number"}
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, String)
    assert isinstance(field.else_clause, Number)

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5}
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, String)
    assert field.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "number"}
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert field.then_clause is None
    assert isinstance(field.else_clause, Number)

    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "number"},
        "default": "default_value"
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert field.default == "default_value"


# LLM-generated content at query #7
#--------------------------

```python
def test_ref_from_json_schema():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/TestSchema"}

    # Test with valid reference
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/TestSchema"
    assert result.definitions is definitions

    # Test with unsupported reference style
    data_unsupported = {"$ref": "external/schema"}
    with pytest.raises(AssertionError):
        ref_from_json_schema(data_unsupported, definitions)


# LLM-generated content at query #8
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5)
    expected = {
        "type": "array",
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"}
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    obj_field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    assert to_json_schema(obj_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="fixed_value")
    expected = {
        "const": "fixed_value"
    }
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "test"}]
    }
    assert to_json_schema(all_of_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected

    # Test Reference field
    ref_field = Reference(to="test", target=String())
    expected = {
        "$ref": "#/components/schemas/test",
        "components": {
            "schemas": {
                "test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(ref_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test Definitions
    definitions = Definitions({"field1": String(), "field2": Integer()})
    expected = {
        "components": {
            "schemas": {
                "field1": {"type": "string"},
                "field2": {"type": "integer"}
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #9
#--------------------------

```python
def test_any_of_from_json_schema():
    # Test with simple anyOf schema
    data = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result = any_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Number)

    # Test with nested anyOf schema
    data = {
        "anyOf": [
            {"type": "string", "minLength": 1},
            {"type": "number", "minimum": 0},
            {"type": "boolean"}
        ]
    }
    result = any_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 3
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Number)
    assert isinstance(result.any_of[2], Boolean)

    # Test with default value
    data = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test"
    }
    result = any_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert result.default == "test"

    # Test with reference in anyOf
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    data = {
        "anyOf": [
            {"$ref": "#/components/schemas/Test"},
            {"type": "number"}
        ]
    }
    result = any_of_from_json_schema(data, definitions=definitions)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], Reference)
    assert isinstance(result.any_of[1], Number)


# LLM-generated content at query #10
#--------------------------

```python
def test_if_then_else_from_json_schema():
    definitions = Definitions()

    # Test with if, then, and else clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "number"}
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, String)
    assert isinstance(field.else_clause, Number)

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5}
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, String)
    assert field.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "number"}
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert field.then_clause is None
    assert isinstance(field.else_clause, Number)

    # Test with only if clause
    data = {
        "if": {"type": "string"}
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert field.then_clause is None
    assert field.else_clause is None

    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "number"},
        "default": "default_value"
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert field.default == "default_value"


# LLM-generated content at query #11
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-zA-Z]+$",
        "default": "hello"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^[a-zA-Z]+$"
    assert field.default == "hello"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default is True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "default": ["hello"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items is True
    assert field.default == ["hello"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #12
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert from_json_schema(True) == Any()
    assert from_json_schema(False) == NeverMatch()

    # Test type constraints
    assert from_json_schema({"type": "string"}) == String()
    assert from_json_schema({"type": "integer"}) == Integer()
    assert from_json_schema({"type": "number"}) == Number()
    assert from_json_schema({"type": "boolean"}) == Boolean()
    assert from_json_schema({"type": "array"}) == Array(items=Any())
    assert from_json_schema({"type": "object"}) == Object()

    # Test enum
    assert from_json_schema({"enum": ["a", "b", "c"]}) == Choice(choices=["a", "b", "c"])

    # Test const
    assert from_json_schema({"const": "value"}) == Const(value="value")

    # Test allOf
    assert from_json_schema({"allOf": [{"type": "string"}, {"minLength": 5}]}) == AllOf([
        String(),
        String(min_length=5)
    ])

    # Test anyOf
    assert from_json_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]}) == OneOf([
        String(),
        Integer()
    ])

    # Test oneOf
    assert from_json_schema({"oneOf": [{"type": "string"}, {"type": "integer"}]}) == OneOf([
        String(),
        Integer()
    ])

    # Test not
    assert from_json_schema({"not": {"type": "string"}}) == Not(String())

    # Test if-then-else
    assert from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    }) == IfThenElse(
        if_schema=String(),
        then_schema=String(min_length=5),
        else_schema=Integer()
    )

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    assert from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=definitions) == Reference("Test", definitions=definitions)

    # Test combined constraints
    schema = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    }
    result = from_json_schema(schema)
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.pattern == "^[a-z]+$"


# LLM-generated content at query #13
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(properties={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(object_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="fixed")
    expected = {
        "const": "fixed"
    }
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const("test")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "test"}]
    }
    assert to_json_schema(all_of_field) == expected

    # Test Reference field
    definitions = Definitions({"Test": String()})
    ref_field = Reference(to="Test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(ref_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()})
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}}
    }
    assert to_json_schema(schema_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(if_clause=String(), then_clause=Integer())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected

    # Test allow_null fields
    string_null_field = String(allow_null=True)
    expected = {
        "type": ["string", "null"]
    }
    assert to_json_schema(string_null_field) == expected

    # Test default values
    string_default_field = String(default="default")
    expected = {
        "type": "string",
        "default": "default"
    }
    assert to_json_schema(string_default_field) == expected


# LLM-generated content at query #14
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test type constraint
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": ["string", "null"]}), Union)

    # Test enum constraint
    enum_field = from_json_schema({"enum": ["a", "b", "c"]})
    assert isinstance(enum_field, Choice)
    assert enum_field.choices == ["a", "b", "c"]

    # Test const constraint
    const_field = from_json_schema({"const": "fixed_value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "fixed_value"

    # Test allOf constraint
    all_of_field = from_json_schema({
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    })
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.schemas) == 2

    # Test anyOf constraint
    any_of_field = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    })
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.schemas) == 2

    # Test oneOf constraint
    one_of_field = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    })
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.schemas) == 2

    # Test not constraint
    not_field = from_json_schema({
        "not": {"type": "string"}
    })
    assert isinstance(not_field, Not)
    assert isinstance(not_field.schema, String)

    # Test if-then-else constraint
    if_then_else_field = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"}
    })
    assert isinstance(if_then_else_field, IfThenElse)
    assert isinstance(if_then_else_field.if_schema, String)
    assert isinstance(if_then_else_field.then_schema, String)
    assert isinstance(if_then_else_field.else_schema, Number)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Person"] = from_json_schema({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        }
    })
    ref_field = from_json_schema({"$ref": "#/components/schemas/Person"}, definitions=definitions)
    assert isinstance(ref_field, Reference)
    assert ref_field.ref == "#/components/schemas/Person"

    # Test combined constraints
    combined_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    })
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.schemas) == 2  # type and pattern constraints

    # Test default case (no constraints)
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #15
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected = {
        "type": "array",
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"},
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=2
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 2
    }
    assert to_json_schema(object_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="fixed_value")
    expected = {
        "const": "fixed_value"
    }
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Integer()])
    expected = {
        "allOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(all_of_field) == expected

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(one_of_field) == expected

    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_then_else_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected

    # Test Reference field
    definitions = Definitions({"TestSchema": String()})
    ref_field = Reference(to="TestSchema", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/TestSchema",
        "components": {
            "schemas": {
                "TestSchema": {"type": "string"}
            }
        }
    }
    assert to_json_schema(ref_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected


# LLM-generated content at query #16
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9]+$"
    assert field.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items == False
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "patternProperties": {
            "^S_": {"type": "string"},
            "^I_": {"type": "integer"}
        },
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert isinstance(field.pattern_properties["^S_"], String)
    assert isinstance(field.pattern_properties["^I_"], Integer)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 30}

    # Test invalid type_string
    with pytest.raises(AssertionError):
        from_json_schema_type({}, "invalid", False, Definitions())


# LLM-generated content at query #17
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "format": "email",
        "pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "default": "test@example.com"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    assert field.default == "test@example.com"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["item1", "item2"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["item1", "item2"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 5,
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 5
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #18
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "default": 50,
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 50,
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test",
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.pattern == "^[a-zA-Z0-9]+$"
    assert field.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True,
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default is True

    # Test array type
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
    assert field.unique_items is True
    assert field.default == ["test"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
        "default": {"name": "test", "age": 25},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 25}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #19
#--------------------------

```python
def test_from_json_schema_type():
    definitions = Definitions()

    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[A-Za-z]+$",
        "default": "hello"
    }
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^[A-Za-z]+$"
    assert field.default == "hello"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["hello"]
    }
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["hello"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}


# LLM-generated content at query #20
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5)
    expected = {
        "type": "array",
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"}
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    obj_field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    assert to_json_schema(obj_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="fixed_value")
    expected = {
        "const": "fixed_value"
    }
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(one_of_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Integer()])
    expected = {
        "allOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(all_of_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected

    # Test Reference field
    definitions = Definitions({"Person": Object(properties={"name": String()})})
    ref_field = Reference(to="Person", definitions=definitions)
    expected = {
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
    assert to_json_schema(ref_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    any_field = Any()
    assert to_json_schema(any_field) == True

    # Test NeverMatch field
    never_match_field = NeverMatch()
    assert to_json_schema(never_match_field) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    integer_field = Integer(minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(integer_field) == expected

    # Test Float field
    float_field = Float(minimum=0.0, maximum=1.0, multiple_of=0.1)
    expected = {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "multipleOf": 0.1
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    boolean_field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(boolean_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=10)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(properties={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(object_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="value")
    expected = {
        "const": "value"
    }
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="value")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "value"}]
    }
    assert to_json_schema(all_of_field) == expected

    # Test IfThenElse field
    if_then_else_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_then_else_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected

    # Test Reference field
    definitions = Definitions()
    reference_field = Reference(to="test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/test",
        "components": {
            "schemas": {}
        }
    }
    assert to_json_schema(reference_field) == expected


# LLM-generated content at query #22
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "default": 50,
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 50,
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "default": "hello",
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.default == "hello"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True,
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "default": ["hello"],
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.default == ["hello"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
        "default": {"name": "John", "age": 30},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #23
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test type constraints
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)
    assert isinstance(from_json_schema({"type": ["string", "number"]}), Union)

    # Test enum constraint
    field = from_json_schema({"enum": ["a", "b", "c"]})
    assert isinstance(field, Choice)
    assert field.choices == ["a", "b", "c"]

    # Test const constraint
    field = from_json_schema({"const": "value"})
    assert isinstance(field, Const)
    assert field.value == "value"

    # Test allOf constraint
    field = from_json_schema({"allOf": [{"type": "string"}, {"minLength": 5}]})
    assert isinstance(field, AllOf)
    assert len(field.constraints) == 2

    # Test anyOf constraint
    field = from_json_schema({"anyOf": [{"type": "string"}, {"type": "number"}]})
    assert isinstance(field, Union)
    assert len(field.options) == 2

    # Test oneOf constraint
    field = from_json_schema({"oneOf": [{"type": "string"}, {"type": "number"}]})
    assert isinstance(field, OneOf)
    assert len(field.options) == 2

    # Test not constraint
    field = from_json_schema({"not": {"type": "string"}})
    assert isinstance(field, Not)
    assert isinstance(field.constraint, String)

    # Test if-then-else constraint
    field = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"}
    })
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_constraint, String)
    assert isinstance(field.then_constraint, String)
    assert isinstance(field.else_constraint, Number)

    # Test reference constraint
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    field = from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.ref == "#/components/schemas/Test"

    # Test combined constraints
    field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    })
    assert isinstance(field, AllOf)
    assert len(field.constraints) == 4

    # Test no constraints
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #24
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test type constraints
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)

    # Test enum
    enum_field = from_json_schema({"enum": ["a", "b", "c"]})
    assert isinstance(enum_field, Choice)
    assert enum_field.choices == ["a", "b", "c"]

    # Test const
    const_field = from_json_schema({"const": "fixed_value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "fixed_value"

    # Test allOf
    all_of_field = from_json_schema({
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    })
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.schemas) == 2

    # Test anyOf
    any_of_field = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(any_of_field, Union)
    assert len(any_of_field.schemas) == 2

    # Test oneOf
    one_of_field = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.schemas) == 2

    # Test not
    not_field = from_json_schema({
        "not": {"type": "string"}
    })
    assert isinstance(not_field, Not)
    assert isinstance(not_field.schema, String)

    # Test if-then-else
    if_then_else_field = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    })
    assert isinstance(if_then_else_field, IfThenElse)
    assert isinstance(if_then_else_field.if_schema, String)
    assert isinstance(if_then_else_field.then_schema, String)
    assert isinstance(if_then_else_field.else_schema, Integer)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Person"] = {"type": "object", "properties": {"name": {"type": "string"}}}
    ref_field = from_json_schema({"$ref": "#/components/schemas/Person"}, definitions=definitions)
    assert isinstance(ref_field, Reference)
    assert ref_field.reference == "#/components/schemas/Person"

    # Test combined constraints
    combined_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    })
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.schemas) == 2  # type and pattern constraints


# LLM-generated content at query #25
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test simple type schema
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)

    # Test enum schema
    enum_schema = {"enum": ["a", "b", "c"]}
    enum_field = from_json_schema(enum_schema)
    assert isinstance(enum_field, Choice)
    assert enum_field.choices == ["a", "b", "c"]

    # Test const schema
    const_schema = {"const": "fixed_value"}
    const_field = from_json_schema(const_schema)
    assert isinstance(const_field, Const)
    assert const_field.value == "fixed_value"

    # Test allOf schema
    all_of_schema = {
        "allOf": [
            {"type": "string"},
            {"minLength": 5}
        ]
    }
    all_of_field = from_json_schema(all_of_schema)
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.schemas) == 2

    # Test anyOf schema
    any_of_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    any_of_field = from_json_schema(any_of_schema)
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.schemas) == 2

    # Test oneOf schema
    one_of_schema = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    one_of_field = from_json_schema(one_of_schema)
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.schemas) == 2

    # Test not schema
    not_schema = {"not": {"type": "string"}}
    not_field = from_json_schema(not_schema)
    assert isinstance(not_field, Not)

    # Test if/then/else schema
    if_then_else_schema = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"}
    }
    if_then_else_field = from_json_schema(if_then_else_schema)
    assert isinstance(if_then_else_field, IfThenElse)

    # Test reference schema
    definitions = Definitions()
    definitions["#/components/schemas/Person"] = {"type": "object"}
    ref_schema = {"$ref": "#/components/schemas/Person"}
    ref_field = from_json_schema(ref_schema, definitions=definitions)
    assert isinstance(ref_field, Reference)

    # Test combined constraints
    combined_schema = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10
    }
    combined_field = from_json_schema(combined_schema)
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.schemas) == 2


# LLM-generated content at query #26
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test type constraints
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)

    # Test multiple types
    schema = from_json_schema({"type": ["string", "number"]})
    assert isinstance(schema, OneOf)
    assert len(schema.one_of) == 2

    # Test enum
    schema = from_json_schema({"enum": ["a", "b", "c"]})
    assert isinstance(schema, Choice)
    assert schema.choices == ["a", "b", "c"]

    # Test const
    schema = from_json_schema({"const": "value"})
    assert isinstance(schema, Const)
    assert schema.value == "value"

    # Test allOf
    schema = from_json_schema({"allOf": [{"type": "string"}, {"minLength": 5}]})
    assert isinstance(schema, AllOf)
    assert len(schema.all_of) == 2

    # Test anyOf
    schema = from_json_schema({"anyOf": [{"type": "string"}, {"type": "number"}]})
    assert isinstance(schema, OneOf)
    assert len(schema.one_of) == 2

    # Test oneOf
    schema = from_json_schema({"oneOf": [{"type": "string"}, {"type": "number"}]})
    assert isinstance(schema, OneOf)
    assert len(schema.one_of) == 2

    # Test not
    schema = from_json_schema({"not": {"type": "string"}})
    assert isinstance(schema, Not)
    assert isinstance(schema.not_, String)

    # Test if-then-else
    schema = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"}
    })
    assert isinstance(schema, IfThenElse)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    schema = from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=definitions)
    assert isinstance(schema, Reference)

    # Test combined constraints
    schema = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    })
    assert isinstance(schema, AllOf)
    assert len(schema.all_of) == 4

    # Test with components
    data = {
        "components": {
            "schemas": {
                "Name": {"type": "string"},
                "Age": {"type": "integer"}
            }
        },
        "type": "object",
        "properties": {
            "name": {"$ref": "#/components/schemas/Name"},
            "age": {"$ref": "#/components/schemas/Age"}
        }
    }
    schema = from_json_schema(data)
    assert isinstance(schema, Object)
    assert len(schema.properties) == 2


# LLM-generated content at query #27
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test simple type schemas
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)

    # Test enum schema
    enum_field = from_json_schema({"enum": ["a", "b", "c"]})
    assert isinstance(enum_field, Choice)
    assert enum_field.choices == ["a", "b", "c"]

    # Test const schema
    const_field = from_json_schema({"const": "value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "value"

    # Test allOf schema
    all_of_field = from_json_schema({
        "allOf": [
            {"type": "string"},
            {"minLength": 5}
        ]
    })
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.constraints) == 2

    # Test anyOf schema
    any_of_field = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.options) == 2

    # Test oneOf schema
    one_of_field = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.options) == 2

    # Test not schema
    not_field = from_json_schema({
        "not": {"type": "string"}
    })
    assert isinstance(not_field, Not)
    assert isinstance(not_field.schema, String)

    # Test if-then-else schema
    if_then_else_field = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    })
    assert isinstance(if_then_else_field, IfThenElse)
    assert isinstance(if_then_else_field.if_schema, String)
    assert isinstance(if_then_else_field.then_schema, String)
    assert isinstance(if_then_else_field.else_schema, Integer)

    # Test reference schema
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "string"})
    ref_field = from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=definitions)
    assert isinstance(ref_field, Reference)
    assert ref_field.ref == "#/components/schemas/Test"

    # Test combined constraints
    combined_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    })
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.constraints) == 4


# LLM-generated content at query #28
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(minimum=0.0, maximum=1.0, multiple_of=0.1)
    expected = {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "multipleOf": 0.1
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean()
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=10, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 10
    }
    assert to_json_schema(object_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {"enum": ["a", "b"]}
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="fixed_value")
    expected = {"const": "fixed_value"}
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(union_field) == expected

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(one_of_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Integer()])
    expected = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(all_of_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected

    # Test Reference field
    definitions = Definitions({"Test": String()})
    ref_field = Reference(to="Test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {"schemas": {"Test": {"type": "string"}}}
    }
    assert to_json_schema(ref_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test allow_null in fields
    string_field_nullable = String(allow_null=True)
    expected = {"type": ["string", "null"]}
    assert to_json_schema(string_field_nullable) == expected


# LLM-generated content at query #29
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum is True
    assert field.exclusive_maximum is True
    assert field.multiple_of == 2
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum is True
    assert field.exclusive_maximum is True
    assert field.multiple_of == 2
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9]+$"
    assert field.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default is True

    # Test array type
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
    assert field.unique_items is True
    assert field.default == ["test"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 10,
        "default": {"name": "test", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.default == {"name": "test", "age": 30}


# LLM-generated content at query #30
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test simple type constraints
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)

    # Test enum constraint
    result = from_json_schema({"enum": [1, 2, 3]})
    assert isinstance(result, Choice)
    assert result.choices == [1, 2, 3]

    # Test const constraint
    result = from_json_schema({"const": "test"})
    assert isinstance(result, Const)
    assert result.value == "test"

    # Test allOf constraint
    result = from_json_schema({"allOf": [{"type": "string"}, {"minLength": 1}]})
    assert isinstance(result, AllOf)
    assert len(result.schemas) == 2

    # Test anyOf constraint
    result = from_json_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(result, OneOf)
    assert len(result.schemas) == 2

    # Test oneOf constraint
    result = from_json_schema({"oneOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(result, OneOf)
    assert len(result.schemas) == 2

    # Test not constraint
    result = from_json_schema({"not": {"type": "string"}})
    assert isinstance(result, Not)
    assert isinstance(result.schema, String)

    # Test if-then-else constraint
    result = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 1},
        "else": {"type": "integer"}
    })
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_schema, String)
    assert isinstance(result.then_schema, String)
    assert isinstance(result.else_schema, Integer)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    result = from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=definitions)
    assert isinstance(result, Reference)
    assert result.reference == "#/components/schemas/Test"

    # Test combined constraints
    result = from_json_schema({
        "type": "string",
        "minLength": 1,
        "maxLength": 10
    })
    assert isinstance(result, AllOf)
    assert len(result.schemas) == 3

    # Test no constraints
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #31
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test simple type
    assert isinstance(from_json_schema({"type": "string"}), String)
    assert isinstance(from_json_schema({"type": "integer"}), Integer)
    assert isinstance(from_json_schema({"type": "number"}), Number)
    assert isinstance(from_json_schema({"type": "boolean"}), Boolean)
    assert isinstance(from_json_schema({"type": "array"}), Array)
    assert isinstance(from_json_schema({"type": "object"}), Object)

    # Test enum
    enum_field = from_json_schema({"enum": ["a", "b", "c"]})
    assert isinstance(enum_field, Choice)
    assert enum_field.choices == ["a", "b", "c"]

    # Test const
    const_field = from_json_schema({"const": "fixed_value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "fixed_value"

    # Test allOf
    all_of_field = from_json_schema({
        "allOf": [
            {"type": "string"},
            {"minLength": 5}
        ]
    })
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.schemas) == 2

    # Test anyOf
    any_of_field = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.schemas) == 2

    # Test oneOf
    one_of_field = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.schemas) == 2

    # Test not
    not_field = from_json_schema({
        "not": {"type": "string"}
    })
    assert isinstance(not_field, Not)
    assert isinstance(not_field.schema, String)

    # Test if-then-else
    if_field = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    })
    assert isinstance(if_field, IfThenElse)
    assert isinstance(if_field.if_schema, String)
    assert isinstance(if_field.then_schema, String)
    assert isinstance(if_field.else_schema, Integer)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    ref_field = from_json_schema(
        {"$ref": "#/components/schemas/Test"},
        definitions=definitions
    )
    assert isinstance(ref_field, Reference)
    assert ref_field.reference == "#/components/schemas/Test"

    # Test combined constraints
    combined_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10
    })
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.schemas) == 2

    # Test empty schema
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #32
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    assert to_json_schema(field) == True

    # Test NeverMatch field
    field = NeverMatch()
    assert to_json_schema(field) == False

    # Test String field
    field = String(allow_null=True, min_length=1, max_length=10, pattern="[a-z]+", format="email")
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "[a-z]+"
    assert result["format"] == "email"

    # Test Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=True, multiple_of=2)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == True
    assert result["multipleOf"] == 2

    # Test Float field
    field = Float(allow_null=True, minimum=0.0, maximum=1.0)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test Boolean field
    field = Boolean(allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "boolean"

    # Test Array field
    field = Array(allow_null=True, min_items=1, max_items=10, items=String(), additional_items=False)
    result = to_json_schema(field)
    assert result["type"] == ["array", "null"]
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["items"] == {"type": "string"}
    assert result["additionalItems"] == False

    # Test Object field
    field = Object(
        allow_null=False,
        properties={"name": String()},
        additional_properties=False,
        required=["name"]
    )
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"] == {"name": {"type": "string"}}
    assert result["additionalProperties"] == False
    assert result["required"] == ["name"]

    # Test Choice field
    field = Choice(choices=[("a", "a"), ("b", "b")])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result["anyOf"] == [{"type": "string"}, {"type": "integer"}]

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert result["oneOf"] == [{"type": "string"}, {"type": "integer"}]

    # Test AllOf field
    field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(field)
    assert result["allOf"] == [{"type": "string"}, {"const": "test"}]

    # Test IfThenElse field
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(field)
    assert result["if"] == {"type": "string"}
    assert result["then"] == {"type": "integer"}
    assert result["else"] == {"type": "boolean"}

    # Test Not field
    field = Not(negated=String())
    result = to_json_schema(field)
    assert result["not"] == {"type": "string"}

    # Test Reference field
    definitions = Definitions({"Test": String()})
    field = Reference(to="Test", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/Test"
    assert result["components"]["schemas"]["Test"] == {"type": "string"}

    # Test Schema field
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["properties"] == {"name": {"type": "string"}}
    assert result["required"] == ["name"]


# LLM-generated content at query #33
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 2
    assert field.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9]+$"
    assert field.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items == False
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "patternProperties": {
            "^S_": {"type": "string"},
            "^I_": {"type": "integer"}
        },
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert "age" in field.properties
    assert isinstance(field.properties["age"], Integer)
    assert "^S_" in field.pattern_properties
    assert isinstance(field.pattern_properties["^S_"], String)
    assert "^I_" in field.pattern_properties
    assert isinstance(field.pattern_properties["^I_"], Integer)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 30}


# LLM-generated content at query #34
#--------------------------

```python
def test_to_json_schema():
    # Test Any
    assert to_json_schema(Any()) == True

    # Test NeverMatch
    assert to_json_schema(NeverMatch()) == False

    # Test String
    field = String(allow_null=True, min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(field) == expected

    # Test Integer
    field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=2)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 2
    }
    assert to_json_schema(field) == expected

    # Test Float
    field = Float(allow_null=True, exclusive_minimum=0, exclusive_maximum=1)
    expected = {
        "type": ["number", "null"],
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 1
    }
    assert to_json_schema(field) == expected

    # Test Boolean
    field = Boolean(allow_null=False)
    expected = {"type": "boolean"}
    assert to_json_schema(field) == expected

    # Test Array
    field = Array(
        allow_null=True,
        items=String(),
        additional_items=False,
        min_items=1,
        max_items=5,
        unique_items=True
    )
    expected = {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(field) == expected

    # Test Object
    field = Object(
        allow_null=False,
        properties={"name": String()},
        additional_properties=False,
        required=["name"]
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "additionalProperties": False,
        "required": ["name"]
    }
    assert to_json_schema(field) == expected

    # Test Choice
    field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {"enum": ["a", "b"]}
    assert to_json_schema(field) == expected

    # Test Const
    field = Const(const="value")
    expected = {"const": "value"}
    assert to_json_schema(field) == expected

    # Test Union
    field = Union(any_of=[String(), Integer()])
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(field) == expected

    # Test AllOf
    field = AllOf(all_of=[String(), Const(const="test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(field) == expected

    # Test Reference
    definitions = Definitions()
    target_field = String()
    definitions["test"] = target_field
    field = Reference(to="test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/test",
        "components": {"schemas": {"test": {"type": "string"}}}
    }
    assert to_json_schema(field) == expected

    # Test IfThenElse
    field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(field) == expected

    # Test Not
    field = Not(negated=String())
    expected = {"not": {"type": "string"}}
    assert to_json_schema(field) == expected


# LLM-generated content at query #35
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {"type": "number", "minimum": 0, "maximum": 100}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100

    # Test integer type
    data = {"type": "integer", "minimum": 0, "maximum": 100}
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100

    # Test string type
    data = {"type": "string", "minLength": 1, "maxLength": 100}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 100

    # Test boolean type
    data = {"type": "boolean"}
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)

    # Test array type
    data = {"type": "array", "items": {"type": "string"}}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)

    # Test object type
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)

    # Test invalid type_string
    with pytest.raises(AssertionError):
        from_json_schema_type({"type": "invalid"}, "invalid", False, Definitions())


