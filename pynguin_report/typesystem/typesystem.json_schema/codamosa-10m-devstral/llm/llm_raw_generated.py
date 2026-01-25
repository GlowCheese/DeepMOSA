####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert all(isinstance(field, String) for field in result.all_of)
    assert result.default == "test"

    # Test with nested allOf
    nested_data = {
        "allOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"allOf": [{"type": "object", "properties": {"age": {"type": "integer"}}}]}
        ]
    }
    nested_result = all_of_from_json_schema(nested_data, definitions)
    assert isinstance(nested_result, AllOf)
    assert len(nested_result.all_of) == 2
    assert isinstance(nested_result.all_of[0], Object)
    assert isinstance(nested_result.all_of[1], AllOf)

    # Test with empty allOf
    empty_data = {"allOf": []}
    empty_result = all_of_from_json_schema(empty_data, definitions)
    assert isinstance(empty_result, AllOf)
    assert len(empty_result.all_of) == 0


# LLM-generated content at query #2
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with if, then, and else clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "number"}
    }
    definitions = Definitions()
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, String)
    assert isinstance(result.else_clause, Number)

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, String)
    assert result.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "number"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert result.then_clause is None
    assert isinstance(result.else_clause, Number)

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
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "number"},
        "default": "default_value"
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert result.default == "default_value"


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

    # Test nested fields with definitions
    nested_field = Object(properties={"nested": Reference(to="Nested", definitions=Definitions({"Nested": Integer()}))})
    expected = {
        "type": "object",
        "properties": {
            "nested": {
                "$ref": "#/components/schemas/Nested"
            }
        },
        "components": {
            "schemas": {
                "Nested": {"type": "integer"}
            }
        }
    }
    assert to_json_schema(nested_field) == expected


# LLM-generated content at query #4
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
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10

    # Test boolean type
    data = {"type": "boolean"}
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)

    # Test array type
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5

    # Test object type
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert "name" in field.required

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #5
#--------------------------

```python
def test_all_of_from_json_schema():
    # Test with simple allOf
    data = {
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    }
    result = all_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert all(isinstance(field, String) for field in result.all_of)

    # Test with nested allOf
    data = {
        "allOf": [
            {
                "allOf": [
                    {"type": "integer", "minimum": 0},
                    {"type": "integer", "maximum": 100}
                ]
            },
            {"type": "integer", "multipleOf": 2}
        ]
    }
    result = all_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], AllOf)
    assert isinstance(result.all_of[1], Integer)

    # Test with default value
    data = {
        "allOf": [{"type": "boolean"}],
        "default": True
    }
    result = all_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, AllOf)
    assert result.default == True

    # Test with reference in allOf
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String(min_length=1)
    data = {
        "allOf": [
            {"$ref": "#/components/schemas/Test"},
            {"type": "string", "maxLength": 10}
        ]
    }
    result = all_of_from_json_schema(data, definitions=definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], Reference)
    assert isinstance(result.all_of[1], String)


# LLM-generated content at query #6
#--------------------------

```python
def test_ref_from_json_schema():
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()

    data = {"$ref": "#/components/schemas/Test"}
    field = ref_from_json_schema(data, definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/Test"
    assert field.definitions is definitions

    with pytest.raises(AssertionError):
        ref_from_json_schema({"$ref": "unsupported_ref"}, definitions)


# LLM-generated content at query #7
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
    result = one_of_from_json_schema(schema, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Number)

    # Test with nested oneOf schema
    schema = {
        "oneOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "array", "items": {"type": "integer"}}
        ],
        "default": {"name": "default"}
    }
    result = one_of_from_json_schema(schema, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Object)
    assert isinstance(result.one_of[1], Array)
    assert result.default == {"name": "default"}

    # Test with reference in oneOf
    definitions["#/components/schemas/Person"] = Object(properties={"name": String()})
    schema = {
        "oneOf": [
            {"$ref": "#/components/schemas/Person"},
            {"type": "string"}
        ]
    }
    result = one_of_from_json_schema(schema, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Reference)
    assert isinstance(result.one_of[1], String)


# LLM-generated content at query #8
#--------------------------

```python
def test_one_of_from_json_schema():
    definitions = Definitions()

    # Test with simple types
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Float)

    # Test with nested objects
    data = {
        "oneOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "object", "properties": {"age": {"type": "number"}}}
        ]
    }
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Object)
    assert isinstance(result.one_of[1], Object)

    # Test with default value
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "default_value"
    }
    result = one_of_from_json_schema(data, definitions)
    assert result.default == "default_value"

    # Test with reference
    definitions["#/components/schemas/Test"] = String()
    data = {
        "oneOf": [
            {"$ref": "#/components/schemas/Test"},
            {"type": "number"}
        ]
    }
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert isinstance(result.one_of[0], Reference)
    assert result.one_of[0].to == "#/components/schemas/Test"


# LLM-generated content at query #9
#--------------------------

```python
def test_one_of_from_json_schema():
    # Test with simple oneOf schema
    schema = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result = one_of_from_json_schema(schema, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Number)

    # Test with oneOf schema containing references
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "string"})
    schema_with_ref = {
        "oneOf": [
            {"$ref": "#/components/schemas/Test"},
            {"type": "number"}
        ]
    }
    result_with_ref = one_of_from_json_schema(schema_with_ref, definitions=definitions)
    assert isinstance(result_with_ref, OneOf)
    assert len(result_with_ref.one_of) == 2
    assert isinstance(result_with_ref.one_of[0], Reference)
    assert isinstance(result_with_ref.one_of[1], Number)

    # Test with oneOf schema containing default value
    schema_with_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test"
    }
    result_with_default = one_of_from_json_schema(schema_with_default, definitions=Definitions())
    assert isinstance(result_with_default, OneOf)
    assert result_with_default.default == "test"

    # Test with empty oneOf schema
    empty_schema = {"oneOf": []}
    result_empty = one_of_from_json_schema(empty_schema, definitions=Definitions())
    assert isinstance(result_empty, OneOf)
    assert len(result_empty.one_of) == 0


# LLM-generated content at query #10
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
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100
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
    obj_field = Object(
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
    ref_field = Reference(to="Test", target=String(), definitions={})
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {"schemas": {"Test": {"type": "string"}}}
    }
    assert to_json_schema(ref_field) == expected


# LLM-generated content at query #11
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type
    data = {"type": "string"}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, String)

    # Test multiple types (union)
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

    # Test nullable type
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test empty type with allow_null
    data = {"type": []}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, NeverMatch)

    # Test empty type with allow_null=True
    data = {"type": [], "allow_null": True}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test complex nested type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        }
    }
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert "age" in result.properties


# LLM-generated content at query #12
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
    data = {"type": "string", "minLength": 1, "maxLength": 10}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10

    # Test boolean type
    data = {"type": "boolean"}
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)

    # Test array type
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 5
    assert isinstance(field.items, String)

    # Test object type
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert "name" in field.required

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #13
#--------------------------

```python
def test_enum_from_json_schema():
    # Test with a simple enum
    data = {"enum": ["a", "b", "c"]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b"), ("c", "c")]
    assert field.default == NO_DEFAULT

    # Test with a default value
    data = {"enum": ["a", "b", "c"], "default": "b"}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b"), ("c", "c")]
    assert field.default == "b"

    # Test with different types in enum
    data = {"enum": [1, 2, 3]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [(1, 1), (2, 2), (3, 3)]
    assert field.default == NO_DEFAULT

    # Test with mixed types in enum
    data = {"enum": [1, "a", True]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [(1, 1), ("a", "a"), (True, True)]
    assert field.default == NO_DEFAULT


# LLM-generated content at query #14
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schema
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test reference schema
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = {"type": "string"}
    assert isinstance(from_json_schema({"$ref": "#/components/schemas/Test"}, definitions), String)

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

    # Test combined constraints
    combined_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    })
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.schemas) == 4

    # Test no constraints
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #15
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
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    }
    all_of_field = from_json_schema(all_of_schema)
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.schemas) == 2

    # Test anyOf
    any_of_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    any_of_field = from_json_schema(any_of_schema)
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.schemas) == 2

    # Test oneOf
    one_of_schema = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
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
        "else": {"type": "integer"}
    }
    if_then_else_field = from_json_schema(if_then_else_schema)
    assert isinstance(if_then_else_field, IfThenElse)
    assert isinstance(if_then_else_field.if_schema, String)
    assert isinstance(if_then_else_field.then_schema, String)
    assert isinstance(if_then_else_field.else_schema, Integer)

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
        "pattern": "^[a-z]+$"
    }
    combined_field = from_json_schema(combined_schema)
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.schemas) == 2  # type and pattern

    # Test no constraints (should return Any)
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #16
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

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
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
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected

    # Test nullable fields
    nullable_string = String(allow_null=True)
    expected = {"type": ["string", "null"]}
    assert to_json_schema(nullable_string) == expected


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
        "maxProperties": 2,
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #18
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
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=True, multiple_of=2)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "multipleOf": 2
    }
    assert to_json_schema(integer_field) == expected

    # Test Boolean field
    boolean_field = Boolean(allow_null=True)
    expected = {
        "type": ["boolean", "null"]
    }
    assert to_json_schema(boolean_field) == expected

    # Test Array field
    array_field = Array(allow_null=False, min_items=1, max_items=5, items=String(), additional_items=False)
    expected = {
        "type": "array",
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"},
        "additionalItems": False,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
        allow_null=True,
        properties={"name": String()},
        additional_properties=False,
        min_properties=1,
        max_properties=5
    )
    expected = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}},
        "additionalProperties": False,
        "minProperties": 1,
        "maxProperties": 5
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


# LLM-generated content at query #19
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type
    data = {"type": "string"}
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, String)

    # Test multiple types (union)
    data = {"type": ["string", "number"]}
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2
    assert isinstance(field.any_of[0], String)
    assert isinstance(field.any_of[1], Number)

    # Test nullable type
    data = {"type": "string", "nullable": True}
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, String)
    assert field.allow_null is True

    # Test no type with nullable
    data = {"nullable": True}
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Const)
    assert field.value is None

    # Test no type without nullable
    data = {}
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, NeverMatch)

    # Test object type with properties
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        }
    }
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Number)

    # Test array type with items
    data = {
        "type": "array",
        "items": {"type": "string"}
    }
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)

    # Test boolean type
    data = {"type": "boolean"}
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Boolean)

    # Test integer type
    data = {"type": "integer"}
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Integer)

    # Test number type
    data = {"type": "number"}
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Number)

    # Test with constraints
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[A-Za-z]+$"
    }
    field = type_from_json_schema(data, definitions=definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.pattern == "^[A-Za-z]+$"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    int_field = Integer(allow_null=False, minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
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
    array_field = Array(allow_null=True, min_items=1, max_items=5, items=String())
    expected = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"}
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
        allow_null=False,
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

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    expected = {"allOf": [{"type": "string", "minLength": 1}, {"type": "string", "maxLength": 10}]}
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
    if_field = IfThenElse(if_clause=String(), then_clause=Integer())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    assert to_json_schema(if_field) == expected

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
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test Definitions
    definitions = Definitions({
        "Person": Object(properties={"name": String()}),
        "Address": Object(properties={"street": String()})
    })
    expected = {
        "components": {
            "schemas": {
                "Person": {"type": "object", "properties": {"name": {"type": "string"}}},
                "Address": {"type": "object", "properties": {"street": {"type": "string"}}}
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #2
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
        "propertyNames": {"pattern": "^[a-zA-Z0-9]+$"},
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


# LLM-generated content at query #3
#--------------------------

```python
def test_enum_from_json_schema():
    definitions = Definitions()
    data = {"enum": ["a", "b", "c"]}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b"), ("c", "c")]
    assert field.default == NO_DEFAULT

    data_with_default = {"enum": [1, 2, 3], "default": 2}
    field_with_default = enum_from_json_schema(data_with_default, definitions)
    assert field_with_default.choices == [(1, 1), (2, 2), (3, 3)]
    assert field_with_default.default == 2


# LLM-generated content at query #4
#--------------------------

```python
def test_enum_from_json_schema():
    # Test with a simple enum
    data = {"enum": ["a", "b", "c"]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b"), ("c", "c")]
    assert field.default == NO_DEFAULT

    # Test with a default value
    data = {"enum": ["x", "y", "z"], "default": "y"}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [("x", "x"), ("y", "y"), ("z", "z")]
    assert field.default == "y"

    # Test with numeric enum
    data = {"enum": [1, 2, 3]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [(1, 1), (2, 2), (3, 3)]
    assert field.default == NO_DEFAULT

    # Test with mixed type enum
    data = {"enum": [True, False, None]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [(True, True), (False, False), (None, None)]
    assert field.default == NO_DEFAULT


# LLM-generated content at query #5
#--------------------------

```python
def test_if_then_else_from_json_schema():
    definitions = Definitions()

    # Test with if, then, and else clauses
    schema = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "number"}
    }
    field = if_then_else_from_json_schema(schema, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, String)
    assert isinstance(field.else_clause, Number)

    # Test with only if and then clauses
    schema = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5}
    }
    field = if_then_else_from_json_schema(schema, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, String)
    assert field.else_clause is None

    # Test with only if and else clauses
    schema = {
        "if": {"type": "string"},
        "else": {"type": "number"}
    }
    field = if_then_else_from_json_schema(schema, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert field.then_clause is None
    assert isinstance(field.else_clause, Number)

    # Test with only if clause
    schema = {
        "if": {"type": "string"}
    }
    field = if_then_else_from_json_schema(schema, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert field.then_clause is None
    assert field.else_clause is None

    # Test with default value
    schema = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "number"},
        "default": "default_value"
    }
    field = if_then_else_from_json_schema(schema, definitions)
    assert field.default == "default_value"


# LLM-generated content at query #6
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
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #7
#--------------------------

```python
def test_one_of_from_json_schema():
    # Test basic oneOf schema
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    field = one_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2
    assert isinstance(field.one_of[0], String)
    assert isinstance(field.one_of[1], Number)

    # Test oneOf with default value
    data_with_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "default_value"
    }
    field_with_default = one_of_from_json_schema(data_with_default, definitions=Definitions())
    assert field_with_default.default == "default_value"

    # Test oneOf with complex schemas
    data_complex = {
        "oneOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "array", "items": {"type": "integer"}}
        ]
    }
    field_complex = one_of_from_json_schema(data_complex, definitions=Definitions())
    assert isinstance(field_complex, OneOf)
    assert len(field_complex.one_of) == 2
    assert isinstance(field_complex.one_of[0], Object)
    assert isinstance(field_complex.one_of[1], Array)


# LLM-generated content at query #8
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
            {"type": "string", "minLength": 1},
            {"type": "number", "minimum": 0}
        ],
        "default": "default_value"
    }
    field = one_of_from_json_schema(schema, definitions)
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2
    assert field.one_of[0].min_length == 1
    assert field.one_of[1].minimum == 0
    assert field.default == "default_value"

    # Test with reference in oneOf
    definitions["#/components/schemas/Test"] = String()
    schema = {
        "oneOf": [
            {"$ref": "#/components/schemas/Test"},
            {"type": "number"}
        ]
    }
    field = one_of_from_json_schema(schema, definitions)
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2
    assert isinstance(field.one_of[0], Reference)
    assert isinstance(field.one_of[1], Number)


# LLM-generated content at query #9
#--------------------------

```python
def test_all_of_from_json_schema():
    # Test basic allOf with two simple schemas
    data = {
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    }
    result = all_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], String)
    assert isinstance(result.all_of[1], String)

    # Test allOf with nested references
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "integer"})
    data = {
        "allOf": [
            {"$ref": "#/components/schemas/Test"},
            {"type": "integer", "minimum": 0}
        ]
    }
    result = all_of_from_json_schema(data, definitions=definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], Reference)
    assert isinstance(result.all_of[1], Integer)

    # Test allOf with default value
    data = {
        "allOf": [
            {"type": "string"},
            {"type": "string", "pattern": "^[A-Z]+"}
        ],
        "default": "TEST"
    }
    result = all_of_from_json_schema(data, definitions=Definitions())
    assert result.default == "TEST"

    # Test empty allOf
    data = {"allOf": []}
    result = all_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 0


