####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_one_of_from_json_schema():
    # Test basic oneOf with multiple schemas
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result = one_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Integer)

    # Test oneOf with default value
    data_with_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test_value"
    }
    result_with_default = one_of_from_json_schema(data_with_default, definitions=Definitions())
    assert isinstance(result_with_default, OneOf)
    assert result_with_default.default == "test_value"

    # Test oneOf with single schema
    data_single = {
        "oneOf": [
            {"type": "boolean"}
        ]
    }
    result_single = one_of_from_json_schema(data_single, definitions=Definitions())
    assert isinstance(result_single, OneOf)
    assert len(result_single.one_of) == 1
    assert isinstance(result_single.one_of[0], Boolean)

    # Test oneOf with complex nested schemas
    data_complex = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            {
                "type": "array",
                "items": {"type": "integer"}
            }
        ]
    }
    result_complex = one_of_from_json_schema(data_complex, definitions=Definitions())
    assert isinstance(result_complex, OneOf)
    assert len(result_complex.one_of) == 2
    assert isinstance(result_complex.one_of[0], Object)
    assert isinstance(result_complex.one_of[1], Array)

    # Test oneOf without default (should use NO_DEFAULT)
    data_no_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result_no_default = one_of_from_json_schema(data_no_default, definitions=Definitions())
    assert isinstance(result_no_default, OneOf)
    assert result_no_default.default is NO_DEFAULT

    # Test oneOf with definitions reference
    defs = Definitions()
    data_with_ref = {
        "oneOf": [
            {"$ref": "#/definitions/StringType"},
            {"type": "integer"}
        ]
    }
    defs["#/definitions/StringType"] = String()
    result_with_ref = one_of_from_json_schema(data_with_ref, definitions=defs)
    assert isinstance(result_with_ref, OneOf)
    assert len(result_with_ref.one_of) == 2
    assert isinstance(result_with_ref.one_of[0], Reference)
    assert isinstance(result_with_ref.one_of[1], Integer)


# LLM-generated content at query #2
#--------------------------

```python
def test_one_of_from_json_schema():
    # Test basic oneOf with simple types
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result = one_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Integer)

    # Test oneOf with default value
    data_with_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test"
    }
    result = one_of_from_json_schema(data_with_default, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert result.default == "test"

    # Test oneOf with complex schemas
    data_complex = {
        "oneOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "array", "items": {"type": "integer"}}
        ]
    }
    result = one_of_from_json_schema(data_complex, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Object)
    assert isinstance(result.one_of[1], Array)

    # Test oneOf with single schema
    data_single = {
        "oneOf": [
            {"type": "boolean"}
        ]
    }
    result = one_of_from_json_schema(data_single, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 1
    assert isinstance(result.one_of[0], Boolean)

    # Test oneOf with no default (should use NO_DEFAULT)
    data_no_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result = one_of_from_json_schema(data_no_default, definitions=Definitions())
    assert result.default is NO_DEFAULT

    # Test oneOf with nested references
    defs = Definitions()
    data_with_ref = {
        "oneOf": [
            {"$ref": "#/definitions/StringType"},
            {"type": "integer"}
        ]
    }
    defs["#/definitions/StringType"] = String()
    result = one_of_from_json_schema(data_with_ref, definitions=defs)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Reference)
    assert isinstance(result.one_of[1], Integer)


# LLM-generated content at query #3
#--------------------------

```python
def test_get_valid_types():
    # Test with single type string
    type_strings, allow_null = get_valid_types({"type": "string"})
    assert type_strings == {"string"}
    assert allow_null is False

    # Test with multiple type strings as list
    type_strings, allow_null = get_valid_types({"type": ["string", "number"]})
    assert type_strings == {"string", "number"}
    assert allow_null is False

    # Test with null type
    type_strings, allow_null = get_valid_types({"type": "null"})
    assert type_strings == set()
    assert allow_null is True

    # Test with null and other types
    type_strings, allow_null = get_valid_types({"type": ["string", "null"]})
    assert type_strings == {"string"}
    assert allow_null is True

    # Test with no type specified (should include all types)
    type_strings, allow_null = get_valid_types({})
    assert type_strings == {"boolean", "object", "array", "number", "string"}
    assert allow_null is False

    # Test with no type and null handling
    type_strings, allow_null = get_valid_types({"type": ["null"]})
    assert type_strings == set()
    assert allow_null is True

    # Test with integer and number (number should remove integer)
    type_strings, allow_null = get_valid_types({"type": ["integer", "number"]})
    assert type_strings == {"number"}
    assert allow_null is False

    # Test with integer only
    type_strings, allow_null = get_valid_types({"type": "integer"})
    assert type_strings == {"integer"}
    assert allow_null is False

    # Test with boolean type
    type_strings, allow_null = get_valid_types({"type": "boolean"})
    assert type_strings == {"boolean"}
    assert allow_null is False

    # Test with object type
    type_strings, allow_null = get_valid_types({"type": "object"})
    assert type_strings == {"object"}
    assert allow_null is False

    # Test with array type
    type_strings, allow_null = get_valid_types({"type": "array"})
    assert type_strings == {"array"}
    assert allow_null is False

    # Test with multiple types including null
    type_strings, allow_null = get_valid_types({"type": ["string", "number", "null", "object"]})
    assert type_strings == {"string", "number", "object"}
    assert allow_null is True

    # Test with integer, number, and null
    type_strings, allow_null = get_valid_types({"type": ["integer", "number", "null"]})
    assert type_strings == {"number"}
    assert allow_null is True

    # Test with all types including null
    type_strings, allow_null = get_valid_types({"type": ["string", "number", "integer", "boolean", "object", "array", "null"]})
    assert type_strings == {"string", "number", "boolean", "object", "array"}
    assert allow_null is True


# LLM-generated content at query #4
#--------------------------

```python
def test_one_of_from_json_schema():
    # Test basic oneOf with simple types
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Integer)

    # Test oneOf with default value
    data_with_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test"
    }
    result = one_of_from_json_schema(data_with_default, definitions)
    assert isinstance(result, OneOf)
    assert result.default == "test"

    # Test oneOf with complex schemas
    data_complex = {
        "oneOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "array", "items": {"type": "integer"}}
        ]
    }
    result = one_of_from_json_schema(data_complex, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Object)
    assert isinstance(result.one_of[1], Array)

    # Test oneOf with single schema
    data_single = {
        "oneOf": [
            {"type": "boolean"}
        ]
    }
    result = one_of_from_json_schema(data_single, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 1
    assert isinstance(result.one_of[0], Boolean)

    # Test oneOf with NO_DEFAULT
    data_no_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result = one_of_from_json_schema(data_no_default, definitions)
    assert isinstance(result, OneOf)
    assert result.default is NO_DEFAULT

    # Test oneOf with nested oneOf
    data_nested = {
        "oneOf": [
            {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"}
                ]
            },
            {"type": "boolean"}
        ]
    }
    result = one_of_from_json_schema(data_nested, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], OneOf)
    assert isinstance(result.one_of[1], Boolean)

    # Test oneOf with references
    test_defs = Definitions()
    test_defs["#/components/schemas/StringSchema"] = String()
    test_defs["#/components/schemas/IntSchema"] = Integer()
    
    data_with_refs = {
        "oneOf": [
            {"$ref": "#/components/schemas/StringSchema"},
            {"$ref": "#/components/schemas/IntSchema"}
        ]
    }
    result = one_of_from_json_schema(data_with_refs, test_defs)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Reference)
    assert isinstance(result.one_of[1], Reference)


# LLM-generated content at query #5
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": "test_default"
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    assert result.default == "test_default"

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None
    assert result.default == NO_DEFAULT

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None
    assert result.default == NO_DEFAULT

    # Test with only if clause
    data = {
        "if": {"type": "object"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None
    assert result.default == NO_DEFAULT

    # Test with complex nested schemas
    data = {
        "if": {"type": "array", "items": {"type": "string"}},
        "then": {"type": "object", "properties": {"name": {"type": "string"}}},
        "else": {"enum": [1, 2, 3]},
        "default": 42
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    assert result.default == 42

    # Test with boolean schemas
    data = {
        "if": True,
        "then": False,
        "else": True
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #6
#--------------------------

```python
def test_enum_from_json_schema():
    # Test basic enum with simple values
    data = {"enum": [1, 2, 3]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1, 1), (2, 2), (3, 3)]
    
    # Test enum with string values
    data = {"enum": ["red", "green", "blue"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("red", "red"), ("green", "green"), ("blue", "blue")]
    
    # Test enum with mixed types
    data = {"enum": [1, "two", 3.0, True, None]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1, 1), ("two", "two"), (3.0, 3.0), (True, True), (None, None)]
    
    # Test enum with default value
    data = {"enum": ["a", "b", "c"], "default": "b"}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]
    assert result.default == "b"
    
    # Test enum without default value
    data = {"enum": [10, 20, 30]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.default is NO_DEFAULT
    
    # Test enum with single value
    data = {"enum": ["only"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("only", "only")]


# LLM-generated content at query #7
#--------------------------

```python
def test_all_of_from_json_schema():
    # Test basic allOf with simple types
    data = {
        "allOf": [
            {"type": "string", "minLength": 1},
            {"type": "string", "maxLength": 10}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], String)
    assert isinstance(result.all_of[1], String)

    # Test allOf with default value
    data_with_default = {
        "allOf": [
            {"type": "integer", "minimum": 0},
            {"type": "integer", "maximum": 100}
        ],
        "default": 50
    }
    result = all_of_from_json_schema(data_with_default, definitions)
    assert isinstance(result, AllOf)
    assert result.default == 50

    # Test allOf with no default
    data_no_default = {
        "allOf": [
            {"type": "boolean"},
            {"const": True}
        ]
    }
    result = all_of_from_json_schema(data_no_default, definitions)
    assert isinstance(result, AllOf)
    assert result.default is NO_DEFAULT

    # Test allOf with complex nested schemas
    data_complex = {
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            {
                "type": "object",
                "required": ["name"]
            }
        ]
    }
    result = all_of_from_json_schema(data_complex, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], Object)
    assert isinstance(result.all_of[1], Object)

    # Test allOf with single item
    data_single = {
        "allOf": [
            {"type": "string"}
        ]
    }
    result = all_of_from_json_schema(data_single, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 1

    # Test allOf with array type
    data_array = {
        "allOf": [
            {"type": "array", "items": {"type": "string"}},
            {"minItems": 1}
        ]
    }
    result = all_of_from_json_schema(data_array, definitions)
    assert isinstance(result, AllOf)
    assert isinstance(result.all_of[0], Array)


# LLM-generated content at query #8
#--------------------------

```python
def test_type_from_json_schema():
    # Test with single type string
    data = {"type": "string", "minLength": 1}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    
    # Test with multiple type strings
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    
    # Test with null type allowed
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True
    
    # Test with only null type
    data = {"type": "null"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)
    
    # Test with no valid types but allow_null is True
    data = {"type": "null"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)
    assert result.const_value is None
    
    # Test with integer type
    data = {"type": "integer", "minimum": 0}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)
    
    # Test with boolean type
    data = {"type": "boolean"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Boolean)
    
    # Test with array type
    data = {"type": "array", "items": {"type": "string"}}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Array)
    
    # Test with object type
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Object)
    
    # Test with number type
    data = {"type": "number", "minimum": 0.0}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Number)
    
    # Test with multiple types including null
    data = {"type": ["string", "integer", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True
    assert len(result.any_of) == 2
    
    # Test with empty type list and allow_null False
    data = {}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, NeverMatch)


# LLM-generated content at query #9
#--------------------------

```python
def test_from_json_schema():
    # Test with boolean schema True
    result = from_json_schema(True)
    assert isinstance(result, Any)

    # Test with boolean schema False
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

    # Test with simple type constraint
    result = from_json_schema({"type": "string"})
    assert isinstance(result, String)

    # Test with integer type
    result = from_json_schema({"type": "integer"})
    assert isinstance(result, Integer)

    # Test with number type
    result = from_json_schema({"type": "number"})
    assert isinstance(result, Number)

    # Test with boolean type
    result = from_json_schema({"type": "boolean"})
    assert isinstance(result, Boolean)

    # Test with array type
    result = from_json_schema({"type": "array"})
    assert isinstance(result, Array)

    # Test with object type
    result = from_json_schema({"type": "object"})
    assert isinstance(result, Object)

    # Test with enum constraint
    result = from_json_schema({"enum": [1, 2, 3]})
    assert isinstance(result, Choice)

    # Test with const constraint
    result = from_json_schema({"const": "value"})
    assert isinstance(result, Const)

    # Test with string constraints
    result = from_json_schema({"type": "string", "minLength": 5, "maxLength": 10})
    assert isinstance(result, AllOf)

    # Test with numeric constraints
    result = from_json_schema({"type": "number", "minimum": 0, "maximum": 100})
    assert isinstance(result, AllOf)

    # Test with array constraints
    result = from_json_schema({"type": "array", "minItems": 1, "maxItems": 5})
    assert isinstance(result, AllOf)

    # Test with object and properties
    result = from_json_schema({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    })
    assert isinstance(result, AllOf)

    # Test with allOf
    result = from_json_schema({
        "allOf": [
            {"type": "string"},
            {"minLength": 5}
        ]
    })
    assert isinstance(result, AllOf)

    # Test with anyOf
    result = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(result, Union)

    # Test with oneOf
    result = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(result, OneOf)

    # Test with not
    result = from_json_schema({
        "not": {"type": "string"}
    })
    assert isinstance(result, Not)

    # Test with if-then-else
    result = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    })
    assert isinstance(result, IfThenElse)

    # Test with $ref and definitions
    defs = Definitions()
    result = from_json_schema(
        {"$ref": "#/components/schemas/User"},
        definitions=defs
    )
    assert isinstance(result, Reference)

    # Test with multiple constraints
    result = from_json_schema({
        "type": "string",
        "enum": ["a", "b", "c"],
        "minLength": 1
    })
    assert isinstance(result, AllOf)

    # Test with empty object
    result = from_json_schema({})
    assert isinstance(result, Any)

    # Test with pattern constraint
    result = from_json_schema({
        "type": "string",
        "pattern": "^[a-z]+$"
    })
    assert isinstance(result, AllOf)

    # Test with components/schemas (nested definitions)
    result = from_json_schema({
        "type": "object",
        "components": {
            "schemas": {
                "User": {"type": "string"}
            }
        }
    })
    assert isinstance(result, AllOf)


# LLM-generated content at query #10
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    result = to_json_schema(String())
    assert result["type"] == "string"
    assert result.get("minLength") is None
    assert result.get("maxLength") is None

    # Test with String field with allow_null
    result = to_json_schema(String(allow_null=True))
    assert result["type"] == ["string", "null"]

    # Test with String field with min_length and max_length
    result = to_json_schema(String(min_length=5, max_length=10))
    assert result["minLength"] == 5
    assert result["maxLength"] == 10

    # Test with String field with pattern
    result = to_json_schema(String(pattern="^[a-z]+$"))
    assert result["pattern"] == "^[a-z]+$"

    # Test with String field with format
    result = to_json_schema(String(format="email"))
    assert result["format"] == "email"

    # Test with Integer field
    result = to_json_schema(Integer())
    assert result["type"] == "integer"

    # Test with Integer field with allow_null
    result = to_json_schema(Integer(allow_null=True))
    assert result["type"] == ["integer", "null"]

    # Test with Integer field with minimum and maximum
    result = to_json_schema(Integer(minimum=0, maximum=100))
    assert result["minimum"] == 0
    assert result["maximum"] == 100

    # Test with Integer field with exclusive_minimum and exclusive_maximum
    result = to_json_schema(Integer(exclusive_minimum=0, exclusive_maximum=100))
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100

    # Test with Integer field with multiple_of
    result = to_json_schema(Integer(multiple_of=5))
    assert result["multipleOf"] == 5

    # Test with Float field
    result = to_json_schema(Float())
    assert result["type"] == "number"

    # Test with Float field with allow_null
    result = to_json_schema(Float(allow_null=True))
    assert result["type"] == ["number", "null"]

    # Test with Boolean field
    result = to_json_schema(Boolean())
    assert result["type"] == "boolean"

    # Test with Boolean field with allow_null
    result = to_json_schema(Boolean(allow_null=True))
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    result = to_json_schema(Array())
    assert result["type"] == "array"

    # Test with Array field with items
    result = to_json_schema(Array(items=String()))
    assert result["items"]["type"] == "string"

    # Test with Array field with min_items and max_items
    result = to_json_schema(Array(min_items=1, max_items=10))
    assert result["minItems"] == 1
    assert result["maxItems"] == 10

    # Test with Array field with unique_items
    result = to_json_schema(Array(unique_items=True))
    assert result["uniqueItems"] is True

    # Test with Array field with additional_items as bool
    result = to_json_schema(Array(additional_items=False))
    assert result["additionalItems"] is False

    # Test with Array field with additional_items as Field
    result = to_json_schema(Array(additional_items=String()))
    assert result["additionalItems"]["type"] == "string"

    # Test with Object field
    result = to_json_schema(Object())
    assert result["type"] == "object"

    # Test with Object field with properties
    result = to_json_schema(Object(properties={"name": String(), "age": Integer()}))
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"

    # Test with Object field with required
    result = to_json_schema(Object(properties={"name": String()}, required=["name"]))
    assert result["required"] == ["name"]

    # Test with Object field with additional_properties as bool
    result = to_json_schema(Object(additional_properties=False))
    assert result["additionalProperties"] is False

    # Test with Object field with additional_properties as Field
    result = to_json_schema(Object(additional_properties=String()))
    assert result["additionalProperties"]["type"] == "string"

    # Test with Object field with property_names
    result = to_json_schema(Object(property_names=String(pattern="^[a-z]+$")))
    assert "propertyNames" in result
    assert result["propertyNames"]["pattern"] == "^[a-z]+$"

    # Test with Object field with min_properties and max_properties
    result = to_json_schema(Object(min_properties=1, max_properties=5))
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5

    # Test with Choice field
    result = to_json_schema(Choice(choices=[("red", "red"), ("green", "green"), ("blue", "blue")]))
    assert result["enum"] == ["red", "green", "blue"]

    # Test with Const field
    result = to_json_schema(Const(const="constant_value"))
    assert result["const"] == "constant_value"

    # Test with Union field
    result = to_json_schema(Union(any_of=[String(), Integer()]))
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

    # Test with OneOf field
    result = to_json_schema(OneOf(one_of=[String(), Integer()]))
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    result = to_json_schema(AllOf(all_of=[String(), String(max_length=10)]))
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with Not field
    result = to_json_schema(Not(negated=String()))
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test with IfThenElse field
    result = to_json_schema(IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    ))
    assert "if" in result
    assert "then" in result
    assert "else" in result
    assert result["if"]["type"] == "string"
    assert result["then"]["type"] == "integer"
    assert result["else"]["type"] == "boolean"

    # Test with IfThenElse field without else_clause
    result = to_json_schema(IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    ))
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with Definitions
    definitions = Definitions({"StringDef": String(), "IntegerDef": Integer()})
    result = to_json_schema(definitions)
    assert "StringDef" in result
    assert "IntegerDef" in result

    # Test with Reference field
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(Reference(to="User", definitions=definitions))
    assert result["$ref"] == "#/components/schemas/User"
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]

    # Test with Schema field
    result = to_json_schema(Schema(fields={"name": String(), "age": Integer()}))
    assert result["type"] == "object"


# LLM-generated content at query #11
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test basic if-then-else structure
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None

    # Test if-then without else
    data = {
        "if": {"type": "integer"},
        "then": {"minimum": 0},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None

    # Test if-else without then
    data = {
        "if": {"type": "boolean"},
        "else": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None

    # Test if only
    data = {
        "if": {"type": "array"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 1},
        "else": {"type": "null"},
        "default": "test_default",
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == "test_default"

    # Test with complex nested schemas
    data = {
        "if": {"properties": {"type": {"enum": ["A", "B"]}}},
        "then": {"properties": {"value": {"type": "string"}}},
        "else": {"properties": {"value": {"type": "number"}}},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None

    # Test with no default (should use NO_DEFAULT)
    data = {
        "if": {"type": "object"},
        "then": {"minProperties": 1},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == NO_DEFAULT


# LLM-generated content at query #12
#--------------------------

```python
def test_type_from_json_schema():
    """Test type_from_json_schema function with various JSON schema types."""
    
    # Test single type: string
    result = type_from_json_schema({"type": "string"}, definitions)
    assert isinstance(result, String)
    
    # Test single type: integer
    result = type_from_json_schema({"type": "integer"}, definitions)
    assert isinstance(result, Integer)
    
    # Test single type: number
    result = type_from_json_schema({"type": "number"}, definitions)
    assert isinstance(result, Number)
    
    # Test single type: boolean
    result = type_from_json_schema({"type": "boolean"}, definitions)
    assert isinstance(result, Boolean)
    
    # Test single type: array
    result = type_from_json_schema({"type": "array"}, definitions)
    assert isinstance(result, Array)
    
    # Test single type: object
    result = type_from_json_schema({"type": "object"}, definitions)
    assert isinstance(result, Object)
    
    # Test multiple types (union)
    result = type_from_json_schema({"type": ["string", "integer"]}, definitions)
    assert isinstance(result, Union)
    
    # Test null type only
    result = type_from_json_schema({"type": "null"}, definitions)
    assert isinstance(result, Const)
    
    # Test type with null (allow_null)
    result = type_from_json_schema({"type": ["string", "null"]}, definitions)
    assert isinstance(result, String)
    
    # Test multiple types with null
    result = type_from_json_schema({"type": ["string", "integer", "null"]}, definitions)
    assert isinstance(result, Union)
    
    # Test with constraints (minLength, maxLength)
    result = type_from_json_schema(
        {"type": "string", "minLength": 1, "maxLength": 10}, definitions
    )
    assert isinstance(result, String)
    
    # Test with constraints (minimum, maximum)
    result = type_from_json_schema(
        {"type": "integer", "minimum": 0, "maximum": 100}, definitions
    )
    assert isinstance(result, Integer)
    
    # Test with pattern
    result = type_from_json_schema(
        {"type": "string", "pattern": "^[a-z]+$"}, definitions
    )
    assert isinstance(result, String)
    
    # Test with items (array items)
    result = type_from_json_schema(
        {"type": "array", "items": {"type": "string"}}, definitions
    )
    assert isinstance(result, Array)
    
    # Test with properties (object properties)
    result = type_from_json_schema(
        {"type": "object", "properties": {"name": {"type": "string"}}}, definitions
    )
    assert isinstance(result, Object)
    
    # Test no type constraint
    result = type_from_json_schema({}, definitions)
    assert isinstance(result, NeverMatch)
    
    # Test empty type list
    result = type_from_json_schema({"type": []}, definitions)
    assert isinstance(result, NeverMatch)


# LLM-generated content at query #13
#--------------------------

```python
def test_from_json_schema_type():
    """Test from_json_schema_type function with various type strings and data."""
    defs = Definitions()
    
    # Test number type
    data = {
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 10,
        "exclusiveMaximum": 90,
        "multipleOf": 5,
        "default": 50
    }
    result = from_json_schema_type(data, "number", False, defs)
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.exclusive_minimum == 10
    assert result.exclusive_maximum == 90
    assert result.multiple_of == 5
    assert result.default == 50
    assert result.allow_null is False
    
    # Test number type with allow_null
    result = from_json_schema_type(data, "number", True, defs)
    assert isinstance(result, Float)
    assert result.allow_null is True
    
    # Test integer type
    data = {
        "minimum": 1,
        "maximum": 10,
        "default": 5
    }
    result = from_json_schema_type(data, "integer", False, defs)
    assert isinstance(result, Integer)
    assert result.minimum == 1
    assert result.maximum == 10
    assert result.default == 5
    
    # Test string type
    data = {
        "minLength": 2,
        "maxLength": 50,
        "pattern": "^[a-z]+$",
        "format": "email",
        "default": "test"
    }
    result = from_json_schema_type(data, "string", False, defs)
    assert isinstance(result, String)
    assert result.min_length == 2
    assert result.max_length == 50
    assert result.pattern == "^[a-z]+$"
    assert result.format == "email"
    assert result.default == "test"
    
    # Test string type with minLength 0
    data = {"minLength": 0}
    result = from_json_schema_type(data, "string", False, defs)
    assert isinstance(result, String)
    assert result.allow_blank is True
    assert result.min_length is None
    
    # Test string type with minLength 1
    data = {"minLength": 1}
    result = from_json_schema_type(data, "string", False, defs)
    assert result.min_length is None
    
    # Test boolean type
    data = {"default": True}
    result = from_json_schema_type(data, "boolean", False, defs)
    assert isinstance(result, Boolean)
    assert result.default is True
    assert result.allow_null is False
    
    # Test boolean type with allow_null
    result = from_json_schema_type(data, "boolean", True, defs)
    assert result.allow_null is True
    
    # Test array type with no items
    data = {}
    result = from_json_schema_type(data, "array", False, defs)
    assert isinstance(result, Array)
    assert result.items is None
    assert result.min_items == 0
    assert result.max_items is None
    assert result.unique_items is False
    
    # Test array type with items schema
    data = {
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["a", "b"]
    }
    result = from_json_schema_type(data, "array", False, defs)
    assert isinstance(result, Array)
    assert result.min_items == 1
    assert result.max_items == 10
    assert result.unique_items is True
    assert result.default == ["a", "b"]
    
    # Test array type with items as list
    data = {
        "items": [{"type": "string"}, {"type": "integer"}]
    }
    result = from_json_schema_type(data, "array", False, defs)
    assert isinstance(result, Array)
    assert isinstance(result.items, list)
    assert len(result.items) == 2
    
    # Test array type with additionalItems as boolean
    data = {"additionalItems": False}
    result = from_json_schema_type(data, "array", False, defs)
    assert result.additional_items is False
    
    # Test array type with additionalItems as schema
    data = {"additionalItems": {"type": "number"}}
    result = from_json_schema_type(data, "array", False, defs)
    assert isinstance(result.additional_items, Field)
    
    # Test object type with no properties
    data = {}
    result = from_json_schema_type(data, "object", False, defs)
    assert isinstance(result, Object)
    assert result.properties is None
    
    # Test object type with properties
    data = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 5
    }
    result = from_json_schema_type(data, "object", False, defs)
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert "age" in result.properties
    assert result.required == ["name"]
    assert result.min_properties == 1
    assert result.max_properties == 5
    
    # Test object type with patternProperties
    data = {
        "patternProperties": {
            "^S_": {"type": "string"},
            "^I_": {"type": "integer"}
        }
    }
    result = from_json_schema_type(data, "object", False, defs)
    assert "^S_" in result.pattern_properties
    assert "^I_" in result.pattern_properties
    
    # Test object type with additionalProperties as boolean
    data = {"additionalProperties": False}
    result = from_json_schema_type(data, "object", False, defs)
    assert result.additional_properties is False
    
    # Test object type with additionalProperties as schema
    data = {"additionalProperties": {"type": "string"}}
    result = from_json_schema_type(data, "object", False, defs)
    assert isinstance(result.additional_properties, Field)
    
    # Test object type with propertyNames
    data = {
        "propertyNames": {"pattern": "^[a-z]+$"}
    }
    result = from_json_schema_type(data, "object", False, defs)
    assert isinstance(result.property_names, Field)
    
    # Test object type with allow_null
    data = {}
    result = from_json_schema_type(data, "object", True, defs)
    assert result.allow_null is True


# LLM-generated content at query #14
#--------------------------

```python
def test_ref_from_json_schema():
    # Test basic reference creation
    data = {"$ref": "#/components/schemas/User"}
    defs = Definitions()
    result = ref_from_json_schema(data, defs)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is defs

    # Test reference with different path
    data = {"$ref": "#/definitions/Product"}
    defs = Definitions()
    result = ref_from_json_schema(data, defs)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/Product"

    # Test reference with nested path
    data = {"$ref": "#/components/schemas/nested/Item"}
    defs = Definitions()
    result = ref_from_json_schema(data, defs)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/nested/Item"

    # Test that invalid ref style raises assertion error
    data = {"$ref": "components/schemas/User"}
    defs = Definitions()
    try:
        ref_from_json_schema(data, defs)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "Unsupported $ref style" in str(e)

    # Test reference definitions are preserved
    defs = Definitions()
    defs["#/definitions/Test"] = String()
    data = {"$ref": "#/definitions/Test"}
    result = ref_from_json_schema(data, defs)
    assert result.definitions is defs
    assert "#/definitions/Test" in result.definitions


# LLM-generated content at query #15
#--------------------------

```python
def test_ref_from_json_schema():
    # Test basic reference creation
    defs = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, defs)
    
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is defs


def test_ref_from_json_schema_with_nested_path():
    # Test reference with nested path
    defs = Definitions()
    data = {"$ref": "#/definitions/Address"}
    result = ref_from_json_schema(data, defs)
    
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/Address"


def test_ref_from_json_schema_preserves_definitions():
    # Test that definitions object is preserved
    defs = Definitions()
    defs["#/components/schemas/Test"] = String()
    data = {"$ref": "#/components/schemas/Test"}
    result = ref_from_json_schema(data, defs)
    
    assert result.definitions is defs
    assert "#/components/schemas/Test" in defs


def test_ref_from_json_schema_invalid_ref_style():
    # Test that unsupported $ref style raises AssertionError
    defs = Definitions()
    data = {"$ref": "schemas/User"}
    
    try:
        ref_from_json_schema(data, defs)
        assert False, "Expected AssertionError for unsupported $ref style"
    except AssertionError as e:
        assert "Unsupported $ref style" in str(e)


def test_ref_from_json_schema_with_empty_definitions():
    # Test reference with empty definitions object
    defs = Definitions()
    data = {"$ref": "#/components/schemas/Empty"}
    result = ref_from_json_schema(data, defs)
    
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/Empty"
    assert len(defs) == 0


def test_ref_from_json_schema_complex_path():
    # Test reference with complex path
    defs = Definitions()
    data = {"$ref": "#/components/schemas/models/v1/User"}
    result = ref_from_json_schema(data, defs)
    
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/models/v1/User"


# LLM-generated content at query #16
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"},
        "default": "test"
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    assert result.default == "test"

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 3}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "number"},
        "else": {"type": "boolean"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None

    # Test with only if clause
    data = {
        "if": {"type": "array"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with complex nested schemas
    data = {
        "if": {"properties": {"type": {"const": "object"}}},
        "then": {"required": ["name", "age"]},
        "else": {"type": "null"},
        "default": None
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default is None

    # Test with no default value
    data = {
        "if": {"type": "string"},
        "then": {"pattern": "^[a-z]+$"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert result.default is NO_DEFAULT


# LLM-generated content at query #17
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": "test_default"
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    assert result.default == "test_default"

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None

    # Test with only if clause
    data = {
        "if": {"type": "string"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "integer"}},
        "else": {"type": "number"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None

    # Test with NO_DEFAULT
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert result.default == NO_DEFAULT


# LLM-generated content at query #18
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert "default" not in result

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with String field with min_length
    string_field_min = String(min_length=5, allow_null=False)
    result = to_json_schema(string_field_min)
    assert result["minLength"] == 5

    # Test with String field with max_length
    string_field_max = String(max_length=10, allow_null=False)
    result = to_json_schema(string_field_max)
    assert result["maxLength"] == 10

    # Test with Integer field
    integer_field = Integer(allow_null=False)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"

    # Test with Float field
    float_field = Float(allow_null=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"

    # Test with Integer field with minimum and maximum
    integer_field_range = Integer(minimum=0, maximum=100, allow_null=False)
    result = to_json_schema(integer_field_range)
    assert result["minimum"] == 0
    assert result["maximum"] == 100

    # Test with Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test with Array field
    array_field = Array(allow_null=False)
    result = to_json_schema(array_field)
    assert result["type"] == "array"

    # Test with Array field with items
    array_field_items = Array(items=String(), allow_null=False)
    result = to_json_schema(array_field_items)
    assert result["items"]["type"] == "string"

    # Test with Array field with min_items and max_items
    array_field_range = Array(min_items=1, max_items=5, allow_null=False)
    result = to_json_schema(array_field_range)
    assert result["minItems"] == 1
    assert result["maxItems"] == 5

    # Test with Object field
    object_field = Object(allow_null=False)
    result = to_json_schema(object_field)
    assert result["type"] == "object"

    # Test with Object field with properties
    object_field_props = Object(
        properties={"name": String(), "age": Integer()},
        allow_null=False
    )
    result = to_json_schema(object_field_props)
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]

    # Test with Object field with required properties
    object_field_required = Object(required=["name"], allow_null=False)
    result = to_json_schema(object_field_required)
    assert result["required"] == ["name"]

    # Test with Choice field
    choice_field = Choice(choices=[("a", 1), ("b", 2)])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(), String()])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without else clause
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with Definitions
    definitions = Definitions({"string_def": String()})
    result = to_json_schema(definitions)
    assert "components" in result or isinstance(result, dict)

    # Test nested structures
    nested_object = Object(
        properties={
            "nested": Object(properties={"field": String()})
        },
        allow_null=False
    )
    result = to_json_schema(nested_object)
    assert "properties" in result
    assert "nested" in result["properties"]

    # Test with default value
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result.get("default") == "default_value"

    # Test with pattern
    import re
    string_with_pattern = String(pattern=r"^[a-z]+$")
    result = to_json_schema(string_with_pattern)
    assert "pattern" in result

    # Test with format
    string_with_format = String(format="email")
    result = to_json_schema(string_with_format)
    assert result["format"] == "email"

    # Test Array with unique_items
    array_unique = Array(unique_items=True, allow_null=False)
    result = to_json_schema(array_unique)
    assert result["uniqueItems"] is True

    # Test Object with additional_properties as bool
    object_additional_bool = Object(additional_properties=False, allow_null=False)
    result = to_json_schema(object_additional_bool)
    assert result["additionalProperties"] is False

    # Test Object with additional_properties as Field
    object_additional_field = Object(
        additional_properties=String(),
        allow_null=False
    )
    result = to_json_schema(object_additional_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test invalid field type raises ValueError
    class InvalidField(Field):
        pass

    invalid_field = InvalidField()
    try:
        to_json_schema(invalid_field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


# LLM-generated content at query #19
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type without null
    result = type_from_json_schema({"type": "string"}, definitions)
    assert isinstance(result, String)
    assert result.allow_null is False

    # Test single type with null
    result = type_from_json_schema({"type": ["string", "null"]}, definitions)
    assert isinstance(result, String)
    assert result.allow_null is True

    # Test multiple types without null
    result = type_from_json_schema({"type": ["string", "integer"]}, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is False
    assert len(result.any_of) == 2

    # Test multiple types with null
    result = type_from_json_schema({"type": ["string", "integer", "null"]}, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True
    assert len(result.any_of) == 2

    # Test null only
    result = type_from_json_schema({"type": "null"}, definitions)
    assert isinstance(result, Const)
    assert result.const is None

    # Test no type with null allowed
    result = type_from_json_schema({}, definitions)
    assert isinstance(result, NeverMatch)

    # Test string type with constraints
    result = type_from_json_schema(
        {"type": "string", "minLength": 1, "maxLength": 10}, definitions
    )
    assert isinstance(result, String)
    assert result.min_length == 1
    assert result.max_length == 10

    # Test integer type with constraints
    result = type_from_json_schema(
        {"type": "integer", "minimum": 0, "maximum": 100}, definitions
    )
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.maximum == 100

    # Test number type with constraints
    result = type_from_json_schema(
        {"type": "number", "minimum": 0.5, "exclusiveMaximum": 10.5}, definitions
    )
    assert isinstance(result, Number)
    assert result.minimum == 0.5
    assert result.exclusive_maximum == 10.5

    # Test array type
    result = type_from_json_schema(
        {"type": "array", "items": {"type": "string"}}, definitions
    )
    assert isinstance(result, Array)

    # Test object type
    result = type_from_json_schema(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        definitions,
    )
    assert isinstance(result, Object)

    # Test boolean type
    result = type_from_json_schema({"type": "boolean"}, definitions)
    assert isinstance(result, Boolean)

    # Test array of types
    result = type_from_json_schema({"type": ["string", "number", "null"]}, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True
    assert len(result.any_of) == 2


# LLM-generated content at query #20
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None

    # Test with only if clause
    data = {
        "if": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "default": 42,
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == 42

    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "number"}},
        "else": {"type": "string", "minLength": 5},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_if_then_else_from_json_schema():
    """Test if_then_else_from_json_schema function."""
    
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    
    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None
    
    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None
    
    # Test with only if clause
    data = {
        "if": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None
    
    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "default": "default_value",
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == "default_value"
    
    # Test with complex nested conditions
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "number"}},
        "else": {"enum": [1, 2, 3]},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    
    # Test with NO_DEFAULT
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert result.default == NO_DEFAULT


# LLM-generated content at query #22
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test basic if-then-else with all clauses
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 1},
        "else": {"type": "number"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None

    # Test if-then without else
    data = {
        "if": {"type": "object"},
        "then": {"minProperties": 1},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None

    # Test if-else without then
    data = {
        "if": {"type": "array"},
        "else": {"minItems": 0},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None

    # Test if only
    data = {
        "if": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "string"},
        "default": "test_default",
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == "test_default"

    # Test with nested complex schemas
    data = {
        "if": {"properties": {"name": {"type": "string"}}},
        "then": {"required": ["name"]},
        "else": {"additionalProperties": False},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #23
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test basic if-then-else with all clauses
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 1},
        "else": {"type": "number"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None

    # Test if-then without else
    data_no_else = {
        "if": {"type": "string"},
        "then": {"minLength": 1}
    }
    result_no_else = if_then_else_from_json_schema(data_no_else, definitions)
    assert isinstance(result_no_else, IfThenElse)
    assert result_no_else.if_clause is not None
    assert result_no_else.then_clause is not None
    assert result_no_else.else_clause is None

    # Test if-else without then
    data_no_then = {
        "if": {"type": "string"},
        "else": {"type": "number"}
    }
    result_no_then = if_then_else_from_json_schema(data_no_then, definitions)
    assert isinstance(result_no_then, IfThenElse)
    assert result_no_then.if_clause is not None
    assert result_no_then.then_clause is None
    assert result_no_then.else_clause is not None

    # Test if only (no then, no else)
    data_if_only = {
        "if": {"type": "string"}
    }
    result_if_only = if_then_else_from_json_schema(data_if_only, definitions)
    assert isinstance(result_if_only, IfThenElse)
    assert result_if_only.if_clause is not None
    assert result_if_only.then_clause is None
    assert result_if_only.else_clause is None

    # Test with default value
    data_with_default = {
        "if": {"type": "string"},
        "then": {"minLength": 1},
        "else": {"type": "number"},
        "default": "test_default"
    }
    result_with_default = if_then_else_from_json_schema(data_with_default, definitions)
    assert isinstance(result_with_default, IfThenElse)
    assert result_with_default.default == "test_default"

    # Test with complex nested conditions
    data_complex = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"required": ["name"]},
        "else": {"type": "array"}
    }
    result_complex = if_then_else_from_json_schema(data_complex, definitions)
    assert isinstance(result_complex, IfThenElse)
    assert result_complex.if_clause is not None
    assert result_complex.then_clause is not None
    assert result_complex.else_clause is not None


# LLM-generated content at query #24
#--------------------------

```python
def test_if_then_else_from_json_schema():
    """Test if_then_else_from_json_schema function."""
    defs = Definitions()
    
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    
    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None
    
    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None
    
    # Test with only if clause
    data = {
        "if": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None
    
    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "default": "test_default",
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.default == "test_default"
    
    # Test with NO_DEFAULT when no default is provided
    data = {
        "if": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert result.default is NO_DEFAULT
    
    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "integer"}},
        "else": {"enum": [1, 2, 3]},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #25
#--------------------------

```python
def test_if_then_else_from_json_schema():
    """Test if_then_else_from_json_schema function"""
    
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 1},
        "else": {"type": "number"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 1},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "number"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None

    # Test with only if clause
    data = {
        "if": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "string"},
        "else": {"type": "number"},
        "default": "test",
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == "test"

    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "object", "required": ["name"]},
        "else": {"type": "array", "items": {"type": "string"}},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #26
#--------------------------

```python
def test_type_from_json_schema():
    # Test with single type string
    data = {"type": "string", "minLength": 1}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    
    # Test with multiple type strings
    data = {"type": ["string", "integer"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    
    # Test with null type allowed
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True
    
    # Test with single type and null
    data = {"type": ["integer", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)
    assert result.allow_null is True
    
    # Test with only null type
    data = {"type": "null"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)
    
    # Test with no type
    data = {"minLength": 1}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, NeverMatch)
    
    # Test with number type
    data = {"type": "number", "minimum": 0, "maximum": 100}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Number)
    
    # Test with boolean type
    data = {"type": "boolean"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Boolean)
    
    # Test with array type
    data = {"type": "array", "items": {"type": "string"}}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Array)
    
    # Test with object type
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Object)
    
    # Test with multiple types including null
    data = {"type": ["string", "number", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True
    assert len(result.any_of) == 2


# LLM-generated content at query #27
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    
    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String field with constraints
    string_field_constrained = String(min_length=5, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_field_constrained)
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test String field with blank not allowed
    string_field_no_blank = String(allow_blank=False)
    result = to_json_schema(string_field_no_blank)
    assert result["minLength"] == 1
    
    # Test Integer field
    integer_field = Integer(allow_null=False)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    
    # Test Integer field with constraints
    integer_constrained = Integer(minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(integer_constrained)
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test Float field with exclusive bounds
    float_constrained = Float(exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_constrained)
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0
    
    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Boolean field with null
    boolean_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(items=String())
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"
    
    # Test Array field with constraints
    array_constrained = Array(min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_constrained)
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    
    # Test Array with tuple items
    array_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Array with additional_items as bool
    array_additional_bool = Array(additional_items=False)
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] == False
    
    # Test Array with additional_items as Field
    array_additional_field = Array(additional_items=String())
    result = to_json_schema(array_additional_field)
    assert isinstance(result["additionalItems"], dict)
    
    # Test Object field
    obj_field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    
    # Test Object field with constraints
    obj_constrained = Object(
        min_properties=1,
        max_properties=5,
        required=["name"]
    )
    result = to_json_schema(obj_constrained)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    assert result["required"] == ["name"]
    
    # Test Object with pattern_properties
    obj_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    
    # Test Object with additional_properties as bool
    obj_additional_bool = Object(additional_properties=False)
    result = to_json_schema(obj_additional_bool)
    assert result["additionalProperties"] == False
    
    # Test Object with additional_properties as Field
    obj_additional_field = Object(additional_properties=String())
    result = to_json_schema(obj_additional_field)
    assert isinstance(result["additionalProperties"], dict)
    
    # Test Object with property_names
    obj_prop_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_prop_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", 1), ("b", 2)])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const=42)
    result = to_json_schema(const_field)
    assert result["const"] == 42
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"
    
    # Test IfThenElse field
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without else
    if_then = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with Definitions
    definitions = Definitions({
        "StringDef": String(),
        "IntDef": Integer()
    })
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    
    # Test Reference field
    reference = Reference(to="StringDef", definitions=Definitions({"StringDef": String()}))
    result = to_json_


# LLM-generated content at query #28
#--------------------------

```python
def test_to_json_schema():
    """Test to_json_schema conversion for various field types."""
    
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"
    
    # Test String field with allow_null
    string_null = String(allow_null=True)
    result = to_json_schema(string_null)
    assert result["type"] == ["string", "null"]
    
    # Test String field with allow_blank
    string_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_blank)
    assert "minLength" not in result
    
    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Integer field with allow_null
    int_null = Integer(allow_null=True)
    result = to_json_schema(int_null)
    assert result["type"] == ["integer", "null"]
    
    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0, exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0
    
    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test Boolean field with allow_null
    bool_null = Boolean(allow_null=True)
    result = to_json_schema(bool_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    assert "items" in result
    
    # Test Array field with allow_null
    array_null = Array(allow_null=True, items=Integer())
    result = to_json_schema(array_null)
    assert result["type"] == ["array", "null"]
    
    # Test Array with list of items
    array_list = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_list)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Array with additional_items
    array_additional = Array(allow_null=False, items=String(), additional_items=False)
    result = to_json_schema(array_additional)
    assert result["additionalItems"] == False
    
    # Test Object field
    obj_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    
    # Test Object field with allow_null
    obj_null = Object(allow_null=True, properties={"id": Integer()})
    result = to_json_schema(obj_null)
    assert result["type"] == ["object", "null"]
    
    # Test Object with pattern_properties
    obj_pattern = Object(allow_null=False, pattern_properties={"^S_": String()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]
    
    # Test Object with additional_properties
    obj_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(obj_additional)
    assert result["additionalProperties"] == False
    
    # Test Object with property_names
    obj_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", 1), ("b", 2), ("c", 3)])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b", "c"]
    
    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=1)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without else_clause
    if_then = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with default values
    field_with_default = String(default="default_value")
    result = to_json_schema(field_with_default)
    assert result["default"] == "default_value"
    
    # Test with Definitions
    definitions = Definitions({"MyString": String(), "MyInt": Integer()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas


# LLM-generated content at query #29
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type string
    data = {"type": "string"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)

    # Test single type with constraints
    data = {"type": "integer", "minimum": 0}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)

    # Test multiple types (union)
    data = {"type": ["string", "integer"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)

    # Test type with null
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test null type only
    data = {"type": "null"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)

    # Test multiple types with null
    data = {"type": ["string", "number", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test number type
    data = {"type": "number"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Number)

    # Test boolean type
    data = {"type": "boolean"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Boolean)

    # Test array type
    data = {"type": "array", "items": {"type": "string"}}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Array)

    # Test object type
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Object)

    # Test no type constraint
    data = {}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, NeverMatch)

    # Test empty type array
    data = {"type": []}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, NeverMatch)

    # Test string type with pattern
    data = {"type": "string", "pattern": "^[a-z]+$"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)

    # Test multiple types without null
    data = {"type": ["string", "boolean", "integer"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is False

    # Test float type
    data = {"type": "number"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, (Number, Float))


# LLM-generated content at query #30
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, min_length=2, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 2
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with String field with blank not allowed
    string_field_no_blank = String(allow_null=False, allow_blank=False)
    result = to_json_schema(string_field_no_blank)
    assert result["minLength"] == 1

    # Test with Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test with Float field
    float_field = Float(allow_null=False, exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0

    # Test with Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test with Array field with tuple items
    array_field_tuple = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test with Array field with additionalItems as bool
    array_field_additional = Array(allow_null=False, items=String(), additional_items=False)
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"] is False

    # Test with Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10

    # Test with Object field with pattern properties
    object_field_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String()}
    )
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]

    # Test with Object field with additional properties as bool
    object_field_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(object_field_additional)
    assert result["additionalProperties"] is False

    # Test with Object field with property names
    object_field_prop_names = Object(
        allow_null=False,
        property_names=String(pattern="^[a-z]+$")
    )
    result = to_json_schema(object_field_prop_names)
    assert "propertyNames" in result

    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(), Object()])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without then and else
    if_then_else_minimal = IfThenElse(if_clause=String())
    result = to_json_schema(if_then_else_minimal)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result

    # Test with Definitions
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]

    # Test with Reference field
    reference_field = Reference(to="User", definitions=Definitions({"User": Object()}))
    result = to_json_schema(reference_field)
    assert result["$ref"] == "#/components/schemas/User"
    assert "components" in result

    # Test with default values
    string_with_default = String(default="test_default")
    result = to_json_schema(string_with_default)
    assert result["default"] == "test_default"

    # Test error with unsupported field type
    class UnsupportedField(Field):
        pass

    try:
        to_json_schema(UnsupportedField())
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


# LLM-generated content at query #31
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type string
    data = {"type": "string"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)

    # Test single type with null
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    assert result.allow_null is True

    # Test multiple types (union)
    data = {"type": ["string", "integer"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)

    # Test multiple types with null
    data = {"type": ["string", "integer", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test only null type
    data = {"type": "null"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)
    assert result.const_value is None

    # Test null type only (no other types)
    data = {"type": ["null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)

    # Test integer type
    data = {"type": "integer"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)

    # Test number type
    data = {"type": "number"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Number)

    # Test boolean type
    data = {"type": "boolean"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Boolean)

    # Test array type
    data = {"type": "array"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Array)

    # Test object type
    data = {"type": "object"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Object)

    # Test with constraints (string with minLength)
    data = {"type": "string", "minLength": 5}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    assert result.min_length == 5

    # Test with constraints (integer with minimum)
    data = {"type": "integer", "minimum": 10}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)
    assert result.minimum == 10

    # Test empty type list with allow_null False
    data = {}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, NeverMatch)

    # Test multiple types with constraints
    data = {"type": ["string", "number"], "minimum": 0}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)


# LLM-generated content at query #32
#--------------------------

```python
def test_type_from_json_schema():
    # Test with single type string
    data = {"type": "string", "minLength": 1}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    
    # Test with multiple type strings
    data = {"type": ["string", "integer"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    
    # Test with null type
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True
    
    # Test with only null type
    data = {"type": "null"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)
    
    # Test with no type (allow_null=False)
    data = {}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, NeverMatch)
    
    # Test integer type
    data = {"type": "integer", "minimum": 0}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)
    
    # Test number type
    data = {"type": "number", "maximum": 100}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Number)
    
    # Test boolean type
    data = {"type": "boolean"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Boolean)
    
    # Test array type
    data = {"type": "array", "items": {"type": "string"}}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Array)
    
    # Test object type
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Object)
    
    # Test with allow_null=True for single type
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True
    
    # Test with multiple types and null
    data = {"type": ["string", "integer", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True
    assert len(result.any_of) == 2


# LLM-generated content at query #33
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with String field with blank allowed
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result

    # Test with Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=5, exclusive_maximum=95, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5
    assert result["exclusiveMaximum"] == 95
    assert result["multipleOf"] == 5

    # Test with Integer field allowing null
    int_field_null = Integer(allow_null=True)
    result = to_json_schema(int_field_null)
    assert result["type"] == ["integer", "null"]

    # Test with Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test with Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=5, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test with Array field allowing null
    array_field_null = Array(allow_null=True)
    result = to_json_schema(array_field_null)
    assert result["type"] == ["array", "null"]

    # Test with Array field with tuple items
    array_field_tuple = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test with Array field with additional items boolean
    array_field_additional = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"] is False

    # Test with Array field with additional items field
    array_field_additional_field = Array(allow_null=False, additional_items=String())
    result = to_json_schema(array_field_additional_field)
    assert isinstance(result["additionalItems"], dict)

    # Test with Object field
    object_field = Object(allow_null=False, properties={"name": String(), "age": Integer()}, required=["name"], min_properties=1, max_properties=10)
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10

    # Test with Object field allowing null
    object_field_null = Object(allow_null=True)
    result = to_json_schema(object_field_null)
    assert result["type"] == ["object", "null"]

    # Test with Object field with pattern properties
    object_field_pattern = Object(allow_null=False, pattern_properties={"^S_": String()})
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result

    # Test with Object field with additional properties boolean
    object_field_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(object_field_additional)
    assert result["additionalProperties"] is False

    # Test with Object field with additional properties field
    object_field_additional_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(object_field_additional_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test with Object field with property names
    object_field_property_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_field_property_names)
    assert "propertyNames" in result

    # Test with Choice field
    choice_field = Choice(choices=[("a", 1), ("b", 2)])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=5)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without then and else
    if_only_field = IfThenElse(if_clause=String())
    result = to_json_schema(if_only_field)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with Definitions
    definitions = Definitions({"StringDef": String(), "IntDef": Integer()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["


# LLM-generated content at query #34
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type: string
    result = type_from_json_schema({"type": "string"}, Definitions())
    assert isinstance(result, String)
    
    # Test single type: integer
    result = type_from_json_schema({"type": "integer"}, Definitions())
    assert isinstance(result, Integer)
    
    # Test single type: number
    result = type_from_json_schema({"type": "number"}, Definitions())
    assert isinstance(result, Number)
    
    # Test single type: boolean
    result = type_from_json_schema({"type": "boolean"}, Definitions())
    assert isinstance(result, Boolean)
    
    # Test single type: array
    result = type_from_json_schema({"type": "array"}, Definitions())
    assert isinstance(result, Array)
    
    # Test single type: object
    result = type_from_json_schema({"type": "object"}, Definitions())
    assert isinstance(result, Object)
    
    # Test multiple types (union)
    result = type_from_json_schema({"type": ["string", "integer"]}, Definitions())
    assert isinstance(result, Union)
    
    # Test type with null
    result = type_from_json_schema({"type": ["string", "null"]}, Definitions())
    assert isinstance(result, Union)
    assert result.allow_null is True
    
    # Test null type only
    result = type_from_json_schema({"type": "null"}, Definitions())
    assert isinstance(result, Const)
    
    # Test empty type list
    result = type_from_json_schema({}, Definitions())
    assert isinstance(result, NeverMatch)
    
    # Test type with constraints (string with minLength)
    result = type_from_json_schema(
        {"type": "string", "minLength": 5}, Definitions()
    )
    assert isinstance(result, String)
    assert result.min_length == 5
    
    # Test type with constraints (integer with minimum)
    result = type_from_json_schema(
        {"type": "integer", "minimum": 10}, Definitions()
    )
    assert isinstance(result, Integer)
    assert result.minimum == 10
    
    # Test type with constraints (array with items)
    result = type_from_json_schema(
        {"type": "array", "items": {"type": "string"}}, Definitions()
    )
    assert isinstance(result, Array)
    
    # Test type with constraints (object with properties)
    result = type_from_json_schema(
        {"type": "object", "properties": {"name": {"type": "string"}}}, 
        Definitions()
    )
    assert isinstance(result, Object)
    
    # Test multiple types with constraints
    result = type_from_json_schema(
        {"type": ["string", "integer"], "minLength": 1}, Definitions()
    )
    assert isinstance(result, Union)
    
    # Test type with pattern constraint
    result = type_from_json_schema(
        {"type": "string", "pattern": "^[a-z]+$"}, Definitions()
    )
    assert isinstance(result, String)
    assert result.pattern == "^[a-z]+$"
    
    # Test number type with multipleOf
    result = type_from_json_schema(
        {"type": "number", "multipleOf": 0.5}, Definitions()
    )
    assert isinstance(result, Number)
    
    # Test type with exclusive bounds
    result = type_from_json_schema(
        {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100},
        Definitions()
    )
    assert isinstance(result, Integer)


# LLM-generated content at query #35
#--------------------------

```python
def test_type_from_json_schema():
    # Test with single type string
    data = {"type": "string", "minLength": 1}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)

    # Test with single type integer
    data = {"type": "integer", "minimum": 0}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)

    # Test with single type boolean
    data = {"type": "boolean"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Boolean)

    # Test with single type number
    data = {"type": "number"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Number)

    # Test with single type array
    data = {"type": "array", "items": {"type": "string"}}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Array)

    # Test with single type object
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Object)

    # Test with multiple types (Union)
    data = {"type": ["string", "integer"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)

    # Test with null type allowed
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test with only null type
    data = {"type": "null"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)

    # Test with no type constraint but with null allowed
    data = {"type": ["null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)

    # Test with multiple types including null
    data = {"type": ["string", "number", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test with string type and constraints
    data = {"type": "string", "pattern": "^[a-z]+$", "maxLength": 10}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)

    # Test with number type and constraints
    data = {"type": "number", "minimum": 0, "maximum": 100}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Number)

    # Test with empty type list
    data = {}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, NeverMatch)

    # Test with array of types as list
    data = {"type": ["string", "integer", "boolean"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)


# LLM-generated content at query #36
#--------------------------

```python
def test_to_json_schema():
    # Test Any type
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch type
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    
    # Test String field with null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String with constraints
    string_constrained = String(min_length=2, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_constrained)
    assert result["minLength"] == 2
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test Integer field
    int_field = Integer(allow_null=False)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    
    # Test Integer with constraints
    int_constrained = Integer(minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(int_constrained)
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test Float with exclusive bounds
    float_exclusive = Float(exclusive_minimum=0.0, exclusive_maximum=100.0)
    result = to_json_schema(float_exclusive)
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 100.0
    
    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test Boolean with null
    bool_null = Boolean(allow_null=True)
    result = to_json_schema(bool_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"
    
    # Test Array with additional items
    array_additional = Array(additional_items=String())
    result = to_json_schema(array_additional)
    assert "additionalItems" in result
    
    # Test Object field
    obj_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test Object with pattern properties
    obj_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    
    # Test Object with additional properties
    obj_additional_bool = Object(additional_properties=False)
    result = to_json_schema(obj_additional_bool)
    assert result["additionalProperties"] is False
    
    obj_additional_field = Object(additional_properties=String())
    result = to_json_schema(obj_additional_field)
    assert isinstance(result["additionalProperties"], dict)
    
    # Test Object with property names constraint
    obj_prop_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_prop_names)
    assert "propertyNames" in result
    
    # Test Object with min/max properties
    obj_props_range = Object(min_properties=1, max_properties=10)
    result = to_json_schema(obj_props_range)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Apple"), ("b", "Banana")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Choice(choices=[("a", "A")])])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without else clause
    if_then = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String()
    )
    result = to_json_schema(if_then)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with default values
    string_with_default = String(default="hello")
    result = to_json_schema(string_with_default)
    assert result.get("default") == "hello"
    
    # Test Definitions
    definitions = Definitions({
        "User": Object(properties={"name": String()})
    })
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    
    # Test Reference field
    ref_definitions = Definitions({
        "User": Object(properties={"name": String()})
    })
    ref_field = Reference(to="User", definitions=ref_definitions)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/User"
    assert "components" in result


# LLM-generated content at query #37
#--------------------------

```python
def test_to_json_schema():
    # Test Any() returns True
    assert to_json_schema(Any()) is True

    # Test NeverMatch() returns False
    assert to_json_schema(NeverMatch()) is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0, exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0

    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test Array field with items
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=5, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test Array with tuple items
    array_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10

    # Test Object with pattern properties
    object_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(object_pattern)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]

    # Test Object with additional properties as Field
    object_additional = Object(additional_properties=String())
    result = to_json_schema(object_additional)
    assert isinstance(result["additionalProperties"], dict)

    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Choice(choices=[("a", "A")])])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse without else
    if_then_field = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test Reference field with definitions
    definitions = Definitions()
    ref_field = Reference(to="MySchema", definitions=definitions)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/MySchema"

    # Test with Definitions object
    defs = Definitions()
    result = to_json_schema(defs)
    assert isinstance(result, dict)

    # Test invalid field type raises error
    try:
        class InvalidField:
            pass
        invalid = InvalidField()
        to_json_schema(invalid)
        assert False, "Should have raised ValueError"
    except (ValueError, AttributeError, TypeError):
        pass

    # Test with default values
    string_with_default = String(default="hello")
    result = to_json_schema(string_with_default)
    assert "default" in result
    assert result["default"] == "hello"

    # Test String with allow_blank=False
    string_no_blank = String(allow_blank=False)
    result = to_json_schema(string_no_blank)
    assert result["minLength"] == 1

    # Test Array with additional_items as bool
    array_additional_bool = Array(additional_items=False)
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] is False

    # Test Array with additional_items as Field
    array_additional_field = Array(additional_items=String())
    result = to_json_schema(array_additional_field)
    assert isinstance(result["additionalItems"], dict)

    # Test Object with property_names
    object_prop_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_prop_names)
    assert "propertyNames" in result

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert result["required"] == ["name"]


# LLM-generated content at query #38
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    field = String(allow_null=False, min_length=2, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 2
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test with String field allowing null
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

    # Test with Integer field
    field = Integer(allow_null=False, minimum=1, maximum=100, exclusive_minimum=0, exclusive_maximum=101)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 1
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 101

    # Test with Float field
    field = Float(allow_null=False, minimum=0.5, maximum=99.5, multiple_of=0.1)
    result = to_json_schema(field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.5
    assert result["maximum"] == 99.5
    assert result["multipleOf"] == 0.1

    # Test with Boolean field
    field = Boolean(allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    field = Array(allow_null=False, items=String(), min_items=1, max_items=5, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test with Array field with tuple items
    field = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert len(result["items"]) == 2
    assert result["items"][0]["type"] == "string"
    assert result["items"][1]["type"] == "integer"

    # Test with Array field with additional_items as bool
    field = Array(allow_null=False, items=String(), additional_items=False)
    result = to_json_schema(field)
    assert result["additionalItems"] is False

    # Test with Array field with additional_items as Field
    field = Array(allow_null=False, items=String(), additional_items=Integer())
    result = to_json_schema(field)
    assert result["additionalItems"]["type"] == "integer"

    # Test with Object field
    field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10

    # Test with Object field with pattern_properties
    field = Object(
        allow_null=False,
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(field)
    assert "patternProperties" in result
    assert result["patternProperties"]["^S_"]["type"] == "string"
    assert result["patternProperties"]["^I_"]["type"] == "integer"

    # Test with Object field with additional_properties as bool
    field = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(field)
    assert result["additionalProperties"] is False

    # Test with Object field with additional_properties as Field
    field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(field)
    assert result["additionalProperties"]["type"] == "string"

    # Test with Object field with property_names
    field = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(field)
    assert "propertyNames" in result
    assert result["propertyNames"]["type"] == "string"

    # Test with Schema field
    schema = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]

    # Test with Choice field
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    field = Const(const="constant_value")
    result = to_json_schema(field)
    assert result["const"] == "constant_value"

    # Test with Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

    # Test with OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with IfThenElse field
    field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    assert result["if"]["type"] == "string"
    assert result["then"]["type"] == "integer"
    assert result["else"]["type"] == "boolean"

    # Test with IfThenElse field without else clause
    field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with Not field
    field = Not(negated=String())
    result = to_json_schema(field)
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test with Reference field
    definitions = Definitions()
    definitions["TestSchema"] = Schema(fields={"id": Integer()})
    field = Reference(to="TestSchema", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/TestSchema"
    assert "components" in result
    assert "schemas" in result["components"]

    #


# LLM-generated content at query #39
#--------------------------

```python
def test_to_json_schema():
    # Test with Any type
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch type
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test with String field allowing null
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

    # Test with String field with allow_blank
    field = String(allow_null=False, allow_blank=True)
    result = to_json_schema(field)
    assert "minLength" not in result

    # Test with Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test with Integer field allowing null
    field = Integer(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["integer", "null"]

    # Test with Float field
    field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test with Float field with exclusive bounds
    field = Float(allow_null=False, exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(field)
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0

    # Test with Boolean field
    field = Boolean(allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    field = Array(allow_null=False, min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True

    # Test with Array field with items
    field = Array(allow_null=False, items=String())
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"

    # Test with Array field with tuple items
    field = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test with Array field with additional_items as bool
    field = Array(allow_null=False, additional_items=False)
    result = to_json_schema(field)
    assert result["additionalItems"] is False

    # Test with Array field with additional_items as Field
    field = Array(allow_null=False, additional_items=String())
    result = to_json_schema(field)
    assert result["additionalItems"]["type"] == "string"

    # Test with Object field
    field = Object(allow_null=False, min_properties=1, max_properties=10)
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10

    # Test with Object field with properties
    field = Object(allow_null=False, properties={"name": String(), "age": Integer()})
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]

    # Test with Object field with pattern_properties
    field = Object(allow_null=False, pattern_properties={"^S_": String()})
    result = to_json_schema(field)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]

    # Test with Object field with additional_properties as bool
    field = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(field)
    assert result["additionalProperties"] is False

    # Test with Object field with additional_properties as Field
    field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(field)
    assert result["additionalProperties"]["type"] == "string"

    # Test with Object field with property_names
    field = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(field)
    assert "propertyNames" in result

    # Test with Object field with required properties
    field = Object(allow_null=False, required=["name", "age"])
    result = to_json_schema(field)
    assert result["required"] == ["name", "age"]

    # Test with Choice field
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    field = Const(const="constant_value")
    result = to_json_schema(field)
    assert result["const"] == "constant_value"

    # Test with Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with IfThenElse field
    field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without else clause
    field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with Not field
    field = Not(negated=String())
    result = to_json_schema(field)
    assert "not" in result

    # Test with Reference field
    definitions = Definitions({"TestRef": String()})
    field = Reference(to="TestRef", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/TestRef"
    assert "components" in result
    assert "schemas" in result["components"]

    # Test with Definitions object
    definitions


# LLM-generated content at query #40
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    
    # Test String field with null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String with constraints
    string_constrained = String(min_length=5, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_constrained)
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test Integer field
    int_field = Integer(allow_null=False)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    
    # Test Integer with constraints
    int_constrained = Integer(minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(int_constrained)
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test Float with exclusive bounds
    float_constrained = Float(exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_constrained)
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0
    
    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test Boolean with null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(items=String(), allow_null=False)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"
    
    # Test Array with constraints
    array_constrained = Array(min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_constrained)
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    
    # Test Array with tuple items
    array_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Object field
    obj_field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    
    # Test Object with required fields
    obj_required = Object(properties={"name": String()}, required=["name"])
    result = to_json_schema(obj_required)
    assert result["required"] == ["name"]
    
    # Test Object with pattern properties
    obj_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    
    # Test Object with additional properties
    obj_additional = Object(additional_properties=String())
    result = to_json_schema(obj_additional)
    assert result["additionalProperties"]["type"] == "string"
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), String()])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without else
    if_then = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(if_then)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with definitions
    definitions = Definitions({"TestSchema": String()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    
    # Test Reference field
    ref_field = Reference(to="TestSchema", definitions=Definitions({"TestSchema": String()}))
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/TestSchema"
    
    # Test String with no blank allowed
    string_no_blank = String(allow_blank=False)
    result = to_json_schema(string_no_blank)
    assert result["minLength"] == 1
    
    # Test Array with additional items as bool
    array_additional_bool = Array(additional_items=False)
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] == False
    
    # Test Array with additional items as field
    array_additional_field = Array(additional_items=String())
    result = to_json_schema(array_additional_field)
    assert result["additionalItems"]["type"] == "string"
    
    # Test Object with property names
    obj_prop_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_prop_names)
    assert "propertyNames" in result
    
    # Test Object with min/max properties
    obj_min_max = Object(min_properties=1, max_properties=5)
    result = to_json_schema(obj_min_max)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    
    # Test Schema field
    schema = Schema(fields={"name


# LLM-generated content at query #41
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    result = to_json_schema(Any())
    assert result is True

    # Test NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True, min_length=None)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result

    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=1, exclusive_maximum=99, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 1
    assert result["exclusiveMaximum"] == 99
    assert result["multipleOf"] == 5

    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test Boolean field with allow_null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with items
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=5, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test Array field with list of items
    array_field_list = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_list)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array field with additionalItems
    array_field_additional = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"] is False

    # Test Object field with properties
    obj_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    assert result["required"] == ["name"]

    # Test Object field with pattern_properties
    obj_field_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(obj_field_pattern)
    assert "patternProperties" in result

    # Test Object field with additional_properties
    obj_field_additional = Object(allow_null=False, additional_properties=True)
    result = to_json_schema(obj_field_additional)
    assert result["additionalProperties"] is True

    # Test Object field with property_names
    obj_field_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_field_names)
    assert "propertyNames" in result

    # Test Choice field
    choice_field = Choice(choices=[("a", 1), ("b", 2)])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=5)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse field without else
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with Definitions
    definitions = Definitions()
    definitions["CustomString"] = String(min_length=5)
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "CustomString" in result["components"]["schemas"]

    # Test Reference field
    ref_field = Reference(to="CustomType", definitions=Definitions())
    result = to_json_schema(ref_field)
    assert "$ref" in result

    # Test Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]

    # Test with default values
    string_with_default = String(default="test_default")
    result = to_json_schema(string_with_default)
    assert result.get("default") == "test_default"

    # Test nested structures
    nested_obj = Object(
        properties={
            "user": Object(properties={"name": String(), "email": String()}),
            "items": Array(items=Object(properties={"id": Integer(), "value": String()}))
        }
    )
    result = to_json_schema


# LLM-generated content at query #42
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) is True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) is False

    # Test String field
    string_field = String(allow_null=False)
    result = to_json_schema(string_field)
    assert result["type"] == "string"

    # Test String field with null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test String with constraints
    string_field_constrained = String(min_length=2, max_length=10, allow_blank=False)
    result = to_json_schema(string_field_constrained)
    assert result["minLength"] == 2
    assert result["maxLength"] == 10

    # Test Integer field
    int_field = Integer(allow_null=False)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"

    # Test Integer with constraints
    int_field_constrained = Integer(minimum=0, maximum=100)
    result = to_json_schema(int_field_constrained)
    assert result["minimum"] == 0
    assert result["maximum"] == 100

    # Test Float field
    float_field = Float(allow_null=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"

    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test Boolean with null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test Array field
    array_field = Array(items=String(), allow_null=False)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert "items" in result

    # Test Array with min/max items
    array_field_constrained = Array(min_items=1, max_items=5)
    result = to_json_schema(array_field_constrained)
    assert result["minItems"] == 1
    assert result["maxItems"] == 5

    # Test Object field
    obj_field = Object(properties={"name": String()}, allow_null=False)
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]

    # Test Object with required fields
    obj_field_required = Object(
        properties={"name": String()},
        required=["name"]
    )
    result = to_json_schema(obj_field_required)
    assert result["required"] == ["name"]

    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Choice(choices=[("a", "A")])])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse without else clause
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test Reference field
    definitions_obj = Definitions()
    definitions_obj["MyType"] = String()
    ref_field = Reference(to="MyType", definitions=definitions_obj)
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/MyType"

    # Test Definitions
    definitions = Definitions()
    definitions["StringType"] = String()
    definitions["IntType"] = Integer()
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]

    # Test nested Object
    nested_obj = Object(
        properties={
            "user": Object(properties={"name": String(), "age": Integer()}),
            "tags": Array(items=String())
        }
    )
    result = to_json_schema(nested_obj)
    assert result["type"] == "object"
    assert "properties" in result
    assert "user" in result["properties"]
    assert "tags" in result["properties"]

    # Test Array with unique items
    array_unique = Array(items=String(), unique_items=True)
    result = to_json_schema(array_unique)
    assert result["uniqueItems"] is True

    # Test Object with pattern properties
    obj_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]

    # Test Object with additional properties
    obj_additional = Object(additional_properties=String())
    result = to_json_schema(obj_additional)
    assert "additionalProperties" in result

    # Test String with pattern
    string_pattern = String(pattern="^[a-z]+$")
    result = to_json_schema(string_pattern)
    assert "pattern" in result
    assert result["pattern"] == "^[a-z]+$"

    # Test String with format
    string_format = String(format="email")
    result = to_json_schema(string_format)
    assert result["format"] == "email"

    # Test Float with exclusive bounds
    float_exclusive = Float(exclusive_minimum=0.0, exclusive_maximum=100.0)
    result = to_json_schema(float_exclusive)
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 100.0

    # Test Integer with multiple_of
    int_multiple = Integer(multiple_of=5)
    result = to_json_schema(int_multiple)
    assert result["multipleOf"] == 5


# LLM-generated content at query #43
#--------------------------

```python
def test_to_json_schema():
    # Test Any field returns True
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch field returns False
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String()
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert "default" not in result or result["default"] == NO_DEFAULT
    
    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String field with min_length and max_length
    string_field_constrained = String(min_length=2, max_length=10)
    result = to_json_schema(string_field_constrained)
    assert result["minLength"] == 2
    assert result["maxLength"] == 10
    
    # Test String field with pattern
    import re
    string_field_pattern = String(pattern=re.compile(r"^\d+$"))
    result = to_json_schema(string_field_pattern)
    assert result["pattern"] == r"^\d+$"
    
    # Test String field with format
    string_field_format = String(format="email")
    result = to_json_schema(string_field_format)
    assert result["format"] == "email"
    
    # Test Integer field
    integer_field = Integer()
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    
    # Test Integer field with constraints
    integer_field_constrained = Integer(minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(integer_field_constrained)
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float()
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test Boolean field
    boolean_field = Boolean()
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Boolean field with allow_null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(items=String())
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"
    
    # Test Array field with min/max items
    array_field_constrained = Array(items=Integer(), min_items=1, max_items=10)
    result = to_json_schema(array_field_constrained)
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    
    # Test Array field with unique items
    array_field_unique = Array(items=String(), unique_items=True)
    result = to_json_schema(array_field_unique)
    assert result["uniqueItems"] is True
    
    # Test Object field
    object_field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    
    # Test Object field with required properties
    object_field_required = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(object_field_required)
    assert result["required"] == ["name"]
    
    # Test Object field with pattern properties
    object_field_pattern = Object(
        pattern_properties={"^S_": String()}
    )
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=1)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=Choice(choices=[("a", "A"), ("b", "B")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test Reference field with definitions
    definitions = Definitions()
    reference_field = Reference(to="CustomSchema", definitions=definitions)
    result = to_json_schema(reference_field)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/CustomSchema"
    
    # Test with Definitions object
    definitions = Definitions()
    definitions["MyString"] = String()
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    
    # Test String field with allow_blank=False
    string_field_no_blank = String(allow_blank=False)
    result = to_json_schema(string_field_no_blank)
    assert result["minLength"] == 1
    
    # Test Array with tuple items
    array_tuple_items = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple_items)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Array with additional_items as bool
    array_additional_bool = Array(items=String(), additional_items=False)
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] is False
    
    # Test Array with additional_items as Field
    array_additional_field = Array(items=String(), additional_items=Integer())
    result = to_json_schema(array_additional_field)
    assert isinstance(result["additionalItems"], dict)
    
    # Test Object with additional_properties as bool
    object_additional_bool = Object(additional_properties=False)
    result = to_json_schema(object_additional_bool)
    assert result["additionalProperties"] is False
    
    # Test Object with additional_properties as Field
    object_additional_field = Object(additional_properties=String())
    result = to_json_schema(object_additional_field)
    assert isinstance(result["additionalProperties"], dict)
    
    # Test Object with property_names
    object_property_names


# LLM-generated content at query #44
#--------------------------

```python
def test_to_json_schema():
    # Test Any type
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch type
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    
    # Test String field with null
    string_field_nullable = String(allow_null=True, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None)
    result = to_json_schema(string_field_nullable)
    assert result["type"] == ["string", "null"]
    
    # Test String with constraints
    string_constrained = String(allow_null=False, allow_blank=False, min_length=2, max_length=10, format="email", pattern="^[a-z]+$")
    result = to_json_schema(string_constrained)
    assert result["type"] == "string"
    assert result["minLength"] == 2
    assert result["maxLength"] == 10
    assert result["format"] == "email"
    assert result["pattern"] == "^[a-z]+$"
    
    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    
    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    
    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Boolean with null
    boolean_nullable = Boolean(allow_null=True)
    result = to_json_schema(boolean_nullable)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, items=String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None), min_items=0, max_items=None, unique_items=False, additional_items=None)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert "items" in result
    
    # Test Array with min/max items
    array_constrained = Array(allow_null=False, items=Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None), min_items=1, max_items=5, unique_items=True, additional_items=None)
    result = to_json_schema(array_constrained)
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] == True
    
    # Test Object field
    object_field = Object(allow_null=False, properties={"name": String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None)}, pattern_properties=None, additional_properties=None, property_names=None, min_properties=None, max_properties=None, required=["name"])
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None), Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None), Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None), Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None))
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=Boolean(allow_null=False),
        then_clause=String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None),
        else_clause=Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test with definitions
    definitions = Definitions()
    definitions["StringType"] = String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None)
    result = to_json_schema(definitions)
    assert "StringType" in result
    
    # Test Reference field with definitions
    ref_definitions = Definitions()
    ref_definitions["User"] = Object(allow_null=False, properties={"name": String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None)}, pattern_properties=None, additional_properties=None, property_names=None, min_properties=None, max_properties=None, required=None)
    reference_field = Reference(to="User", definitions=ref_definitions)
    result = to_json_schema(reference_field)
    assert "$ref" in result
    assert "components" in result
    assert "schemas" in result["components"]


# LLM-generated content at query #45
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    result = to_json_schema(Any())
    assert result is True

    # Test NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result

    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=1, exclusive_maximum=99, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 1
    assert result["exclusiveMaximum"] == 99
    assert result["multipleOf"] == 5

    # Test Integer field with allow_null
    int_field_null = Integer(allow_null=True)
    result = to_json_schema(int_field_null)
    assert result["type"] == ["integer", "null"]

    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=100.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 100.0

    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test Boolean field with allow_null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with items
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test Array field with list of items
    array_field_list = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_list)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array field with additional_items
    array_field_additional = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"] is False

    # Test Object field
    obj_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        pattern_properties={"^S_": String()},
        additional_properties=False,
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert "^S_" in result["patternProperties"]
    assert result["additionalProperties"] is False
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]

    # Test Object field with allow_null
    obj_field_null = Object(allow_null=True)
    result = to_json_schema(obj_field_null)
    assert result["type"] == ["object", "null"]

    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=5)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse without else clause
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with default value
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"

    # Test Definitions
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]

    # Test Reference field
    definitions_obj = Definitions({"User": Object()})
    ref_field = Reference(to="User", definitions=definitions_obj)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/User"
    assert "components" in result

    # Test Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]


# LLM-generated content at query #46
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    assert to_json_schema(Any()) == True
    
    # Test with NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test with String field
    string_field = String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None, default=NO_DEFAULT, coerce_types=False)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert "default" not in result or result.get("default") == NO_DEFAULT
    
    # Test with String field allowing null
    string_field_null = String(allow_null=True, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None, default=NO_DEFAULT, coerce_types=False)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test with String field with constraints
    string_field_constrained = String(allow_null=False, allow_blank=False, min_length=2, max_length=10, format="email", pattern=None, default=NO_DEFAULT, coerce_types=False)
    result = to_json_schema(string_field_constrained)
    assert result["type"] == "string"
    assert result["minLength"] == 2
    assert result["maxLength"] == 10
    assert result["format"] == "email"
    
    # Test with Integer field
    integer_field = Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None, default=NO_DEFAULT, coerce_types=False)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    
    # Test with Integer field with constraints
    integer_field_constrained = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=None, exclusive_maximum=None, multiple_of=5, default=NO_DEFAULT, coerce_types=False)
    result = to_json_schema(integer_field_constrained)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test with Float field
    float_field = Float(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None, default=NO_DEFAULT, coerce_types=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test with Boolean field
    boolean_field = Boolean(allow_null=False, default=NO_DEFAULT, coerce_types=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test with Boolean field allowing null
    boolean_field_null = Boolean(allow_null=True, default=NO_DEFAULT, coerce_types=False)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test with Array field
    array_field = Array(allow_null=False, min_items=0, max_items=None, additional_items=True, items=None, unique_items=False, default=NO_DEFAULT)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    
    # Test with Array field with items
    string_item = String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None, default=NO_DEFAULT, coerce_types=False)
    array_field_items = Array(allow_null=False, min_items=1, max_items=10, additional_items=True, items=string_item, unique_items=True, default=NO_DEFAULT)
    result = to_json_schema(array_field_items)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    assert result["items"]["type"] == "string"
    
    # Test with Object field
    object_field = Object(allow_null=False, properties=None, pattern_properties=None, additional_properties=None, property_names=None, min_properties=None, max_properties=None, required=None, default=NO_DEFAULT)
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    
    # Test with Object field with properties
    properties = {
        "name": String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None, default=NO_DEFAULT, coerce_types=False),
        "age": Integer(allow_null=False, minimum=0, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None, default=NO_DEFAULT, coerce_types=False)
    }
    object_field_props = Object(allow_null=False, properties=properties, pattern_properties=None, additional_properties=None, property_names=None, min_properties=None, max_properties=None, required=["name"], default=NO_DEFAULT)
    result = to_json_schema(object_field_props)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")], default=NO_DEFAULT)
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test with Const field
    const_field = Const(const="constant_value", default=NO_DEFAULT)
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"
    
    # Test with Union field
    union_field = Union(any_of=[
        String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None, default=NO_DEFAULT, coerce_types=False),
        Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None, default=NO_DEFAULT, coerce_types=False)
    ], allow_null=False, default=NO_DEFAULT)
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test with OneOf field
    one_of_field = OneOf(one_of=[
        String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None, default=NO_DEFAULT, coerce_types=False),
        Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None, default=NO_DEFAULT, coerce_types=False)
    ], default=NO_DEFAULT)
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test with AllOf field
    all_of_field = AllOf(all_of=[
        String(allow_null=False, allow_blank=True, min_length=None, max_length=None, format=None, pattern=None, default=NO_DEFAULT, coerce_types=False),
        String(allow_null=False, allow_blank=False, min_length=1, max_length=None, format=None, pattern=None, default=NO_DEFAULT, coerce_types=False)
    ], default=NO_DEFAULT)
    result = to_json_schema


# LLM-generated content at query #47
#--------------------------

```python
def test_to_json_schema():
    # Test with Any type
    assert to_json_schema(Any()) == True
    
    # Test with NeverMatch type
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    result = to_json_schema(String())
    assert result["type"] == "string"
    assert result["default"] == NO_DEFAULT
    
    # Test String with allow_null
    result = to_json_schema(String(allow_null=True))
    assert result["type"] == ["string", "null"]
    
    # Test String with constraints
    result = to_json_schema(String(min_length=5, max_length=10, allow_blank=False))
    assert result["type"] == "string"
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    
    # Test String with pattern
    result = to_json_schema(String(pattern=r"^[a-z]+$"))
    assert result["pattern"] == "^[a-z]+$"
    
    # Test String with format
    result = to_json_schema(String(format="email"))
    assert result["format"] == "email"
    
    # Test Integer field
    result = to_json_schema(Integer())
    assert result["type"] == "integer"
    
    # Test Integer with constraints
    result = to_json_schema(Integer(minimum=0, maximum=100))
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    
    # Test Float field
    result = to_json_schema(Float())
    assert result["type"] == "number"
    
    # Test Float with exclusive bounds
    result = to_json_schema(Float(exclusive_minimum=0.0, exclusive_maximum=1.0))
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0
    
    # Test Boolean field
    result = to_json_schema(Boolean())
    assert result["type"] == "boolean"
    
    # Test Boolean with allow_null
    result = to_json_schema(Boolean(allow_null=True))
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    result = to_json_schema(Array())
    assert result["type"] == "array"
    
    # Test Array with items
    result = to_json_schema(Array(items=String()))
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"
    
    # Test Array with min/max items
    result = to_json_schema(Array(min_items=1, max_items=10))
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    
    # Test Array with unique_items
    result = to_json_schema(Array(unique_items=True))
    assert result["uniqueItems"] == True
    
    # Test Object field
    result = to_json_schema(Object())
    assert result["type"] == "object"
    
    # Test Object with properties
    result = to_json_schema(Object(properties={"name": String(), "age": Integer()}))
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    
    # Test Object with required fields
    result = to_json_schema(Object(required=["name"]))
    assert result["required"] == ["name"]
    
    # Test Object with min/max properties
    result = to_json_schema(Object(min_properties=1, max_properties=5))
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    
    # Test Choice field
    result = to_json_schema(Choice(choices=[("a", "A"), ("b", "B")]))
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    result = to_json_schema(Const(const="fixed_value"))
    assert result["const"] == "fixed_value"
    
    # Test Union field
    result = to_json_schema(Union(any_of=[String(), Integer()]))
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    result = to_json_schema(OneOf(one_of=[String(), Integer()]))
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    result = to_json_schema(AllOf(all_of=[String(), String(min_length=5)]))
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    result = to_json_schema(Not(negated=String()))
    assert "not" in result
    assert result["not"]["type"] == "string"
    
    # Test IfThenElse field
    result = to_json_schema(IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    ))
    assert "if" in result
    assert "then" in result
    assert "else" in result
    assert result["if"]["type"] == "string"
    assert result["then"]["type"] == "integer"
    assert result["else"]["type"] == "boolean"
    
    # Test IfThenElse with only if clause
    result = to_json_schema(IfThenElse(if_clause=String()))
    assert "if" in result
    assert "then" not in result
    assert "else" not in result
    
    # Test Reference field
    definitions = Definitions()
    result = to_json_schema(Reference(to="User", definitions=definitions))
    assert result["$ref"] == "#/components/schemas/User"
    
    # Test with Definitions
    defs = Definitions({"User": String()})
    result = to_json_schema(defs)
    assert "User" in result
    
    # Test root level with definitions
    result = to_json_schema(
        Object(properties={"name": String()}),
        _definitions=None
    )
    assert result["type"] == "object"
    assert "properties" in result
    
    # Test Array with list of items
    result = to_json_schema(Array(items=[String(), Integer()]))
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Array with additional_items as bool
    result = to_json_schema(Array(additional_items=False))
    assert result["additionalItems"] == False
    
    # Test Object with pattern_properties
    result = to_json_schema(Object(pattern_properties={"^S_": String()}))
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]
    
    # Test Object with additional_properties as bool
    result = to_json_schema(Object(additional_properties=False))
    assert result["additionalProperties"] == False
    
    # Test Object with property_names
    result = to_json_schema(Object(property_names=String(pattern="^[a-z_]+$")))
    assert "propertyNames" in result
    assert result["propertyNames"]["pattern"] == "^[a-z_]+$"
    
    # Test String with no blank allowed and no min_length
    result = to_json_schema(String(allow_blank=False))
    assert result["minLength"] == 1
    
    # Test multiple_of constraint
    result = to_json_schema(Integer(multiple_of=5))
    assert result["multipleOf"] == 5
    
    # Test error on unsupported field type
    class CustomField(Field):
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(CustomField())


# LLM-generated content at query #48
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    result = to_json_schema(Any())
    assert result is True

    # Test NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test String field with allow_blank
    string_field_blank = String(allow_blank=True, min_length=None)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result

    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=1, exclusive_maximum=99, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 1
    assert result["exclusiveMaximum"] == 99
    assert result["multipleOf"] == 5

    # Test Integer field with allow_null
    integer_field_null = Integer(allow_null=True)
    result = to_json_schema(integer_field_null)
    assert result["type"] == ["integer", "null"]

    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test Boolean field with allow_null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with items
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test Array field with tuple items
    array_field_tuple = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array field with additional_items as bool
    array_field_bool = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_field_bool)
    assert result["additionalItems"] is False

    # Test Array field with additional_items as Field
    array_field_field = Array(allow_null=False, additional_items=String())
    result = to_json_schema(array_field_field)
    assert isinstance(result["additionalItems"], dict)

    # Test Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]

    # Test Object field with pattern_properties
    object_field_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result

    # Test Object field with additional_properties as bool
    object_field_bool = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(object_field_bool)
    assert result["additionalProperties"] is False

    # Test Object field with additional_properties as Field
    object_field_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(object_field_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test Object field with property_names
    object_field_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_field_names)
    assert "propertyNames" in result

    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Object()])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse field without then/else
    if_then_else_minimal = IfThenElse(if_clause=String())
    result = to_json_schema(if_then_else_minimal)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result

    # Test with Definitions
    definitions = Definitions()
    definitions["StringType"] = String()
    definitions["IntegerType"] = Integer()
    result = to_json_schema(definitions)
    assert "StringType" in result
    assert "IntegerType" in result
    assert result["StringType"]["type"] == "string"
    assert result["IntegerType"]["type"] == "integer"

    # Test Reference field
    definitions_obj =


# LLM-generated content at query #49
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False)
    result = to_json_schema(string_field)
    assert result["type"] == "string"

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test String with min_length
    string_field_min = String(min_length=5)
    result = to_json_schema(string_field_min)
    assert result["minLength"] == 5

    # Test String with max_length
    string_field_max = String(max_length=10)
    result = to_json_schema(string_field_max)
    assert result["maxLength"] == 10

    # Test String with pattern
    import re
    string_field_pattern = String(pattern=re.compile(r"^[a-z]+$"))
    result = to_json_schema(string_field_pattern)
    assert result["pattern"] == "^[a-z]+$"

    # Test String with format
    string_field_format = String(format="email")
    result = to_json_schema(string_field_format)
    assert result["format"] == "email"

    # Test with Integer field
    int_field = Integer(allow_null=False)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"

    # Test Integer with null
    int_field_null = Integer(allow_null=True)
    result = to_json_schema(int_field_null)
    assert result["type"] == ["integer", "null"]

    # Test Integer with minimum
    int_field_min = Integer(minimum=0)
    result = to_json_schema(int_field_min)
    assert result["minimum"] == 0

    # Test Integer with maximum
    int_field_max = Integer(maximum=100)
    result = to_json_schema(int_field_max)
    assert result["maximum"] == 100

    # Test Integer with exclusive bounds
    int_field_excl = Integer(exclusive_minimum=5, exclusive_maximum=95)
    result = to_json_schema(int_field_excl)
    assert result["exclusiveMinimum"] == 5
    assert result["exclusiveMaximum"] == 95

    # Test Integer with multiple_of
    int_field_mult = Integer(multiple_of=5)
    result = to_json_schema(int_field_mult)
    assert result["multipleOf"] == 5

    # Test with Float field
    float_field = Float(allow_null=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"

    # Test Float with null
    float_field_null = Float(allow_null=True)
    result = to_json_schema(float_field_null)
    assert result["type"] == ["number", "null"]

    # Test with Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test Boolean with null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(items=String(), allow_null=False)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"

    # Test Array with null
    array_field_null = Array(items=String(), allow_null=True)
    result = to_json_schema(array_field_null)
    assert result["type"] == ["array", "null"]

    # Test Array with min/max items
    array_field_bounds = Array(items=String(), min_items=1, max_items=10)
    result = to_json_schema(array_field_bounds)
    assert result["minItems"] == 1
    assert result["maxItems"] == 10

    # Test Array with unique items
    array_field_unique = Array(items=String(), unique_items=True)
    result = to_json_schema(array_field_unique)
    assert result["uniqueItems"] is True

    # Test Array with tuple items
    array_field_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_field_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array with additional items as bool
    array_field_add_bool = Array(items=String(), additional_items=False)
    result = to_json_schema(array_field_add_bool)
    assert result["additionalItems"] is False

    # Test Array with additional items as Field
    array_field_add_field = Array(items=String(), additional_items=Integer())
    result = to_json_schema(array_field_add_field)
    assert isinstance(result["additionalItems"], dict)
    assert result["additionalItems"]["type"] == "integer"

    # Test with Object field
    object_field = Object(properties={"name": String()}, allow_null=False)
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"

    # Test Object with null
    object_field_null = Object(properties={"name": String()}, allow_null=True)
    result = to_json_schema(object_field_null)
    assert result["type"] == ["object", "null"]

    # Test Object with pattern properties
    object_field_pattern = Object(
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result

    # Test Object with additional properties as bool
    object_field_add_bool = Object(properties={"name": String()}, additional_properties=False)
    result = to_json_schema(object_field_add_bool)
    assert result["additionalProperties"] is False

    # Test Object with additional properties as Field
    object_field_add_field = Object(
        properties={"name": String()}, additional_properties=String()
    )
    result = to_json_schema(object_field_add_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test Object with property names
    object_field_prop_names = Object(
        properties={"name": String()},
        property_names=String(pattern=re.compile(r"^[a-z]+$"))
    )
    result = to_json_schema(object_field_prop_names)
    assert "propertyNames" in result

    # Test Object with min/max properties
    object_field_bounds = Object(
        properties={"name": String()},
        min_properties=1,
        max_properties=5
    )
    result = to_json_schema(object_field_bounds)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5

    # Test Object with required fields
    object_field_required = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(object_field_required)
    assert result["required"] == ["name"]

    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const


# LLM-generated content at query #50
#--------------------------

```python
def test_to_json_schema():
    # Test Any type
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch type
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String()
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    
    # Test String with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String with constraints
    string_constrained = String(min_length=5, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_constrained)
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test Integer field
    integer_field = Integer()
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    
    # Test Integer with constraints
    integer_constrained = Integer(minimum=0, maximum=100)
    result = to_json_schema(integer_constrained)
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    
    # Test Float field
    float_field = Float()
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test Boolean field
    boolean_field = Boolean()
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Array field
    array_field = Array(items=String())
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert "items" in result
    
    # Test Array with constraints
    array_constrained = Array(items=Integer(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_constrained)
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    
    # Test Object field
    object_field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    
    # Test Object with required fields
    object_required = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(object_required)
    assert result["required"] == ["name"]
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Object(properties={"key": String()})])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test Reference field with definitions
    definitions = Definitions({"MyType": String()})
    ref_field = Reference(to="MyType", definitions=definitions)
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert "components" in result
    assert "schemas" in result["components"]
    
    # Test with default values
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"
    
    # Test Array with tuple items
    array_tuple_items = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple_items)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Object with pattern_properties
    object_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(object_pattern)
    assert "patternProperties" in result
    
    # Test Object with additional_properties as Field
    object_additional = Object(additional_properties=String())
    result = to_json_schema(object_additional)
    assert "additionalProperties" in result
    
    # Test String without allow_blank
    string_no_blank = String(allow_blank=False)
    result = to_json_schema(string_no_blank)
    assert result["minLength"] == 1
    
    # Test Float with exclusive bounds
    float_exclusive = Float(exclusive_minimum=0, exclusive_maximum=100)
    result = to_json_schema(float_exclusive)
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100
    
    # Test invalid field type raises ValueError
    class InvalidField(Field):
        pass
    
    invalid_field = InvalidField()
    try:
        to_json_schema(invalid_field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)
    
    # Test Definitions conversion
    defs = Definitions({"Type1": String(), "Type2": Integer()})
    result = to_json_schema(defs)
    assert "Type1" in result
    assert "Type2" in result
    
    # Test Array with additional_items as bool
    array_additional_bool = Array(items=String(), additional_items=False)
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] == False
    
    # Test Array with additional_items as Field
    array_additional_field = Array(items=String(), additional_items=Integer())
    result = to_json_schema(array_additional_field)
    assert "additionalItems" in result
    
    # Test Object with property_names
    object_prop_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_prop_names)
    assert "propertyNames" in result
    
    # Test Object with min/max properties
    object_size = Object(min_properties=1, max_properties=5)
    result = to_json_schema(object_size)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5


# LLM-generated content at query #51
#--------------------------

```python
def test_to_json_schema():
    # Test Any type returns True
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch type returns False
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String(allow_null=False, min_length=None, max_length=10, pattern=None, format=None)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["maxLength"] == 10
    
    # Test String field with allow_null
    string_field_null = String(allow_null=True, min_length=None, max_length=None, pattern=None, format=None)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String field with min_length
    string_field_min = String(allow_null=False, min_length=5, max_length=None, pattern=None, format=None)
    result = to_json_schema(string_field_min)
    assert result["minLength"] == 5
    
    # Test String field with blank not allowed
    string_field_no_blank = String(allow_null=False, allow_blank=False, min_length=None, max_length=None, pattern=None, format=None)
    result = to_json_schema(string_field_no_blank)
    assert result["minLength"] == 1
    
    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    
    # Test Integer field with allow_null
    integer_field_null = Integer(allow_null=True, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)
    result = to_json_schema(integer_field_null)
    assert result["type"] == ["integer", "null"]
    
    # Test Float field
    float_field = Float(allow_null=False, minimum=1.5, maximum=9.5, exclusive_minimum=None, exclusive_maximum=None, multiple_of=0.5)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 1.5
    assert result["maximum"] == 9.5
    assert result["multipleOf"] == 0.5
    
    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Boolean field with allow_null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=False)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert "items" in result
    
    # Test Array field with allow_null
    array_field_null = Array(allow_null=True, items=None, min_items=None, max_items=None, unique_items=False)
    result = to_json_schema(array_field_null)
    assert result["type"] == ["array", "null"]
    
    # Test Array with unique_items
    array_field_unique = Array(allow_null=False, items=None, min_items=None, max_items=None, unique_items=True)
    result = to_json_schema(array_field_unique)
    assert result["uniqueItems"] is True
    
    # Test Object field
    object_field = Object(allow_null=False, properties={"name": String()}, min_properties=None, max_properties=None, required=None)
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    
    # Test Object field with allow_null
    object_field_null = Object(allow_null=True, properties=None, min_properties=None, max_properties=None, required=None)
    result = to_json_schema(object_field_null)
    assert result["type"] == ["object", "null"]
    
    # Test Object with required fields
    object_field_required = Object(allow_null=False, properties=None, min_properties=None, max_properties=None, required=["id", "name"])
    result = to_json_schema(object_field_required)
    assert result["required"] == ["id", "name"]
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Object()])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse with only if clause
    if_only_field = IfThenElse(if_clause=String(), then_clause=None, else_clause=None)
    result = to_json_schema(if_only_field)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result
    
    # Test with Definitions
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(definitions)
    assert "User" in result
    
    # Test Reference field
    definitions_dict = {"User": Object(properties={"name": String()})}
    definitions_obj = Definitions(definitions_dict)
    ref_field = Reference(to="User", definitions=definitions_obj)
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert "components" in result
    
    # Test invalid field type raises ValueError
    class CustomField(Field):
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(CustomField())


# LLM-generated content at query #52
#--------------------------

```python
def test_to_json_schema():
    # Test with Any() field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch() field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test with Float field
    float_field = Float(allow_null=False, exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0

    # Test with Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test with Array field with tuple items
    array_field_tuple = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test with Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]

    # Test with Object field with pattern properties
    object_field_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String()},
        additional_properties=False
    )
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result
    assert result["additionalProperties"] is False

    # Test with Choice field
    choice_field = Choice(choices=[("a", 1), ("b", 2)])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without else clause
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]

    # Test with Definitions
    definitions = Definitions({"StringType": String(), "IntType": Integer()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "StringType" in result["components"]["schemas"]
    assert "IntType" in result["components"]["schemas"]

    # Test with Reference field
    definitions_dict = Definitions({"MyString": String()})
    ref_field = Reference(to="MyString", definitions=definitions_dict)
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/MyString"
    assert "components" in result

    # Test with Array field with additional_items as bool
    array_additional_items = Array(allow_null=False, items=String(), additional_items=False)
    result = to_json_schema(array_additional_items)
    assert result["additionalItems"] is False

    # Test with Array field with additional_items as Field
    array_additional_items_field = Array(
        allow_null=False,
        items=String(),
        additional_items=Integer()
    )
    result = to_json_schema(array_additional_items_field)
    assert "additionalItems" in result
    assert isinstance(result["additionalItems"], dict)

    # Test with Object field with property_names
    object_property_names = Object(
        allow_null=False,
        property_names=String(pattern="^[a-z]+$")
    )
    result = to_json_schema(object_property_names)
    assert "propertyNames" in result

    # Test with Object field with min/max properties
    object_min_max = Object(
        allow_null=False,
        min_properties=1,
        max_properties=5
    )
    result = to_json_schema(object_min_max)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5

    # Test with String field with format
    string_format = String(allow_null=False, format="email")
    result = to_json


# LLM-generated content at query #53
#--------------------------

```python
def test_to_json_schema():
    # Test Any type returns True
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch type returns False
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"
    
    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result
    
    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=5, exclusive_maximum=95, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5
    assert result["exclusiveMaximum"] == 95
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test Boolean field with allow_null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"
    
    # Test Array with additional_items as bool
    array_field_bool = Array(allow_null=False, items=String(), additional_items=False)
    result = to_json_schema(array_field_bool)
    assert result["additionalItems"] is False
    
    # Test Array with tuple items
    array_field_tuple = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Object field
    obj_field = Object(allow_null=False, properties={"name": String(), "age": Integer()}, required=["name"], min_properties=1, max_properties=10)
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    
    # Test Object with pattern_properties
    obj_field_pattern = Object(allow_null=False, pattern_properties={"^S_": String()})
    result = to_json_schema(obj_field_pattern)
    assert "^S_" in result["patternProperties"]
    
    # Test Object with additional_properties as bool
    obj_field_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(obj_field_additional)
    assert result["additionalProperties"] is False
    
    # Test Object with property_names
    obj_field_prop_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_field_prop_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(all_of_field)
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(if_clause=Boolean(), then_clause=String(), else_clause=Integer())
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    assert result["if"]["type"] == "boolean"
    assert result["then"]["type"] == "string"
    assert result["else"]["type"] == "integer"
    
    # Test IfThenElse field with only if clause
    if_only_field = IfThenElse(if_clause=Boolean())
    result = to_json_schema(if_only_field)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result
    
    # Test with default values
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result.get("default") == "default_value"
    
    # Test with Definitions
    definitions = Definitions({"MyString": String()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "MyString" in result["components"]["schemas"]
    
    # Test Reference field
    ref_field = Reference(to="MyType", definitions=Definitions({"MyType": String()}))
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/MyType"
    assert "components" in result
    assert "MyType" in result["components"]["schemas"]
    
    # Test Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    result = to_json_schema(String(allow_null=False))
    assert result["type"] == "string"
    assert "null" not in str(result["type"])

    # Test with String field allowing null
    result = to_json_schema(String(allow_null=True))
    assert result["type"] == ["string", "null"]

    # Test with String field with min/max length
    result = to_json_schema(String(min_length=5, max_length=10))
    assert result["minLength"] == 5
    assert result["maxLength"] == 10

    # Test with String field with pattern
    result = to_json_schema(String(pattern="^[a-z]+$"))
    assert result["pattern"] == "^[a-z]+$"

    # Test with String field with format
    result = to_json_schema(String(format="email"))
    assert result["format"] == "email"

    # Test with Integer field
    result = to_json_schema(Integer(allow_null=False))
    assert result["type"] == "integer"

    # Test with Integer field with constraints
    result = to_json_schema(Integer(minimum=0, maximum=100, multiple_of=5))
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test with Float field
    result = to_json_schema(Float(allow_null=False))
    assert result["type"] == "number"

    # Test with Float field with exclusive constraints
    result = to_json_schema(Float(exclusive_minimum=0, exclusive_maximum=100))
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100

    # Test with Boolean field
    result = to_json_schema(Boolean(allow_null=False))
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    result = to_json_schema(Boolean(allow_null=True))
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    result = to_json_schema(Array(items=String(), allow_null=False))
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"

    # Test with Array field with constraints
    result = to_json_schema(Array(min_items=1, max_items=10, unique_items=True))
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True

    # Test with Array field with tuple items
    result = to_json_schema(Array(items=[String(), Integer()]))
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test with Object field
    result = to_json_schema(Object(properties={"name": String(), "age": Integer()}))
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]

    # Test with Object field with required fields
    result = to_json_schema(Object(required=["name", "age"]))
    assert result["required"] == ["name", "age"]

    # Test with Object field with pattern properties
    result = to_json_schema(Object(pattern_properties={"^S_": String()}))
    assert "patternProperties" in result

    # Test with Object field with additional properties as boolean
    result = to_json_schema(Object(additional_properties=False))
    assert result["additionalProperties"] is False

    # Test with Object field with additional properties as Field
    result = to_json_schema(Object(additional_properties=String()))
    assert isinstance(result["additionalProperties"], dict)

    # Test with Object field with property names
    result = to_json_schema(Object(property_names=String(pattern="^[a-z]+$")))
    assert "propertyNames" in result

    # Test with Choice field
    result = to_json_schema(Choice(choices=[("red", "Red"), ("blue", "Blue")]))
    assert result["enum"] == ["red", "blue"]

    # Test with Const field
    result = to_json_schema(Const(const="fixed_value"))
    assert result["const"] == "fixed_value"

    # Test with Union field
    result = to_json_schema(Union(any_of=[String(), Integer()]))
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    result = to_json_schema(OneOf(one_of=[String(), Integer()]))
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    result = to_json_schema(AllOf(all_of=[String(), Object()]))
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with IfThenElse field
    result = to_json_schema(IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    ))
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without else clause
    result = to_json_schema(IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    ))
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with Not field
    result = to_json_schema(Not(negated=String()))
    assert "not" in result

    # Test with Definitions
    definitions = Definitions()
    definitions["User"] = Object(properties={"name": String()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]

    # Test with Reference field
    definitions = Definitions()
    definitions["User"] = Object()
    ref_field = Reference(to="User", definitions=definitions)
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/User"

    # Test with Schema field
    result = to_json_schema(Schema(fields={"name": String(), "age": Integer()}))
    assert result["type"] == "object"
    assert "properties" in result

    # Test with default values
    result = to_json_schema(String(default="default_value"))
    assert result["default"] == "default_value"

    # Test nested structures
    result = to_json_schema(Object(
        properties={
            "user": Object(properties={"name": String(), "age": Integer()}),
            "tags": Array(items=String())
        }
    ))
    assert result["type"] == "object"
    assert result["properties"]["user"]["type"] == "object"
    assert result["properties"]["tags"]["type"] == "array"

    # Test with String field with blank not allowed
    result = to_json_schema(String(allow_blank=False))
    assert result["minLength"] == 1

    # Test with Array field with additional items as boolean
    result = to_json_schema(Array(additional_items=False))
    assert result["additionalItems"] is False

    # Test with Array field with additional items as Field
    result = to_json_schema(Array(additional_items=String()))
    assert isinstance(result["additionalItems"], dict)


# LLM-generated content at query #2
#--------------------------

```python
def test_ref_from_json_schema():
    # Test basic reference
    data = {"$ref": "#/components/schemas/User"}
    defs = Definitions()
    result = ref_from_json_schema(data, defs)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is defs

    # Test reference with nested path
    data = {"$ref": "#/definitions/Address"}
    defs = Definitions()
    result = ref_from_json_schema(data, defs)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/Address"

    # Test reference with complex path
    data = {"$ref": "#/components/schemas/nested/Model"}
    defs = Definitions()
    result = ref_from_json_schema(data, defs)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/nested/Model"

    # Test that unsupported $ref style raises assertion
    data = {"$ref": "external.json#/definitions/Model"}
    defs = Definitions()
    with pytest.raises(AssertionError, match="Unsupported \\$ref style"):
        ref_from_json_schema(data, defs)

    # Test with relative reference (should fail)
    data = {"$ref": "definitions/Model"}
    defs = Definitions()
    with pytest.raises(AssertionError, match="Unsupported \\$ref style"):
        ref_from_json_schema(data, defs)

    # Test that Reference uses provided definitions object
    data = {"$ref": "#/definitions/Test"}
    defs = Definitions()
    defs["#/definitions/Test"] = String()
    result = ref_from_json_schema(data, defs)
    assert result.definitions is defs
    assert "#/definitions/Test" in result.definitions


# LLM-generated content at query #3
#--------------------------

```python
def test_ref_from_json_schema():
    """Test ref_from_json_schema function."""
    defs = Definitions()
    
    # Test basic reference
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, defs)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is defs
    
    # Test another reference format
    data = {"$ref": "#/definitions/Address"}
    result = ref_from_json_schema(data, defs)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/Address"
    
    # Test with nested path reference
    data = {"$ref": "#/components/schemas/models/Product"}
    result = ref_from_json_schema(data, defs)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/models/Product"
    
    # Test that unsupported $ref style raises AssertionError
    data = {"$ref": "external.json#/definitions/Item"}
    try:
        ref_from_json_schema(data, defs)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "Unsupported $ref style" in str(e)
    
    # Test with relative reference (should fail)
    data = {"$ref": "schemas/User"}
    try:
        ref_from_json_schema(data, defs)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "Unsupported $ref style" in str(e)
    
    # Test with different definitions object
    defs2 = Definitions()
    data = {"$ref": "#/definitions/Custom"}
    result = ref_from_json_schema(data, defs2)
    assert result.definitions is defs2
    assert result.definitions is not defs


# LLM-generated content at query #4
#--------------------------

```python
def test_one_of_from_json_schema():
    """Test one_of_from_json_schema function."""
    
    # Test basic oneOf with simple types
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    defs = Definitions()
    result = one_of_from_json_schema(data, defs)
    
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Integer)
    
    # Test oneOf with default value
    data_with_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "boolean"}
        ],
        "default": "test"
    }
    result_with_default = one_of_from_json_schema(data_with_default, defs)
    
    assert isinstance(result_with_default, OneOf)
    assert result_with_default.default == "test"
    assert len(result_with_default.one_of) == 2
    
    # Test oneOf with complex schemas
    data_complex = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            {
                "type": "array",
                "items": {"type": "number"}
            }
        ]
    }
    result_complex = one_of_from_json_schema(data_complex, defs)
    
    assert isinstance(result_complex, OneOf)
    assert len(result_complex.one_of) == 2
    assert isinstance(result_complex.one_of[0], Object)
    assert isinstance(result_complex.one_of[1], Array)
    
    # Test oneOf with no default
    data_no_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result_no_default = one_of_from_json_schema(data_no_default, defs)
    
    assert isinstance(result_no_default, OneOf)
    assert result_no_default.default is NO_DEFAULT
    assert len(result_no_default.one_of) == 2
    
    # Test oneOf with single item
    data_single = {
        "oneOf": [
            {"type": "string"}
        ]
    }
    result_single = one_of_from_json_schema(data_single, defs)
    
    assert isinstance(result_single, OneOf)
    assert len(result_single.one_of) == 1
    assert isinstance(result_single.one_of[0], String)
    
    # Test oneOf with references
    defs_with_ref = Definitions()
    defs_with_ref["#/components/schemas/StringType"] = String()
    
    data_with_ref = {
        "oneOf": [
            {"$ref": "#/components/schemas/StringType"},
            {"type": "integer"}
        ]
    }
    result_with_ref = one_of_from_json_schema(data_with_ref, defs_with_ref)
    
    assert isinstance(result_with_ref, OneOf)
    assert len(result_with_ref.one_of) == 2
    assert isinstance(result_with_ref.one_of[0], Reference)
    assert isinstance(result_with_ref.one_of[1], Integer)


# LLM-generated content at query #5
#--------------------------

```python
def test_enum_from_json_schema():
    # Test with basic enum values
    data = {"enum": [1, 2, 3]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1, 1), (2, 2), (3, 3)]
    assert result.default == NO_DEFAULT

    # Test with string enum values
    data = {"enum": ["red", "green", "blue"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("red", "red"), ("green", "green"), ("blue", "blue")]
    assert result.default == NO_DEFAULT

    # Test with mixed type enum values
    data = {"enum": [1, "two", None, True]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1, 1), ("two", "two"), (None, None), (True, True)]
    assert result.default == NO_DEFAULT

    # Test with default value
    data = {"enum": ["a", "b", "c"], "default": "b"}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]
    assert result.default == "b"

    # Test with single enum value
    data = {"enum": [42]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(42, 42)]
    assert result.default == NO_DEFAULT

    # Test with boolean enum values
    data = {"enum": [True, False]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(True, True), (False, False)]
    assert result.default == NO_DEFAULT

    # Test with null value in enum
    data = {"enum": [None, "value"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(None, None), ("value", "value")]
    assert result.default == NO_DEFAULT


# LLM-generated content at query #6
#--------------------------

```python
def test_all_of_from_json_schema():
    # Test basic allOf with multiple constraints
    data = {
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], String)
    assert isinstance(result.all_of[1], String)
    assert result.default is NO_DEFAULT

    # Test allOf with default value
    data_with_default = {
        "allOf": [
            {"type": "integer", "minimum": 0},
            {"type": "integer", "maximum": 100}
        ],
        "default": 50
    }
    result = all_of_from_json_schema(data_with_default, definitions)
    assert isinstance(result, AllOf)
    assert result.default == 50
    assert len(result.all_of) == 2

    # Test allOf with single item
    data_single = {
        "allOf": [
            {"type": "boolean"}
        ]
    }
    result = all_of_from_json_schema(data_single, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 1
    assert isinstance(result.all_of[0], Boolean)

    # Test allOf with complex nested schemas
    data_complex = {
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"}
                }
            }
        ]
    }
    result = all_of_from_json_schema(data_complex, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], Object)
    assert isinstance(result.all_of[1], Object)

    # Test allOf with references
    test_defs = Definitions()
    test_defs["#/definitions/StringField"] = String(min_length=1)
    data_with_ref = {
        "allOf": [
            {"$ref": "#/definitions/StringField"},
            {"type": "string", "maxLength": 50}
        ]
    }
    result = all_of_from_json_schema(data_with_ref, test_defs)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], Reference)
    assert isinstance(result.all_of[1], String)


# LLM-generated content at query #7
#--------------------------

```python
def test_all_of_from_json_schema():
    # Test basic allOf with simple types
    data = {
        "allOf": [
            {"type": "string", "minLength": 1},
            {"type": "string", "maxLength": 10}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], String)
    assert isinstance(result.all_of[1], String)
    
    # Test allOf with default value
    data_with_default = {
        "allOf": [
            {"type": "integer", "minimum": 0},
            {"type": "integer", "maximum": 100}
        ],
        "default": 50
    }
    result = all_of_from_json_schema(data_with_default, definitions)
    assert isinstance(result, AllOf)
    assert result.default == 50
    
    # Test allOf with no default
    data_no_default = {
        "allOf": [
            {"type": "boolean"},
            {"const": True}
        ]
    }
    result = all_of_from_json_schema(data_no_default, definitions)
    assert isinstance(result, AllOf)
    assert result.default is NO_DEFAULT
    
    # Test allOf with complex nested schemas
    data_complex = {
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            {
                "type": "object",
                "required": ["name"]
            }
        ]
    }
    result = all_of_from_json_schema(data_complex, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], Object)
    assert isinstance(result.all_of[1], Object)
    
    # Test allOf with single item
    data_single = {
        "allOf": [
            {"type": "string"}
        ]
    }
    result = all_of_from_json_schema(data_single, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 1
    
    # Test allOf with empty constraints
    data_empty = {
        "allOf": []
    }
    result = all_of_from_json_schema(data_empty, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 0


# LLM-generated content at query #8
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type: string
    result = type_from_json_schema({"type": "string"}, definitions)
    assert isinstance(result, String)
    
    # Test single type: integer
    result = type_from_json_schema({"type": "integer"}, definitions)
    assert isinstance(result, Integer)
    
    # Test single type: number
    result = type_from_json_schema({"type": "number"}, definitions)
    assert isinstance(result, Number)
    
    # Test single type: boolean
    result = type_from_json_schema({"type": "boolean"}, definitions)
    assert isinstance(result, Boolean)
    
    # Test single type: array
    result = type_from_json_schema({"type": "array"}, definitions)
    assert isinstance(result, Array)
    
    # Test single type: object
    result = type_from_json_schema({"type": "object"}, definitions)
    assert isinstance(result, Object)
    
    # Test multiple types (union)
    result = type_from_json_schema({"type": ["string", "integer"]}, definitions)
    assert isinstance(result, Union)
    
    # Test null type with allow_null
    result = type_from_json_schema({"type": ["string", "null"]}, definitions)
    assert isinstance(result, String)
    assert result.allow_null is True
    
    # Test only null type
    result = type_from_json_schema({"type": "null"}, definitions)
    assert isinstance(result, Const)
    
    # Test no type (returns NeverMatch)
    result = type_from_json_schema({}, definitions)
    assert isinstance(result, NeverMatch)
    
    # Test with constraints (minLength for string)
    result = type_from_json_schema(
        {"type": "string", "minLength": 5}, definitions
    )
    assert isinstance(result, String)
    assert result.min_length == 5
    
    # Test with constraints (minimum for number)
    result = type_from_json_schema(
        {"type": "number", "minimum": 10}, definitions
    )
    assert isinstance(result, Number)
    assert result.minimum == 10
    
    # Test multiple types with null
    result = type_from_json_schema(
        {"type": ["string", "integer", "null"]}, definitions
    )
    assert isinstance(result, Union)
    assert result.allow_null is True
    
    # Test array with items constraint
    result = type_from_json_schema(
        {"type": "array", "items": {"type": "string"}}, definitions
    )
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    
    # Test object with properties
    result = type_from_json_schema(
        {"type": "object", "properties": {"name": {"type": "string"}}}, definitions
    )
    assert isinstance(result, Object)
    assert "name" in result.properties


# LLM-generated content at query #9
#--------------------------

```python
def test_from_json_schema():
    # Test with boolean schema - True
    result = from_json_schema(True)
    assert isinstance(result, Any)

    # Test with boolean schema - False
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

    # Test with simple type constraint
    result = from_json_schema({"type": "string"})
    assert isinstance(result, String)

    # Test with integer type
    result = from_json_schema({"type": "integer"})
    assert isinstance(result, Integer)

    # Test with number type
    result = from_json_schema({"type": "number"})
    assert isinstance(result, Number)

    # Test with boolean type
    result = from_json_schema({"type": "boolean"})
    assert isinstance(result, Boolean)

    # Test with array type
    result = from_json_schema({"type": "array"})
    assert isinstance(result, Array)

    # Test with object type
    result = from_json_schema({"type": "object"})
    assert isinstance(result, Object)

    # Test with enum constraint
    result = from_json_schema({"enum": [1, 2, 3]})
    assert isinstance(result, Choice)

    # Test with const constraint
    result = from_json_schema({"const": "fixed_value"})
    assert isinstance(result, Const)

    # Test with string constraints
    result = from_json_schema({
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    })
    assert isinstance(result, String)

    # Test with numeric constraints
    result = from_json_schema({
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 5
    })
    assert isinstance(result, Number)

    # Test with array constraints
    result = from_json_schema({
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True
    })
    assert isinstance(result, Array)

    # Test with object properties
    result = from_json_schema({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    })
    assert isinstance(result, Object)

    # Test with allOf constraint
    result = from_json_schema({
        "allOf": [
            {"type": "string"},
            {"minLength": 5}
        ]
    })
    assert isinstance(result, AllOf)

    # Test with anyOf constraint
    result = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(result, Union)

    # Test with oneOf constraint
    result = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(result, OneOf)

    # Test with not constraint
    result = from_json_schema({
        "not": {"type": "null"}
    })
    assert isinstance(result, Not)

    # Test with if-then-else constraint
    result = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    })
    assert isinstance(result, IfThenElse)

    # Test with multiple constraints combined
    result = from_json_schema({
        "type": "string",
        "enum": ["red", "green", "blue"],
        "minLength": 3
    })
    assert isinstance(result, AllOf)

    # Test with $ref
    defs = Definitions()
    defs["#/components/schemas/StringType"] = String()
    result = from_json_schema(
        {"$ref": "#/components/schemas/StringType"},
        definitions=defs
    )
    assert isinstance(result, Reference)

    # Test with definitions in data
    result = from_json_schema({
        "type": "object",
        "components": {
            "schemas": {
                "MyString": {"type": "string"}
            }
        }
    })
    assert isinstance(result, Object)

    # Test empty object returns Any
    result = from_json_schema({})
    assert isinstance(result, Any)

    # Test with additionalProperties
    result = from_json_schema({
        "type": "object",
        "additionalProperties": {"type": "string"}
    })
    assert isinstance(result, Object)

    # Test with patternProperties
    result = from_json_schema({
        "type": "object",
        "patternProperties": {
            "^[a-z]+$": {"type": "string"}
        }
    })
    assert isinstance(result, Object)


# LLM-generated content at query #10
#--------------------------

```python
def test_all_of_from_json_schema():
    # Test basic allOf with simple constraints
    data = {
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    }
    result = all_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    
    # Test allOf with default value
    data_with_default = {
        "allOf": [
            {"type": "integer", "minimum": 0},
            {"type": "integer", "maximum": 100}
        ],
        "default": 50
    }
    result_with_default = all_of_from_json_schema(data_with_default, definitions=Definitions())
    assert isinstance(result_with_default, AllOf)
    assert result_with_default.default == 50
    
    # Test allOf with no default
    data_no_default = {
        "allOf": [
            {"type": "boolean"},
            {"const": True}
        ]
    }
    result_no_default = all_of_from_json_schema(data_no_default, definitions=Definitions())
    assert isinstance(result_no_default, AllOf)
    assert result_no_default.default is NO_DEFAULT
    
    # Test allOf with single constraint
    data_single = {
        "allOf": [
            {"type": "number", "minimum": 0}
        ]
    }
    result_single = all_of_from_json_schema(data_single, definitions=Definitions())
    assert isinstance(result_single, AllOf)
    assert len(result_single.all_of) == 1
    
    # Test allOf with nested references
    defs = Definitions()
    data_with_ref = {
        "allOf": [
            {"$ref": "#/definitions/StringField"},
            {"minLength": 1}
        ]
    }
    result_with_ref = all_of_from_json_schema(data_with_ref, definitions=defs)
    assert isinstance(result_with_ref, AllOf)
    assert len(result_with_ref.all_of) == 2
    
    # Test allOf with complex nested structures
    data_complex = {
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            {
                "required": ["name"]
            }
        ]
    }
    result_complex = all_of_from_json_schema(data_complex, definitions=Definitions())
    assert isinstance(result_complex, AllOf)
    assert len(result_complex.all_of) == 2


# LLM-generated content at query #11
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test basic if-then-else structure
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    
    # Test if-then without else
    data_no_else = {
        "if": {"type": "object"},
        "then": {"minProperties": 1}
    }
    result_no_else = if_then_else_from_json_schema(data_no_else, definitions)
    assert isinstance(result_no_else, IfThenElse)
    assert result_no_else.if_clause is not None
    assert result_no_else.then_clause is not None
    assert result_no_else.else_clause is None
    
    # Test if-else without then
    data_no_then = {
        "if": {"type": "array"},
        "else": {"maxItems": 10}
    }
    result_no_then = if_then_else_from_json_schema(data_no_then, definitions)
    assert isinstance(result_no_then, IfThenElse)
    assert result_no_then.if_clause is not None
    assert result_no_then.then_clause is None
    assert result_no_then.else_clause is not None
    
    # Test with default value
    data_with_default = {
        "if": {"type": "boolean"},
        "then": {"const": True},
        "else": {"const": False},
        "default": False
    }
    result_with_default = if_then_else_from_json_schema(data_with_default, definitions)
    assert isinstance(result_with_default, IfThenElse)
    assert result_with_default.default == False
    
    # Test with complex nested schemas
    data_complex = {
        "if": {"properties": {"type": {"const": "string"}}},
        "then": {"properties": {"value": {"type": "string"}}},
        "else": {"properties": {"value": {"type": "number"}}}
    }
    result_complex = if_then_else_from_json_schema(data_complex, definitions)
    assert isinstance(result_complex, IfThenElse)
    assert result_complex.if_clause is not None
    assert result_complex.then_clause is not None
    assert result_complex.else_clause is not None
    
    # Test if-only structure (no then, no else)
    data_if_only = {
        "if": {"type": "integer"}
    }
    result_if_only = if_then_else_from_json_schema(data_if_only, definitions)
    assert isinstance(result_if_only, IfThenElse)
    assert result_if_only.if_clause is not None
    assert result_if_only.then_clause is None
    assert result_if_only.else_clause is None


# LLM-generated content at query #12
#--------------------------

```python
def test_from_json_schema_type():
    """Test from_json_schema_type function with various type strings."""
    defs = Definitions()
    
    # Test number type
    result = from_json_schema_type(
        {"minimum": 0, "maximum": 100, "multipleOf": 5},
        type_string="number",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.multiple_of == 5
    assert result.allow_null is False
    
    # Test number type with null
    result = from_json_schema_type(
        {"minimum": -10, "maximum": 10},
        type_string="number",
        allow_null=True,
        definitions=defs
    )
    assert isinstance(result, Float)
    assert result.allow_null is True
    
    # Test integer type
    result = from_json_schema_type(
        {"minimum": 1, "maximum": 10},
        type_string="integer",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Integer)
    assert result.minimum == 1
    assert result.maximum == 10
    
    # Test integer with exclusive bounds
    result = from_json_schema_type(
        {"exclusiveMinimum": 0, "exclusiveMaximum": 100},
        type_string="integer",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Integer)
    assert result.exclusive_minimum == 0
    assert result.exclusive_maximum == 100
    
    # Test string type
    result = from_json_schema_type(
        {"minLength": 1, "maxLength": 50, "pattern": "^[a-z]+$"},
        type_string="string",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, String)
    assert result.min_length == 1
    assert result.max_length == 50
    assert result.pattern == "^[a-z]+$"
    
    # Test string with allow_blank
    result = from_json_schema_type(
        {"minLength": 0},
        type_string="string",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, String)
    assert result.allow_blank is True
    
    # Test string with format
    result = from_json_schema_type(
        {"format": "email"},
        type_string="string",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, String)
    assert result.format == "email"
    
    # Test boolean type
    result = from_json_schema_type(
        {},
        type_string="boolean",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Boolean)
    assert result.allow_null is False
    
    # Test boolean with null
    result = from_json_schema_type(
        {},
        type_string="boolean",
        allow_null=True,
        definitions=defs
    )
    assert isinstance(result, Boolean)
    assert result.allow_null is True
    
    # Test array type
    result = from_json_schema_type(
        {"minItems": 1, "maxItems": 10, "uniqueItems": True},
        type_string="array",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Array)
    assert result.min_items == 1
    assert result.max_items == 10
    assert result.unique_items is True
    
    # Test array with items schema
    result = from_json_schema_type(
        {"items": {"type": "string"}},
        type_string="array",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    
    # Test array with items as list
    result = from_json_schema_type(
        {"items": [{"type": "string"}, {"type": "integer"}]},
        type_string="array",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Array)
    assert isinstance(result.items, list)
    assert len(result.items) == 2
    
    # Test array with additionalItems
    result = from_json_schema_type(
        {"additionalItems": {"type": "number"}},
        type_string="array",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Array)
    assert isinstance(result.additional_items, Float)
    
    # Test object type
    result = from_json_schema_type(
        {"properties": {"name": {"type": "string"}}, "required": ["name"]},
        type_string="object",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert isinstance(result.properties["name"], String)
    assert result.required == ["name"]
    
    # Test object with patternProperties
    result = from_json_schema_type(
        {"patternProperties": {"^S_": {"type": "string"}}},
        type_string="object",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Object)
    assert "^S_" in result.pattern_properties
    
    # Test object with additionalProperties as boolean
    result = from_json_schema_type(
        {"additionalProperties": False},
        type_string="object",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Object)
    assert result.additional_properties is False
    
    # Test object with additionalProperties as schema
    result = from_json_schema_type(
        {"additionalProperties": {"type": "string"}},
        type_string="object",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Object)
    assert isinstance(result.additional_properties, String)
    
    # Test object with propertyNames
    result = from_json_schema_type(
        {"propertyNames": {"pattern": "^[a-z]+$"}},
        type_string="object",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Object)
    assert isinstance(result.property_names, String)
    
    # Test object with min/max properties
    result = from_json_schema_type(
        {"minProperties": 1, "maxProperties": 5},
        type_string="object",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Object)
    assert result.min_properties == 1
    assert result.max_properties == 5
    
    # Test with default values
    result = from_json_schema_type(
        {"default": 42},
        type_string="integer",
        allow_null=False,
        definitions=defs
    )
    assert isinstance(result, Integer)
    assert result.default == 42


# LLM-generated content at query #13
#--------------------------

```python
def test_if_then_else_from_json_schema():
    """Test if_then_else_from_json_schema function."""
    
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    
    # Test with only if and then
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None
    
    # Test with only if and else
    data = {
        "if": {"type": "string"},
        "else": {"type": "number"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None
    
    # Test with only if clause
    data = {
        "if": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None
    
    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"},
        "default": "test_default",
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == "test_default"
    
    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"properties": {"age": {"type": "integer"}}},
        "else": {"type": "array", "items": {"type": "string"}},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #14
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    result = from_json_schema_type(
        {
            "minimum": 0,
            "maximum": 100,
            "exclusiveMinimum": 10,
            "exclusiveMaximum": 90,
            "multipleOf": 5,
            "default": 50,
        },
        type_string="number",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Float)
    assert result.allow_null is False
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.exclusive_minimum == 10
    assert result.exclusive_maximum == 90
    assert result.multiple_of == 5
    assert result.default == 50

    # Test number type with allow_null
    result = from_json_schema_type(
        {"minimum": 0},
        type_string="number",
        allow_null=True,
        definitions=Definitions(),
    )
    assert isinstance(result, Float)
    assert result.allow_null is True

    # Test integer type
    result = from_json_schema_type(
        {
            "minimum": 1,
            "maximum": 10,
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 11,
            "multipleOf": 2,
            "default": 5,
        },
        type_string="integer",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Integer)
    assert result.minimum == 1
    assert result.maximum == 10
    assert result.exclusive_minimum == 0
    assert result.exclusive_maximum == 11
    assert result.multiple_of == 2
    assert result.default == 5

    # Test string type with minLength > 1
    result = from_json_schema_type(
        {
            "minLength": 5,
            "maxLength": 20,
            "pattern": "^[a-z]+$",
            "format": "email",
            "default": "test",
        },
        type_string="string",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, String)
    assert result.allow_blank is False
    assert result.min_length == 5
    assert result.max_length == 20
    assert result.pattern == "^[a-z]+$"
    assert result.format == "email"
    assert result.default == "test"

    # Test string type with minLength == 0
    result = from_json_schema_type(
        {"minLength": 0, "maxLength": 10},
        type_string="string",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, String)
    assert result.allow_blank is True
    assert result.min_length is None

    # Test string type with minLength == 1
    result = from_json_schema_type(
        {"minLength": 1},
        type_string="string",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, String)
    assert result.min_length is None

    # Test boolean type
    result = from_json_schema_type(
        {"default": True},
        type_string="boolean",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Boolean)
    assert result.allow_null is False
    assert result.default is True

    # Test boolean type with allow_null
    result = from_json_schema_type(
        {},
        type_string="boolean",
        allow_null=True,
        definitions=Definitions(),
    )
    assert isinstance(result, Boolean)
    assert result.allow_null is True

    # Test array type without items
    result = from_json_schema_type(
        {"minItems": 1, "maxItems": 10, "uniqueItems": True},
        type_string="array",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Array)
    assert result.min_items == 1
    assert result.max_items == 10
    assert result.unique_items is True
    assert result.items is None

    # Test array type with items as schema
    result = from_json_schema_type(
        {"items": {"type": "string"}},
        type_string="array",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

    # Test array type with items as list
    result = from_json_schema_type(
        {"items": [{"type": "string"}, {"type": "integer"}]},
        type_string="array",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Array)
    assert isinstance(result.items, list)
    assert len(result.items) == 2

    # Test array type with additionalItems as boolean
    result = from_json_schema_type(
        {"additionalItems": False},
        type_string="array",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Array)
    assert result.additional_items is False

    # Test array type with additionalItems as schema
    result = from_json_schema_type(
        {"additionalItems": {"type": "number"}},
        type_string="array",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Array)
    assert isinstance(result.additional_items, Float)

    # Test object type without properties
    result = from_json_schema_type(
        {},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert result.properties is None

    # Test object type with properties
    result = from_json_schema_type(
        {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert isinstance(result.properties, dict)
    assert "name" in result.properties
    assert "age" in result.properties

    # Test object type with patternProperties
    result = from_json_schema_type(
        {"patternProperties": {"^S_": {"type": "string"}}},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert isinstance(result.pattern_properties, dict)
    assert "^S_" in result.pattern_properties

    # Test object type with additionalProperties as boolean
    result = from_json_schema_type(
        {"additionalProperties": False},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert result.additional_properties is False

    # Test object type with additionalProperties as schema
    result = from_json_schema_type(
        {"additionalProperties": {"type": "string"}},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert isinstance(result.additional_properties, String)

    # Test object type with propertyNames
    result = from_json_schema_type(
        {"propertyNames": {"pattern": "^[a-z]+$"}},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert isinstance(result.property_names, String)

    # Test object type with required, minProperties, maxProperties
    result = from_json_schema_type(
        {
            "required": ["name"],
            "minProperties": 1,
            "maxProperties": 5,
        },
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)


# LLM-generated content at query #15
#--------------------------

```python
def test_if_then_else_from_json_schema():
    """Test if_then_else_from_json_schema function."""
    
    # Test with if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None
    
    # Test with if, then, and else clauses
    data = {
        "if": {"type": "integer"},
        "then": {"minimum": 0},
        "else": {"maximum": 0},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    
    # Test with only if clause (no then or else)
    data = {
        "if": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None
    
    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 1},
        "default": "test",
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == "test"
    
    # Test with complex nested schemas
    data = {
        "if": {"properties": {"type": {"enum": ["A"]}}},
        "then": {"required": ["fieldA"]},
        "else": {"required": ["fieldB"]},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None

    # Test with only if and then clauses
    data_no_else = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
    }
    result_no_else = if_then_else_from_json_schema(data_no_else, definitions)
    assert isinstance(result_no_else, IfThenElse)
    assert result_no_else.if_clause is not None
    assert result_no_else.then_clause is not None
    assert result_no_else.else_clause is None

    # Test with only if and else clauses
    data_no_then = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
    }
    result_no_then = if_then_else_from_json_schema(data_no_then, definitions)
    assert isinstance(result_no_then, IfThenElse)
    assert result_no_then.if_clause is not None
    assert result_no_then.then_clause is None
    assert result_no_then.else_clause is not None

    # Test with only if clause
    data_only_if = {
        "if": {"type": "string"},
    }
    result_only_if = if_then_else_from_json_schema(data_only_if, definitions)
    assert isinstance(result_only_if, IfThenElse)
    assert result_only_if.if_clause is not None
    assert result_only_if.then_clause is None
    assert result_only_if.else_clause is None

    # Test with default value
    data_with_default = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "default": 42,
    }
    result_with_default = if_then_else_from_json_schema(data_with_default, definitions)
    assert isinstance(result_with_default, IfThenElse)
    assert result_with_default.default == 42

    # Test with complex nested schemas
    data_complex = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "integer"}},
        "else": {"enum": [None, "unknown"]},
    }
    result_complex = if_then_else_from_json_schema(data_complex, definitions)
    assert isinstance(result_complex, IfThenElse)
    assert result_complex.if_clause is not None
    assert result_complex.then_clause is not None
    assert result_complex.else_clause is not None


# LLM-generated content at query #17
#--------------------------

```python
def test_if_then_else_from_json_schema():
    """Test if_then_else_from_json_schema with various configurations."""
    defs = Definitions()
    
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    
    # Test with only if clause
    data = {
        "if": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None
    
    # Test with if and then clauses
    data = {
        "if": {"type": "array"},
        "then": {"type": "object"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None
    
    # Test with if and else clauses
    data = {
        "if": {"type": "number"},
        "else": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None
    
    # Test with default value
    data = {
        "if": {"type": "boolean"},
        "then": {"type": "string"},
        "else": {"type": "integer"},
        "default": "test_default",
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.default == "test_default"
    
    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "string"}},
        "else": {"enum": [1, 2, 3]},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_if_then_else_from_json_schema():
    """Test if_then_else_from_json_schema function."""
    defs = Definitions()
    
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, Integer)
    assert isinstance(result.else_clause, Boolean)
    
    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, Integer)
    assert result.else_clause is None
    
    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert result.then_clause is None
    assert isinstance(result.else_clause, Boolean)
    
    # Test with only if clause
    data = {
        "if": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert result.then_clause is None
    assert result.else_clause is None
    
    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": "test_default",
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.default == "test_default"
    
    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "integer"}},
        "else": {"type": "number"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Object)
    assert isinstance(result.then_clause, Array)
    assert isinstance(result.else_clause, Float)


# LLM-generated content at query #19
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None

    # Test with if, then, and else clauses
    data = {
        "if": {"type": "number"},
        "then": {"minimum": 0},
        "else": {"maximum": 0},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None

    # Test with only if clause
    data = {
        "if": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with default value
    data = {
        "if": {"type": "integer"},
        "then": {"minimum": 10},
        "default": 42,
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == 42

    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"required": ["name"]},
        "else": {"properties": {"id": {"type": "integer"}}},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #20
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with String field allowing blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result

    # Test with Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test with Integer field allowing null
    int_field_null = Integer(allow_null=True)
    result = to_json_schema(int_field_null)
    assert result["type"] == ["integer", "null"]

    # Test with Float field
    float_field = Float(allow_null=False, exclusive_minimum=0.0, exclusive_maximum=100.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 100.0

    # Test with Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, min_items=1, max_items=10, items=String(), unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert "items" in result

    # Test with Array field with list of items
    array_field_list = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_list)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test with Object field
    obj_field = Object(allow_null=False, properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]

    # Test with Object field with pattern properties
    obj_field_pattern = Object(allow_null=False, pattern_properties={"^S_": String()})
    result = to_json_schema(obj_field_pattern)
    assert "patternProperties" in result

    # Test with Object field with min/max properties
    obj_field_props = Object(allow_null=False, min_properties=1, max_properties=5)
    result = to_json_schema(obj_field_props)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5

    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=100)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without else clause
    if_then_field = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with Definitions
    definitions = Definitions({"StringDef": String(), "IntDef": Integer()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]

    # Test with Reference field
    ref_field = Reference(to="StringDef", definitions=Definitions({"StringDef": String()}))
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/StringDef"

    # Test with default values
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"

    # Test with Object field with additional properties as bool
    obj_additional_bool = Object(additional_properties=False)
    result = to_json_schema(obj_additional_bool)
    assert result["additionalProperties"] is False

    # Test with Object field with additional properties as Field
    obj_additional_field = Object(additional_properties=String())
    result = to_json_schema(obj_additional_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test with Array field with additional items as bool
    array_additional_bool = Array(additional_items=False)
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] is False

    # Test with Array field with additional items as Field
    array_additional_field = Array(additional_items=String())
    result = to_json_schema(array_additional_field)
    assert isinstance(result["additionalItems"], dict)

    # Test with Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    


# LLM-generated content at query #21
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with String field allowing blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result or result.get("minLength") is None

    # Test with Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=1, exclusive_maximum=99, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 1
    assert result["exclusiveMaximum"] == 99
    assert result["multipleOf"] == 5

    # Test with Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test with Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=5, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test with Array field with additional_items as bool
    array_field_additional = Array(allow_null=False, items=String(), additional_items=False)
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"] is False

    # Test with Array field with additional_items as Field
    array_field_additional_field = Array(allow_null=False, items=String(), additional_items=Integer())
    result = to_json_schema(array_field_additional_field)
    assert result["additionalItems"]["type"] == "integer"

    # Test with Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]

    # Test with Object field with pattern_properties
    object_field_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]
    assert "^I_" in result["patternProperties"]

    # Test with Object field with additional_properties as bool
    object_field_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(object_field_additional)
    assert result["additionalProperties"] is False

    # Test with Object field with additional_properties as Field
    object_field_additional_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(object_field_additional_field)
    assert result["additionalProperties"]["type"] == "string"

    # Test with Object field with property_names
    object_field_property_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_field_property_names)
    assert "propertyNames" in result
    assert result["propertyNames"]["type"] == "string"

    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=5)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without else_clause
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=None
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with Definitions
    definitions = Definitions({"StringDef": String(), "IntegerDef": Integer()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]

    # Test with Reference field
    definitions_obj = Definitions({"MyString": String()})
    ref_field = Reference(to="MyString", definitions=definitions_obj)
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert result["$ref"] == "#/


# LLM-generated content at query #22
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True
    
    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False
    
    # Test with String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"
    
    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test with String field blank allowed
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result
    
    # Test with Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=5, exclusive_maximum=95, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5
    assert result["exclusiveMaximum"] == 95
    assert result["multipleOf"] == 5
    
    # Test with Integer field allowing null
    int_field_null = Integer(allow_null=True)
    result = to_json_schema(int_field_null)
    assert result["type"] == ["integer", "null"]
    
    # Test with Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    
    # Test with Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test with Boolean field allowing null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test with Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert "items" in result
    
    # Test with Array field allowing null
    array_field_null = Array(allow_null=True)
    result = to_json_schema(array_field_null)
    assert result["type"] == ["array", "null"]
    
    # Test with Object field
    obj_field = Object(allow_null=False, properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test with Object field allowing null
    obj_field_null = Object(allow_null=True)
    result = to_json_schema(obj_field_null)
    assert result["type"] == ["object", "null"]
    
    # Test with Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test with Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"
    
    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=5)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test with IfThenElse field
    if_then_else_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test with IfThenElse field without else clause
    if_then_field = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with Reference field
    definitions_dict = {"User": Object(properties={"name": String()})}
    ref_field = Reference(to="User", definitions=definitions_dict)
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/User"
    
    # Test with Definitions
    definitions = Definitions({"User": Object(properties={"name": String()}), "Post": Object(properties={"title": String()})})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    
    # Test with default value
    string_with_default = String(default="test")
    result = to_json_schema(string_with_default)
    assert result["default"] == "test"
    
    # Test with Array of tuple items
    array_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test with additionalItems as boolean
    array_additional = Array(additional_items=False)
    result = to_json_schema(array_additional)
    assert result["additionalItems"] is False
    
    # Test with additionalItems as Field
    array_additional_field = Array(additional_items=String())
    result = to_json_schema(array_additional_field)
    assert isinstance(result["additionalItems"], dict)
    
    # Test with patternProperties
    obj_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    
    # Test with additionalProperties as boolean
    obj_additional = Object(additional_properties=False)
    result = to_json_schema(obj_additional)
    assert result["additionalProperties"] is False
    
    


# LLM-generated content at query #23
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    assert to_json_schema(Any()) == True
    
    # Test with NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test with String field
    string_field = String(allow_null=False)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    
    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test with String field with constraints
    string_field_constrained = String(
        allow_null=False,
        min_length=2,
        max_length=10,
        pattern="^[a-z]+$"
    )
    result = to_json_schema(string_field_constrained)
    assert result["type"] == "string"
    assert result["minLength"] == 2
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test with Integer field
    integer_field = Integer(
        allow_null=False,
        minimum=0,
        maximum=100
    )
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    
    # Test with Float field
    float_field = Float(allow_null=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test with Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test with Array field
    array_field = Array(
        allow_null=False,
        items=String(),
        min_items=1,
        max_items=5
    )
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert "items" in result
    
    # Test with Object field
    object_field = Object(
        allow_null=False,
        properties={
            "name": String(),
            "age": Integer()
        },
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test with Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test with Const field
    const_field = Const(const=42)
    result = to_json_schema(const_field)
    assert result["const"] == 42
    
    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=1)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test with IfThenElse field without else clause
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with Reference field
    definitions = Definitions()
    ref_field = Reference(to="MySchema", definitions=definitions)
    result = to_json_schema(ref_field)
    assert "$ref" in result
    
    # Test with Schema field
    schema_field = Schema(
        fields={
            "name": String(),
            "age": Integer()
        },
        required=["name"]
    )
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]
    
    # Test with Array field with unique_items
    array_unique = Array(items=String(), unique_items=True)
    result = to_json_schema(array_unique)
    assert result["uniqueItems"] == True
    
    # Test with Array field with additional_items
    array_additional = Array(
        items=[String(), Integer()],
        additional_items=Boolean()
    )
    result = to_json_schema(array_additional)
    assert isinstance(result["items"], list)
    assert "additionalItems" in result
    
    # Test with Object field with pattern_properties
    object_pattern = Object(
        pattern_properties={
            "^S_": String(),
            "^I_": Integer()
        }
    )
    result = to_json_schema(object_pattern)
    assert "patternProperties" in result
    
    # Test with Object field with additional_properties
    object_additional = Object(
        additional_properties=String()
    )
    result = to_json_schema(object_additional)
    assert "additionalProperties" in result
    
    # Test with Object field with property_names
    object_property_names = Object(
        property_names=String(pattern="^[a-z]+$")
    )
    result = to_json_schema(object_property_names)
    assert "propertyNames" in result
    
    # Test with Definitions
    definitions_obj = Definitions()
    definitions_obj["MyString"] = String()
    result = to_json_schema(definitions_obj)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "MyString" in result["components"]["schemas"]
    
    # Test with String allowing blank
    string_blank = String(allow_blank=True, min_length=0)
    result = to_json_schema(string_blank)
    assert "minLength" not in result or result.get("minLength") != 1
    
    # Test with String not allowing blank
    string_no_blank = String(allow_blank=False)
    result = to_json_schema(string_no_blank)
    assert result["minLength"] == 1
    
    # Test with Float with constraints
    float_constrained = Float(
        minimum=0.5,
        maximum=99.9,
        exclusive_minimum=0.1,
        exclusive_maximum=100.0,
        multiple_of=0.5
    )
    result = to_json_schema(float_constrained)
    assert result["minimum"] == 0.5
    assert result["maximum"] == 99.9
    assert result["exclusiveMinimum"] == 0.1
    assert result


# LLM-generated content at query #24
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with String field with blank allowed
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result

    # Test with Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test with Integer field allowing null
    int_field_null = Integer(allow_null=True)
    result = to_json_schema(int_field_null)
    assert result["type"] == ["integer", "null"]

    # Test with Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=100.0, exclusive_minimum=0.1, exclusive_maximum=99.9)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 100.0
    assert result["exclusiveMinimum"] == 0.1
    assert result["exclusiveMaximum"] == 99.9

    # Test with Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test with Array field with tuple items
    array_tuple_field = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_tuple_field)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test with Array field with additional items as bool
    array_additional_bool = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] is False

    # Test with Array field with additional items as Field
    array_additional_field = Array(allow_null=False, additional_items=String())
    result = to_json_schema(array_additional_field)
    assert result["additionalItems"]["type"] == "string"

    # Test with Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]

    # Test with Object field with pattern properties
    object_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(object_pattern)
    assert "patternProperties" in result

    # Test with Object field with additional properties as bool
    object_additional_bool = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(object_additional_bool)
    assert result["additionalProperties"] is False

    # Test with Object field with additional properties as Field
    object_additional_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(object_additional_field)
    assert result["additionalProperties"]["type"] == "string"

    # Test with Object field with property names
    object_prop_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_prop_names)
    assert "propertyNames" in result

    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(), Choice(choices=[("a", "A")])])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without else clause
    if_then_field = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with Reference field
    definitions = Definitions()
    ref_field = Reference(to="TestSchema", definitions=definitions)
    result = to_json_schema(ref_field)
    assert "$ref" in result
    


# LLM-generated content at query #25
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"
    
    # Test String field with null
    string_null = String(allow_null=True)
    assert to_json_schema(string_null)["type"] == ["string", "null"]
    
    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False, exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0
    
    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test Boolean with null
    bool_null = Boolean(allow_null=True)
    assert to_json_schema(bool_null)["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    assert "items" in result
    
    # Test Array with tuple items
    array_tuple = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Object field
    obj_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    
    # Test Object with pattern properties
    obj_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    
    # Test Object with additional properties as bool
    obj_additional_bool = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(obj_additional_bool)
    assert result["additionalProperties"] == False
    
    # Test Object with additional properties as Field
    obj_additional_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(obj_additional_field)
    assert isinstance(result["additionalProperties"], dict)
    
    # Test Object with property names
    obj_prop_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_prop_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field with all clauses
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse with only if clause
    if_only = IfThenElse(if_clause=String())
    result = to_json_schema(if_only)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result
    
    # Test Definitions
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    
    # Test Reference field
    reference_field = Reference(to="User", definitions=definitions)
    result = to_json_schema(reference_field)
    assert result["$ref"] == "#/components/schemas/User"
    assert "components" in result
    
    # Test Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]
    
    # Test Array with additional_items as bool
    array_add_items_bool = Array(allow_null=False, items=[String()], additional_items=False)
    result = to_json_schema(array_add_items_bool)
    assert result["additionalItems"] == False
    
    # Test Array with additional_items as Field
    array_add_items_field = Array(allow_null=False, items=[String()], additional_items=Integer())
    result = to_json_schema(array_add_items_field)
    assert isinstance(result["additionalItems"], dict)
    
    # Test with default values preserved
    string_with_default = String(default="test_default")
    result = to_json_schema(string_with_default)
    assert result["type"]


# LLM-generated content at query #26
#--------------------------

```python
def test_to_json_schema():
    # Test Any type
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch type
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert "minLength" not in result
    
    # Test String field with null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String with constraints
    string_constrained = String(min_length=5, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_constrained)
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    
    # Test Float field
    float_field = Float(allow_null=False, exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0
    
    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test Boolean with null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert "items" in result
    
    # Test Array with tuple items
    array_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Array with unique items
    array_unique = Array(items=String(), unique_items=True)
    result = to_json_schema(array_unique)
    assert result["uniqueItems"] == True
    
    # Test Object field
    obj_field = Object(properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test Object with pattern properties
    obj_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    
    # Test Object with additional properties
    obj_additional = Object(additional_properties=String())
    result = to_json_schema(obj_additional)
    assert "additionalProperties" in result
    
    # Test Object with property names
    obj_prop_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_prop_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Choice(choices=[("a", "A")])])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without else clause
    if_then = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String()
    )
    result = to_json_schema(if_then)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with definitions
    definitions = Definitions({"StringType": String()})
    result = to_json_schema(definitions)
    assert "StringType" in result
    
    # Test Reference field
    ref_field = Reference(to="MySchema", definitions=Definitions({"MySchema": String()}))
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert "components" in result
    
    # Test Object with min/max properties
    obj_props = Object(min_properties=1, max_properties=5)
    result = to_json_schema(obj_props)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    
    # Test Schema field
    schema = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]
    
    # Test Array with additional items as boolean
    array_add_items = Array(items=String(), additional_items=False)
    result = to_json_schema(array_add_items)
    assert result["additionalItems"] == False
    
    # Test Array with additional items as field
    array_add_items_field = Array(items=String(), additional_items=Integer())
    result = to_json_schema(array_add_items_field)
    assert isinstance(result["additionalItems"], dict)
    
    # Test Object with additional properties as boolean
    obj_add_props = Object(additional_properties=True)
    result = to_json_schema(obj_add_props)
    assert result["additionalProperties"] == True


# LLM-generated content at query #27
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    assert to_json_schema(Any()) is True
    
    # Test with NeverMatch field
    assert to_json_schema(NeverMatch()) is False
    
    # Test with String field
    string_field = String(allow_null=False, min_length=1, max_length=10)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    
    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test with String field with pattern
    string_pattern = String(pattern_regex=re.compile(r"^[a-z]+$"))
    result = to_json_schema(string_pattern)
    assert result["pattern"] == "^[a-z]+$"
    
    # Test with String field with format
    string_format = String(format="email")
    result = to_json_schema(string_format)
    assert result["format"] == "email"
    
    # Test with Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    
    # Test with Integer field allowing null
    int_field_null = Integer(allow_null=True)
    result = to_json_schema(int_field_null)
    assert result["type"] == ["integer", "null"]
    
    # Test with Float field
    float_field = Float(allow_null=False, exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0
    
    # Test with Float field with multiple_of
    float_multiple = Float(multiple_of=0.5)
    result = to_json_schema(float_multiple)
    assert result["multipleOf"] == 0.5
    
    # Test with Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test with Boolean field allowing null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test with Array field
    array_field = Array(allow_null=False, min_items=1, max_items=10, items=String())
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["items"]["type"] == "string"
    
    # Test with Array field with unique items
    array_unique = Array(unique_items=True, items=Integer())
    result = to_json_schema(array_unique)
    assert result["uniqueItems"] is True
    
    # Test with Array field with tuple items
    array_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test with Array field with additionalItems as bool
    array_additional_bool = Array(additional_items=False, items=String())
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] is False
    
    # Test with Array field with additionalItems as Field
    array_additional_field = Array(additional_items=Integer(), items=String())
    result = to_json_schema(array_additional_field)
    assert result["additionalItems"]["type"] == "integer"
    
    # Test with Object field
    obj_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test with Object field with pattern properties
    obj_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]
    
    # Test with Object field with additional properties as bool
    obj_additional_bool = Object(additional_properties=False)
    result = to_json_schema(obj_additional_bool)
    assert result["additionalProperties"] is False
    
    # Test with Object field with additional properties as Field
    obj_additional_field = Object(additional_properties=String())
    result = to_json_schema(obj_additional_field)
    assert result["additionalProperties"]["type"] == "string"
    
    # Test with Object field with property names
    obj_property_names = Object(property_names=String(pattern_regex=re.compile("^[a-z]+")))
    result = to_json_schema(obj_property_names)
    assert "propertyNames" in result
    
    # Test with Object field with min/max properties
    obj_min_max = Object(min_properties=1, max_properties=5)
    result = to_json_schema(obj_min_max)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    
    # Test with Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test with Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"
    
    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"
    
    # Test with IfThenElse field
    if_then_else = IfThenElse(
        if_clause=Object(properties={"type": Choice(choices=[("a", "A")])}),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test with IfThenElse field without else clause
    if_then = IfThenElse(
        if_clause=Boolean(),
        then_clause=String()
    )
    result = to_json_schema(if_then


# LLM-generated content at query #28
#--------------------------

```python
def test_to_json_schema():
    # Test Any type returns True
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch type returns False
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"
    
    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String field with blank allowed
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result
    
    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0, exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0
    
    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test Boolean field with allow_null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"
    
    # Test Array with tuple items
    array_field_tuple = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Array with additional_items
    array_field_additional = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"] is False
    
    # Test Object field
    obj_field = Object(allow_null=False, properties={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test Object with pattern_properties
    obj_field_pattern = Object(allow_null=False, pattern_properties={"^S_": String()})
    result = to_json_schema(obj_field_pattern)
    assert "patternProperties" in result
    
    # Test Object with additional_properties
    obj_field_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(obj_field_additional)
    assert result["additionalProperties"] is False
    
    # Test Object with property_names
    obj_field_prop_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_field_prop_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without then/else
    if_only_field = IfThenElse(if_clause=String())
    result = to_json_schema(if_only_field)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result
    
    # Test with Definitions
    definitions = Definitions({"TestSchema": String()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    
    # Test Reference field
    ref_field = Reference(to="TestSchema", definitions=Definitions({"TestSchema": String()}))
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/TestSchema"
    
    # Test Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]
    
    # Test invalid field type raises ValueError
    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(object())


# LLM-generated content at query #29
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with String field allowing blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result

    # Test with Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=5, exclusive_maximum=95, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5
    assert result["exclusiveMaximum"] == 95
    assert result["multipleOf"] == 5

    # Test with Integer field allowing null
    integer_field_null = Integer(allow_null=True)
    result = to_json_schema(integer_field_null)
    assert result["type"] == ["integer", "null"]

    # Test with Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=100.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 100.0

    # Test with Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test with Array field with list of items
    array_field_list = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_list)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test with Array field with additional_items as bool
    array_field_additional = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"] is False

    # Test with Array field with additional_items as Field
    array_field_additional_field = Array(allow_null=False, additional_items=String())
    result = to_json_schema(array_field_additional_field)
    assert isinstance(result["additionalItems"], dict)

    # Test with Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]

    # Test with Object field with pattern_properties
    object_field_pattern = Object(allow_null=False, pattern_properties={"^[a-z]+$": String()})
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result

    # Test with Object field with additional_properties as bool
    object_field_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(object_field_additional)
    assert result["additionalProperties"] is False

    # Test with Object field with additional_properties as Field
    object_field_additional_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(object_field_additional_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test with Object field with property_names
    object_field_property_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_field_property_names)
    assert "propertyNames" in result

    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=5)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without then/else
    if_then_else_minimal = IfThenElse(if_clause=String())
    result = to_json_schema(if_then_else_minimal)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with Definitions
    definitions = Definitions({"TestSchema": String()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "TestSchema" in result["components"]["schemas"]

    # Test with Reference field
    ref_field = Reference(to="TestSchema", definitions=Definitions({"TestSchema": String()}))
    result = to_json_schema(ref_fiel


# LLM-generated content at query #30
#--------------------------

```python
def test_to_json_schema():
    # Test Any type
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch type
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String()
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert "default" not in result
    
    # Test String with null
    string_field_nullable = String(allow_null=True)
    result = to_json_schema(string_field_nullable)
    assert result["type"] == ["string", "null"]
    
    # Test String with constraints
    string_field_constrained = String(min_length=2, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_field_constrained)
    assert result["minLength"] == 2
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test Integer field
    integer_field = Integer()
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    
    # Test Integer with constraints
    integer_field_constrained = Integer(minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(integer_field_constrained)
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float()
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test Float with exclusive bounds
    float_field_exclusive = Float(exclusive_minimum=0.0, exclusive_maximum=100.0)
    result = to_json_schema(float_field_exclusive)
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 100.0
    
    # Test Boolean field
    boolean_field = Boolean()
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Boolean with null
    boolean_field_nullable = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_nullable)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array()
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    
    # Test Array with items
    array_field_items = Array(items=String())
    result = to_json_schema(array_field_items)
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"
    
    # Test Array with constraints
    array_field_constrained = Array(min_items=1, max_items=5, unique_items=True)
    result = to_json_schema(array_field_constrained)
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] == True
    
    # Test Object field
    object_field = Object()
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    
    # Test Object with properties
    object_field_props = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(object_field_props)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    
    # Test Object with required fields
    object_field_required = Object(properties={"name": String()}, required=["name"])
    result = to_json_schema(object_field_required)
    assert result["required"] == ["name"]
    
    # Test Object with pattern properties
    object_field_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(object_field_pattern)
    assert "^S_" in result["patternProperties"]
    
    # Test Object with additional properties
    object_field_additional = Object(additional_properties=String())
    result = to_json_schema(object_field_additional)
    assert result["additionalProperties"]["type"] == "string"
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=5)])
    result = to_json_schema(all_of_field)
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert result["if"]["type"] == "string"
    assert result["then"]["type"] == "integer"
    assert result["else"]["type"] == "boolean"
    
    # Test IfThenElse with only if and then
    if_then_field = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with default value
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"
    
    # Test String with format
    string_with_format = String(format="email")
    result = to_json_schema(string_with_format)
    assert result["format"] == "email"
    
    # Test Array with tuple items
    array_tuple_items = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple_items)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Array with additional items as bool
    array_additional_items_bool = Array(additional_items=False)
    result = to_json_schema(array_additional_items_bool)
    assert result["additionalItems"] == False
    
    # Test Array with additional items as Field
    array_additional_items_field = Array(additional_items=String())
    result = to_json_schema(array_additional_items_field)
    assert result["additionalItems"]["type"] == "string"
    
    # Test Object with property_names
    object_property_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_property_names


# LLM-generated content at query #31
#--------------------------

```python
def test_to_json_schema():
    # Test Any() returns True
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch() returns False
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^test")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^test"
    
    # Test String with allow_null
    string_nullable = String(allow_null=True)
    result = to_json_schema(string_nullable)
    assert result["type"] == ["string", "null"]
    
    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    
    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0, multiple_of=0.1)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    assert result["multipleOf"] == 0.1
    
    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Boolean with allow_null
    boolean_nullable = Boolean(allow_null=True)
    result = to_json_schema(boolean_nullable)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=5)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert "items" in result
    
    # Test Array with unique_items
    array_unique = Array(allow_null=False, items=Integer(), unique_items=True)
    result = to_json_schema(array_unique)
    assert result["uniqueItems"] is True
    
    # Test Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test Object with additional_properties
    object_with_additional = Object(
        allow_null=False,
        properties={"id": Integer()},
        additional_properties=True
    )
    result = to_json_schema(object_with_additional)
    assert result["additionalProperties"] is True
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=100)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without else clause
    if_then_field = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with default values
    string_with_default = String(default="test_default")
    result = to_json_schema(string_with_default)
    assert result.get("default") == "test_default"
    
    # Test Reference field
    definitions = {"TestSchema": String()}
    ref_field = Reference(to="TestSchema", definitions=definitions)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/TestSchema"
    
    # Test with Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]
    
    # Test Definitions
    defs = {"StringDef": String(), "IntegerDef": Integer()}
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]
    
    # Test Array with tuple items
    array_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Object with pattern_properties
    object_pattern = Object(
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(object_pattern)
    assert "patternProperties" in result
    
    # Test Object with property_names
    object_prop_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_prop_names)
    assert "propertyNames" in result
    
    # Test exclusive bounds
    exclusive_field = Float(
        exclusive_minimum=0.0,
        exclusive_maximum=100.0
    )
    result = to_json_schema(exclusive_field)
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 100.0


####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    result = from_json_schema_type(
        {
            "minimum": 0,
            "maximum": 100,
            "exclusiveMinimum": 0.5,
            "exclusiveMaximum": 99.5,
            "multipleOf": 5,
            "default": 50,
        },
        type_string="number",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.exclusive_minimum == 0.5
    assert result.exclusive_maximum == 99.5
    assert result.multiple_of == 5
    assert result.default == 50
    assert result.allow_null is False

    # Test number type with allow_null
    result = from_json_schema_type(
        {"default": 10},
        type_string="number",
        allow_null=True,
        definitions=Definitions(),
    )
    assert isinstance(result, Float)
    assert result.allow_null is True

    # Test integer type
    result = from_json_schema_type(
        {
            "minimum": 1,
            "maximum": 10,
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 11,
            "multipleOf": 2,
            "default": 5,
        },
        type_string="integer",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Integer)
    assert result.minimum == 1
    assert result.maximum == 10
    assert result.exclusive_minimum == 0
    assert result.exclusive_maximum == 11
    assert result.multiple_of == 2
    assert result.default == 5

    # Test string type
    result = from_json_schema_type(
        {
            "minLength": 2,
            "maxLength": 50,
            "pattern": "^[a-z]+$",
            "format": "email",
            "default": "test",
        },
        type_string="string",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, String)
    assert result.min_length == 2
    assert result.max_length == 50
    assert result.pattern == "^[a-z]+$"
    assert result.format == "email"
    assert result.default == "test"
    assert result.allow_blank is False

    # Test string type with minLength=0
    result = from_json_schema_type(
        {"minLength": 0},
        type_string="string",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, String)
    assert result.allow_blank is True
    assert result.min_length is None

    # Test string type with minLength=1
    result = from_json_schema_type(
        {"minLength": 1},
        type_string="string",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, String)
    assert result.min_length is None

    # Test boolean type
    result = from_json_schema_type(
        {"default": True},
        type_string="boolean",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Boolean)
    assert result.default is True
    assert result.allow_null is False

    # Test boolean type with allow_null
    result = from_json_schema_type(
        {},
        type_string="boolean",
        allow_null=True,
        definitions=Definitions(),
    )
    assert isinstance(result, Boolean)
    assert result.allow_null is True

    # Test array type with no items
    result = from_json_schema_type(
        {"minItems": 0, "maxItems": 10, "uniqueItems": True},
        type_string="array",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Array)
    assert result.items is None
    assert result.min_items == 0
    assert result.max_items == 10
    assert result.unique_items is True

    # Test array type with items schema
    result = from_json_schema_type(
        {"items": {"type": "string"}, "minItems": 1},
        type_string="array",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Array)
    assert isinstance(result.items, Field)

    # Test array type with items list
    result = from_json_schema_type(
        {"items": [{"type": "string"}, {"type": "number"}]},
        type_string="array",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Array)
    assert isinstance(result.items, list)
    assert len(result.items) == 2

    # Test array type with additionalItems as boolean
    result = from_json_schema_type(
        {"additionalItems": False},
        type_string="array",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Array)
    assert result.additional_items is False

    # Test array type with additionalItems as schema
    result = from_json_schema_type(
        {"additionalItems": {"type": "string"}},
        type_string="array",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Array)
    assert isinstance(result.additional_items, Field)

    # Test object type with no properties
    result = from_json_schema_type(
        {},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert result.properties is None

    # Test object type with properties
    result = from_json_schema_type(
        {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert isinstance(result.properties, dict)
    assert "name" in result.properties
    assert "age" in result.properties

    # Test object type with patternProperties
    result = from_json_schema_type(
        {"patternProperties": {"^S_": {"type": "string"}, "^I_": {"type": "integer"}}},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert isinstance(result.pattern_properties, dict)
    assert "^S_" in result.pattern_properties

    # Test object type with additionalProperties as boolean
    result = from_json_schema_type(
        {"additionalProperties": False},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert result.additional_properties is False

    # Test object type with additionalProperties as schema
    result = from_json_schema_type(
        {"additionalProperties": {"type": "string"}},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert isinstance(result.additional_properties, Field)

    # Test object type with propertyNames
    result = from_json_schema_type(
        {"propertyNames": {"pattern": "^[a-z]+$"}},
        type_string="object",
        allow_null=False,
        definitions=Definitions(),
    )
    assert isinstance(result, Object)
    assert isinstance(result.property_names, Field)

    # Test object type with required fields
    result = from_json_schema_type(
        {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "minProperties": 1,
            "maxProperties": 5,
        },
        type_string="object",
        allow_null=


# LLM-generated content at query #2
#--------------------------

```python
def test_ref_from_json_schema():
    # Test basic reference resolution
    defs = Definitions()
    data = {"$ref": "#/components/schemas/MySchema"}
    result = ref_from_json_schema(data, defs)
    
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/MySchema"
    assert result.definitions is defs


def test_ref_from_json_schema_with_definitions():
    # Test reference with pre-populated definitions
    defs = Definitions()
    defs["#/components/schemas/User"] = Object(properties={"name": String()})
    
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, defs)
    
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"
    assert "#/components/schemas/User" in defs


def test_ref_from_json_schema_invalid_ref():
    # Test that unsupported $ref style raises assertion error
    defs = Definitions()
    data = {"$ref": "external.json#/definitions/MySchema"}
    
    with pytest.raises(AssertionError, match="Unsupported \\$ref style"):
        ref_from_json_schema(data, defs)


def test_ref_from_json_schema_nested_path():
    # Test reference with nested path
    defs = Definitions()
    data = {"$ref": "#/definitions/components/schemas/ComplexSchema"}
    result = ref_from_json_schema(data, defs)
    
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/components/schemas/ComplexSchema"


def test_ref_from_json_schema_simple_definition():
    # Test simple definition reference
    defs = Definitions()
    data = {"$ref": "#/definitions/String"}
    result = ref_from_json_schema(data, defs)
    
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/String"
    assert result.definitions is defs


# LLM-generated content at query #3
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    result = from_json_schema_type(
        {"minimum": 0, "maximum": 100, "multipleOf": 5},
        type_string="number",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.multiple_of == 5
    assert result.allow_null is False

    # Test number type with allow_null
    result = from_json_schema_type(
        {"minimum": -10, "maximum": 10},
        type_string="number",
        allow_null=True,
        definitions=Definitions()
    )
    assert isinstance(result, Float)
    assert result.allow_null is True

    # Test integer type
    result = from_json_schema_type(
        {"minimum": 1, "maximum": 100},
        type_string="integer",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Integer)
    assert result.minimum == 1
    assert result.maximum == 100

    # Test integer with exclusive bounds
    result = from_json_schema_type(
        {"exclusiveMinimum": 0, "exclusiveMaximum": 100},
        type_string="integer",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Integer)
    assert result.exclusive_minimum == 0
    assert result.exclusive_maximum == 100

    # Test string type
    result = from_json_schema_type(
        {"minLength": 1, "maxLength": 50, "pattern": "^[a-z]+$"},
        type_string="string",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, String)
    assert result.min_length == 1
    assert result.max_length == 50
    assert result.pattern == "^[a-z]+$"

    # Test string with minLength 0
    result = from_json_schema_type(
        {"minLength": 0},
        type_string="string",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, String)
    assert result.allow_blank is True
    assert result.min_length is None

    # Test string with format
    result = from_json_schema_type(
        {"format": "email"},
        type_string="string",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, String)
    assert result.format == "email"

    # Test boolean type
    result = from_json_schema_type(
        {},
        type_string="boolean",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Boolean)
    assert result.allow_null is False

    # Test boolean with allow_null
    result = from_json_schema_type(
        {},
        type_string="boolean",
        allow_null=True,
        definitions=Definitions()
    )
    assert isinstance(result, Boolean)
    assert result.allow_null is True

    # Test array type with no items
    result = from_json_schema_type(
        {},
        type_string="array",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Array)
    assert result.items is None

    # Test array type with single item schema
    result = from_json_schema_type(
        {"items": {"type": "string"}, "minItems": 1, "maxItems": 10, "uniqueItems": True},
        type_string="array",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Array)
    assert isinstance(result.items, Field)
    assert result.min_items == 1
    assert result.max_items == 10
    assert result.unique_items is True

    # Test array type with tuple validation (items as list)
    result = from_json_schema_type(
        {"items": [{"type": "string"}, {"type": "integer"}]},
        type_string="array",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Array)
    assert isinstance(result.items, list)
    assert len(result.items) == 2

    # Test array with additionalItems as boolean
    result = from_json_schema_type(
        {"additionalItems": False},
        type_string="array",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Array)
    assert result.additional_items is False

    # Test array with additionalItems as schema
    result = from_json_schema_type(
        {"additionalItems": {"type": "number"}},
        type_string="array",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Array)
    assert isinstance(result.additional_items, Field)

    # Test object type with no properties
    result = from_json_schema_type(
        {},
        type_string="object",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Object)
    assert result.properties is None

    # Test object type with properties
    result = from_json_schema_type(
        {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
        type_string="object",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Object)
    assert result.properties is not None
    assert "name" in result.properties
    assert "age" in result.properties

    # Test object with pattern properties
    result = from_json_schema_type(
        {"patternProperties": {"^S_": {"type": "string"}}},
        type_string="object",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Object)
    assert result.pattern_properties is not None
    assert "^S_" in result.pattern_properties

    # Test object with additionalProperties as boolean
    result = from_json_schema_type(
        {"additionalProperties": False},
        type_string="object",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Object)
    assert result.additional_properties is False

    # Test object with additionalProperties as schema
    result = from_json_schema_type(
        {"additionalProperties": {"type": "string"}},
        type_string="object",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Object)
    assert isinstance(result.additional_properties, Field)

    # Test object with propertyNames
    result = from_json_schema_type(
        {"propertyNames": {"pattern": "^[a-z_]+$"}},
        type_string="object",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Object)
    assert result.property_names is not None

    # Test object with minProperties and maxProperties
    result = from_json_schema_type(
        {"minProperties": 1, "maxProperties": 5},
        type_string="object",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Object)
    assert result.min_properties == 1
    assert result.max_properties == 5

    # Test object with required fields
    result = from_json_schema_type(
        {"required": ["name", "age"]},
        type_string="object",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, Object)
    assert result.required == ["name", "age"]

    # Test with default value
    result = from_json_schema_type(
        {"default": "test_value"},
        type_string="string",
        allow_null=False,
        definitions=Definitions()
    )
    assert isinstance(result, String)
    assert result.default == "test


# LLM-generated content at query #4
#--------------------------

```python
def test_enum_from_json_schema():
    # Test basic enum with simple values
    data = {"enum": [1, 2, 3]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1, 1), (2, 2), (3, 3)]
    assert result.default is NO_DEFAULT

    # Test enum with string values
    data = {"enum": ["red", "green", "blue"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("red", "red"), ("green", "green"), ("blue", "blue")]

    # Test enum with mixed types
    data = {"enum": [1, "two", None, True]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1, 1), ("two", "two"), (None, None), (True, True)]

    # Test enum with default value
    data = {"enum": [10, 20, 30], "default": 20}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(10, 10), (20, 20), (30, 30)]
    assert result.default == 20

    # Test enum with single value
    data = {"enum": ["only"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("only", "only")]

    # Test enum with empty string
    data = {"enum": ["", "a", "b"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("", ""), ("a", "a"), ("b", "b")]

    # Test enum with numeric zero
    data = {"enum": [0, 1, 2]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(0, 0), (1, 1), (2, 2)]

    # Test enum with False value
    data = {"enum": [False, True]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(False, False), (True, True)]

    # Test enum with float values
    data = {"enum": [1.5, 2.5, 3.5]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1.5, 1.5), (2.5, 2.5), (3.5, 3.5)]


# LLM-generated content at query #5
#--------------------------

```python
def test_get_valid_types():
    # Test with single type string
    type_strings, allow_null = get_valid_types({"type": "string"})
    assert type_strings == {"string"}
    assert allow_null is False

    # Test with multiple type strings as list
    type_strings, allow_null = get_valid_types({"type": ["string", "integer"]})
    assert type_strings == {"string", "integer"}
    assert allow_null is False

    # Test with null type included
    type_strings, allow_null = get_valid_types({"type": ["string", "null"]})
    assert type_strings == {"string"}
    assert allow_null is True

    # Test with only null type
    type_strings, allow_null = get_valid_types({"type": "null"})
    assert type_strings == set()
    assert allow_null is True

    # Test with no type specified (defaults to all types)
    type_strings, allow_null = get_valid_types({})
    assert type_strings == {"boolean", "object", "array", "number", "string"}
    assert allow_null is False

    # Test with number and integer (integer should be removed)
    type_strings, allow_null = get_valid_types({"type": ["number", "integer"]})
    assert type_strings == {"number"}
    assert allow_null is False

    # Test with number, integer, and null
    type_strings, allow_null = get_valid_types({"type": ["number", "integer", "null"]})
    assert type_strings == {"number"}
    assert allow_null is True

    # Test with empty type list (defaults to all types)
    type_strings, allow_null = get_valid_types({"type": []})
    assert type_strings == {"boolean", "object", "array", "number", "string"}
    assert allow_null is False

    # Test with object type
    type_strings, allow_null = get_valid_types({"type": "object"})
    assert type_strings == {"object"}
    assert allow_null is False

    # Test with array type and null
    type_strings, allow_null = get_valid_types({"type": ["array", "null"]})
    assert type_strings == {"array"}
    assert allow_null is True

    # Test with boolean type
    type_strings, allow_null = get_valid_types({"type": "boolean"})
    assert type_strings == {"boolean"}
    assert allow_null is False


# LLM-generated content at query #6
#--------------------------

```python
def test_any_of_from_json_schema():
    # Test basic any_of with multiple types
    data = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result = any_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Integer)

    # Test any_of with default value
    data_with_default = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test_default"
    }
    result_with_default = any_of_from_json_schema(data_with_default, definitions=Definitions())
    assert isinstance(result_with_default, Union)
    assert result_with_default.default == "test_default"

    # Test any_of with complex schemas
    data_complex = {
        "anyOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "array", "items": {"type": "integer"}}
        ]
    }
    result_complex = any_of_from_json_schema(data_complex, definitions=Definitions())
    assert isinstance(result_complex, Union)
    assert len(result_complex.any_of) == 2
    assert isinstance(result_complex.any_of[0], Object)
    assert isinstance(result_complex.any_of[1], Array)

    # Test any_of with single item
    data_single = {
        "anyOf": [
            {"type": "boolean"}
        ]
    }
    result_single = any_of_from_json_schema(data_single, definitions=Definitions())
    assert isinstance(result_single, Union)
    assert len(result_single.any_of) == 1
    assert isinstance(result_single.any_of[0], Boolean)

    # Test any_of without default
    data_no_default = {
        "anyOf": [
            {"type": "string"},
            {"type": "null"}
        ]
    }
    result_no_default = any_of_from_json_schema(data_no_default, definitions=Definitions())
    assert isinstance(result_no_default, Union)
    assert result_no_default.default == NO_DEFAULT


# LLM-generated content at query #7
#--------------------------

```python
def test_enum_from_json_schema():
    # Test basic enum with string values
    data = {"enum": ["red", "green", "blue"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("red", "red"), ("green", "green"), ("blue", "blue")]
    assert result.default == NO_DEFAULT

    # Test enum with integer values
    data = {"enum": [1, 2, 3]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1, 1), (2, 2), (3, 3)]
    assert result.default == NO_DEFAULT

    # Test enum with mixed types
    data = {"enum": ["a", 1, True, None]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), (1, 1), (True, True), (None, None)]
    assert result.default == NO_DEFAULT

    # Test enum with default value
    data = {"enum": ["option1", "option2", "option3"], "default": "option2"}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("option1", "option1"), ("option2", "option2"), ("option3", "option3")]
    assert result.default == "option2"

    # Test enum with single value
    data = {"enum": ["only_option"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("only_option", "only_option")]
    assert result.default == NO_DEFAULT

    # Test enum with boolean values
    data = {"enum": [True, False]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(True, True), (False, False)]
    assert result.default == NO_DEFAULT

    # Test enum with float values
    data = {"enum": [1.5, 2.5, 3.5]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1.5, 1.5), (2.5, 2.5), (3.5, 3.5)]
    assert result.default == NO_DEFAULT

    # Test enum with null value
    data = {"enum": [None, "value"], "default": None}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(None, None), ("value", "value")]
    assert result.default == None


# LLM-generated content at query #8
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"},
        "default": "test_default"
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    assert result.default == "test_default"

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None

    # Test with only if clause
    data = {
        "if": {"type": "string"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with no default
    data = {
        "if": {"type": "number"},
        "then": {"type": "string"},
        "else": {"type": "boolean"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == NO_DEFAULT

    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "number"}},
        "else": {"enum": [1, 2, 3]},
        "default": 42
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    assert result.default == 42


# LLM-generated content at query #9
#--------------------------

```python
def test_enum_from_json_schema():
    # Test basic enum with simple values
    data = {"enum": [1, 2, 3]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1, 1), (2, 2), (3, 3)]
    assert result.default is NO_DEFAULT

    # Test enum with string values
    data = {"enum": ["red", "green", "blue"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("red", "red"), ("green", "green"), ("blue", "blue")]
    assert result.default is NO_DEFAULT

    # Test enum with mixed types
    data = {"enum": [1, "two", 3.0, True, None]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(1, 1), ("two", "two"), (3.0, 3.0), (True, True), (None, None)]
    assert result.default is NO_DEFAULT

    # Test enum with default value
    data = {"enum": ["a", "b", "c"], "default": "b"}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]
    assert result.default == "b"

    # Test enum with single value
    data = {"enum": [42]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(42, 42)]
    assert result.default is NO_DEFAULT

    # Test enum with boolean values
    data = {"enum": [True, False]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(True, True), (False, False)]

    # Test enum with numeric strings
    data = {"enum": ["1", "2", "3"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [("1", "1"), ("2", "2"), ("3", "3")]

    # Test enum with default value not in enum list
    data = {"enum": [10, 20, 30], "default": 20}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.default == 20

    # Test enum with None value
    data = {"enum": [None, "value"]}
    result = enum_from_json_schema(data, definitions)
    assert isinstance(result, Choice)
    assert result.choices == [(None, None), ("value", "value")]


# LLM-generated content at query #10
#--------------------------

```python
def test_all_of_from_json_schema():
    # Test basic allOf with multiple constraints
    data = {
        "allOf": [
            {"type": "string", "minLength": 1},
            {"type": "string", "maxLength": 10}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], String)
    assert isinstance(result.all_of[1], String)

    # Test allOf with default value
    data_with_default = {
        "allOf": [
            {"type": "integer", "minimum": 0},
            {"type": "integer", "maximum": 100}
        ],
        "default": 50
    }
    result = all_of_from_json_schema(data_with_default, definitions)
    assert isinstance(result, AllOf)
    assert result.default == 50

    # Test allOf with single constraint
    data_single = {
        "allOf": [
            {"type": "number", "multipleOf": 2}
        ]
    }
    result = all_of_from_json_schema(data_single, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 1

    # Test allOf with object schemas
    data_objects = {
        "allOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "object", "required": ["name"]}
        ]
    }
    result = all_of_from_json_schema(data_objects, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], Object)
    assert isinstance(result.all_of[1], Object)

    # Test allOf without default
    data_no_default = {
        "allOf": [
            {"type": "boolean"}
        ]
    }
    result = all_of_from_json_schema(data_no_default, definitions)
    assert isinstance(result, AllOf)
    assert result.default is NO_DEFAULT

    # Test allOf with complex nested schemas
    data_complex = {
        "allOf": [
            {"type": "array", "items": {"type": "string"}},
            {"minItems": 1}
        ]
    }
    result = all_of_from_json_schema(data_complex, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2


# LLM-generated content at query #11
#--------------------------

```python
def test_all_of_from_json_schema():
    # Test basic allOf with simple constraints
    data = {
        "allOf": [
            {"type": "string", "minLength": 1},
            {"maxLength": 10}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2

    # Test allOf with default value
    data_with_default = {
        "allOf": [
            {"type": "integer", "minimum": 0},
            {"maximum": 100}
        ],
        "default": 50
    }
    result = all_of_from_json_schema(data_with_default, definitions)
    assert isinstance(result, AllOf)
    assert result.default == 50

    # Test allOf with single constraint
    data_single = {
        "allOf": [
            {"type": "boolean"}
        ]
    }
    result = all_of_from_json_schema(data_single, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 1

    # Test allOf with multiple complex constraints
    data_complex = {
        "allOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"required": ["name"]},
            {"maxProperties": 5}
        ]
    }
    result = all_of_from_json_schema(data_complex, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 3

    # Test allOf with no default (should use NO_DEFAULT)
    data_no_default = {
        "allOf": [
            {"type": "number"},
            {"minimum": 0.0}
        ]
    }
    result = all_of_from_json_schema(data_no_default, definitions)
    assert isinstance(result, AllOf)
    assert result.default is NO_DEFAULT

    # Test allOf with nested references
    test_definitions = Definitions()
    test_definitions["#/components/schemas/StringType"] = String()
    data_with_ref = {
        "allOf": [
            {"$ref": "#/components/schemas/StringType"},
            {"minLength": 5}
        ]
    }
    result = all_of_from_json_schema(data_with_ref, test_definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2


# LLM-generated content at query #12
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True, min_length=None)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result

    # Test with Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=1, exclusive_maximum=99, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 1
    assert result["exclusiveMaximum"] == 99
    assert result["multipleOf"] == 5

    # Test with Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test with Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test with Array field with tuple of items
    array_field_tuple = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test with Array field with additional_items as bool
    array_field_additional = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"] is False

    # Test with Array field with additional_items as Field
    array_field_additional_field = Array(allow_null=False, additional_items=String())
    result = to_json_schema(array_field_additional_field)
    assert isinstance(result["additionalItems"], dict)
    assert result["additionalItems"]["type"] == "string"

    # Test with Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]

    # Test with Object field with pattern_properties
    object_field_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result

    # Test with Object field with additional_properties as bool
    object_field_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(object_field_additional)
    assert result["additionalProperties"] is False

    # Test with Object field with additional_properties as Field
    object_field_additional_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(object_field_additional_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test with Object field with property_names
    object_field_property_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_field_property_names)
    assert "propertyNames" in result

    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(), String(min_length=5)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without else clause
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=None
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]

    # Test with Definitions
    definitions = Definitions({"User": Object(properties={"


# LLM-generated content at query #13
#--------------------------

```python
def test_one_of_from_json_schema():
    # Test basic oneOf with simple types
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result = one_of_from_json_schema(data, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Integer)

    # Test oneOf with default value
    data_with_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test"
    }
    result_with_default = one_of_from_json_schema(data_with_default, definitions)
    assert isinstance(result_with_default, OneOf)
    assert result_with_default.default == "test"

    # Test oneOf with complex schemas
    data_complex = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            {
                "type": "array",
                "items": {"type": "integer"}
            }
        ]
    }
    result_complex = one_of_from_json_schema(data_complex, definitions)
    assert isinstance(result_complex, OneOf)
    assert len(result_complex.one_of) == 2
    assert isinstance(result_complex.one_of[0], Object)
    assert isinstance(result_complex.one_of[1], Array)

    # Test oneOf with single item
    data_single = {
        "oneOf": [
            {"type": "boolean"}
        ]
    }
    result_single = one_of_from_json_schema(data_single, definitions)
    assert isinstance(result_single, OneOf)
    assert len(result_single.one_of) == 1
    assert isinstance(result_single.one_of[0], Boolean)

    # Test oneOf with nested references
    nested_definitions = Definitions()
    nested_definitions["#/definitions/StringType"] = String()
    data_nested = {
        "oneOf": [
            {"$ref": "#/definitions/StringType"},
            {"type": "null"}
        ]
    }
    result_nested = one_of_from_json_schema(data_nested, nested_definitions)
    assert isinstance(result_nested, OneOf)
    assert len(result_nested.one_of) == 2
    assert isinstance(result_nested.one_of[0], Reference)

    # Test oneOf without default
    data_no_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result_no_default = one_of_from_json_schema(data_no_default, definitions)
    assert result_no_default.default is NO_DEFAULT


# LLM-generated content at query #14
#--------------------------

```python
def test_one_of_from_json_schema():
    # Test basic oneOf with simple types
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    result = one_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Integer)

    # Test oneOf with default value
    data_with_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test"
    }
    result_with_default = one_of_from_json_schema(data_with_default, definitions=Definitions())
    assert isinstance(result_with_default, OneOf)
    assert result_with_default.default == "test"

    # Test oneOf with complex objects
    data_complex = {
        "oneOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "array", "items": {"type": "integer"}}
        ]
    }
    result_complex = one_of_from_json_schema(data_complex, definitions=Definitions())
    assert isinstance(result_complex, OneOf)
    assert len(result_complex.one_of) == 2
    assert isinstance(result_complex.one_of[0], Object)
    assert isinstance(result_complex.one_of[1], Array)

    # Test oneOf with single constraint
    data_single = {
        "oneOf": [
            {"type": "boolean"}
        ]
    }
    result_single = one_of_from_json_schema(data_single, definitions=Definitions())
    assert isinstance(result_single, OneOf)
    assert len(result_single.one_of) == 1
    assert isinstance(result_single.one_of[0], Boolean)

    # Test oneOf with references
    defs = Definitions()
    defs["#/definitions/StringType"] = String()
    data_with_ref = {
        "oneOf": [
            {"$ref": "#/definitions/StringType"},
            {"type": "integer"}
        ]
    }
    result_with_ref = one_of_from_json_schema(data_with_ref, definitions=defs)
    assert isinstance(result_with_ref, OneOf)
    assert len(result_with_ref.one_of) == 2
    assert isinstance(result_with_ref.one_of[0], Reference)
    assert isinstance(result_with_ref.one_of[1], Integer)

    # Test oneOf without default
    data_no_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result_no_default = one_of_from_json_schema(data_no_default, definitions=Definitions())
    assert result_no_default.default is NO_DEFAULT

    # Test oneOf with enum and const
    data_enum_const = {
        "oneOf": [
            {"enum": [1, 2, 3]},
            {"const": "fixed"}
        ]
    }
    result_enum_const = one_of_from_json_schema(data_enum_const, definitions=Definitions())
    assert isinstance(result_enum_const, OneOf)
    assert len(result_enum_const.one_of) == 2
    assert isinstance(result_enum_const.one_of[0], Choice)
    assert isinstance(result_enum_const.one_of[1], Const)


# LLM-generated content at query #15
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None

    # Test with only if clause
    data = {
        "if": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": "test_default",
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == "test_default"

    # Test with nested complex schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "integer"}},
        "else": {"enum": [1, 2, 3]},
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_if_then_else_from_json_schema():
    """Test if_then_else_from_json_schema function."""
    defs = Definitions()
    
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    
    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None
    
    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None
    
    # Test with only if clause
    data = {
        "if": {"type": "string"},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None
    
    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "default": 42,
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.default == 42
    
    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "integer"}},
        "else": {"enum": [None, "unknown"]},
    }
    result = if_then_else_from_json_schema(data, defs)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None


# LLM-generated content at query #17
#--------------------------

```python
def test_if_then_else_from_json_schema():
    """Test if_then_else_from_json_schema function."""
    
    # Test basic if-then-else structure
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    
    # Test if-then without else
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None
    
    # Test if-else without then
    data = {
        "if": {"type": "string"},
        "else": {"type": "number"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None
    
    # Test if only
    data = {
        "if": {"type": "string"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None
    
    # Test with default value
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"},
        "default": "test_default"
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == "test_default"
    
    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"properties": {"age": {"type": "integer"}}},
        "else": {"type": "array", "items": {"type": "string"}}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    
    # Test with boolean schemas
    data = {
        "if": True,
        "then": False,
        "else": True
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)


# LLM-generated content at query #18
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with if, then, and else clauses
    data = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"},
        "default": "test"
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    assert result.default == "test"

    # Test with only if and then clauses
    data_then_only = {
        "if": {"type": "object"},
        "then": {"properties": {"name": {"type": "string"}}}
    }
    result_then_only = if_then_else_from_json_schema(data_then_only, definitions)
    assert isinstance(result_then_only, IfThenElse)
    assert result_then_only.if_clause is not None
    assert result_then_only.then_clause is not None
    assert result_then_only.else_clause is None

    # Test with only if and else clauses
    data_else_only = {
        "if": {"type": "array"},
        "else": {"type": "string"}
    }
    result_else_only = if_then_else_from_json_schema(data_else_only, definitions)
    assert isinstance(result_else_only, IfThenElse)
    assert result_else_only.if_clause is not None
    assert result_else_only.then_clause is None
    assert result_else_only.else_clause is not None

    # Test with only if clause
    data_if_only = {
        "if": {"type": "boolean"}
    }
    result_if_only = if_then_else_from_json_schema(data_if_only, definitions)
    assert isinstance(result_if_only, IfThenElse)
    assert result_if_only.if_clause is not None
    assert result_if_only.then_clause is None
    assert result_if_only.else_clause is None

    # Test with nested conditions
    data_nested = {
        "if": {"properties": {"type": {"const": "admin"}}},
        "then": {"required": ["permissions"]},
        "else": {"required": ["username"]},
        "default": None
    }
    result_nested = if_then_else_from_json_schema(data_nested, definitions)
    assert isinstance(result_nested, IfThenElse)
    assert result_nested.default is None

    # Test with complex schema references
    test_definitions = Definitions()
    data_with_ref = {
        "if": {"$ref": "#/definitions/StringType"},
        "then": {"$ref": "#/definitions/LongString"},
        "else": {"$ref": "#/definitions/NumberType"}
    }
    result_ref = if_then_else_from_json_schema(data_with_ref, test_definitions)
    assert isinstance(result_ref, IfThenElse)
    assert result_ref.if_clause is not None
    assert result_ref.then_clause is not None
    assert result_ref.else_clause is not None


# LLM-generated content at query #19
#--------------------------

```python
def test_from_json_schema():
    # Test with boolean schema - True
    result = from_json_schema(True)
    assert isinstance(result, Any)
    
    # Test with boolean schema - False
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)
    
    # Test with empty dict
    result = from_json_schema({})
    assert isinstance(result, Any)
    
    # Test with type constraint
    result = from_json_schema({"type": "string"})
    assert isinstance(result, String)
    
    result = from_json_schema({"type": "integer"})
    assert isinstance(result, Integer)
    
    result = from_json_schema({"type": "number"})
    assert isinstance(result, Number)
    
    result = from_json_schema({"type": "boolean"})
    assert isinstance(result, Boolean)
    
    result = from_json_schema({"type": "array"})
    assert isinstance(result, Array)
    
    result = from_json_schema({"type": "object"})
    assert isinstance(result, Object)
    
    # Test with enum
    result = from_json_schema({"enum": [1, 2, 3]})
    assert isinstance(result, Choice)
    
    # Test with const
    result = from_json_schema({"const": "value"})
    assert isinstance(result, Const)
    
    # Test with allOf
    result = from_json_schema({"allOf": [{"type": "string"}, {"minLength": 1}]})
    assert isinstance(result, AllOf)
    
    # Test with anyOf
    result = from_json_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(result, Union)
    
    # Test with oneOf
    result = from_json_schema({"oneOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(result, OneOf)
    
    # Test with not
    result = from_json_schema({"not": {"type": "string"}})
    assert isinstance(result, Not)
    
    # Test with if-then-else
    result = from_json_schema(
        {"if": {"type": "string"}, "then": {"minLength": 1}}
    )
    assert isinstance(result, IfThenElse)
    
    # Test with $ref
    defs = Definitions()
    defs["#/components/schemas/MySchema"] = String()
    result = from_json_schema({"$ref": "#/components/schemas/MySchema"}, definitions=defs)
    assert isinstance(result, Reference)
    
    # Test with multiple constraints
    result = from_json_schema(
        {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
            "pattern": "^[a-z]+$"
        }
    )
    assert isinstance(result, (String, AllOf))
    
    # Test with definitions in components
    schema_with_defs = {
        "components": {
            "schemas": {
                "MyType": {"type": "string"}
            }
        }
    }
    result = from_json_schema(schema_with_defs)
    assert isinstance(result, Any)
    
    # Test with string constraints
    result = from_json_schema(
        {
            "type": "string",
            "minLength": 5,
            "maxLength": 10,
            "pattern": "^test"
        }
    )
    assert isinstance(result, (String, AllOf))
    
    # Test with numeric constraints
    result = from_json_schema(
        {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "multipleOf": 5
        }
    )
    assert isinstance(result, (Number, AllOf))
    
    # Test with array constraints
    result = from_json_schema(
        {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
            "uniqueItems": True
        }
    )
    assert isinstance(result, (Array, AllOf))
    
    # Test with object constraints
    result = from_json_schema(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
    )
    assert isinstance(result, (Object, AllOf))


# LLM-generated content at query #20
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"},
        "default": "test_default"
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is not None
    assert result.default == "test_default"

    # Test with only if and then clauses
    data = {
        "if": {"type": "object"},
        "then": {"type": "array"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is not None
    assert result.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "number"},
        "else": {"type": "string"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is not None

    # Test with only if clause
    data = {
        "if": {"type": "boolean"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.if_clause is not None
    assert result.then_clause is None
    assert result.else_clause is None

    # Test with no default
    data = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert result.default == NO_DEFAULT

    # Test with complex nested schemas
    data = {
        "if": {"type": "object", "properties": {"name": {"type": "string"}}},
        "then": {"type": "array", "items": {"type": "integer"}},
        "else": {"enum": [1, 2, 3]},
        "default": None
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default is None


# LLM-generated content at query #21
#--------------------------

```python
def test_from_json_schema():
    # Test with boolean schema - True
    result = from_json_schema(True)
    assert isinstance(result, Any)

    # Test with boolean schema - False
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

    # Test with simple type constraint
    result = from_json_schema({"type": "string"})
    assert isinstance(result, String)

    # Test with integer type
    result = from_json_schema({"type": "integer"})
    assert isinstance(result, Integer)

    # Test with number type
    result = from_json_schema({"type": "number"})
    assert isinstance(result, Number)

    # Test with boolean type
    result = from_json_schema({"type": "boolean"})
    assert isinstance(result, Boolean)

    # Test with array type
    result = from_json_schema({"type": "array"})
    assert isinstance(result, Array)

    # Test with object type
    result = from_json_schema({"type": "object"})
    assert isinstance(result, Object)

    # Test with enum constraint
    result = from_json_schema({"enum": [1, 2, 3]})
    assert isinstance(result, Choice)

    # Test with const constraint
    result = from_json_schema({"const": "fixed_value"})
    assert isinstance(result, Const)

    # Test with string constraints
    result = from_json_schema({"type": "string", "minLength": 1, "maxLength": 10})
    assert isinstance(result, AllOf)

    # Test with numeric constraints
    result = from_json_schema({"type": "number", "minimum": 0, "maximum": 100})
    assert isinstance(result, AllOf)

    # Test with array constraints
    result = from_json_schema({"type": "array", "minItems": 1, "maxItems": 5})
    assert isinstance(result, AllOf)

    # Test with object constraints
    result = from_json_schema({"type": "object", "minProperties": 1})
    assert isinstance(result, AllOf)

    # Test with pattern constraint
    result = from_json_schema({"type": "string", "pattern": "^[a-z]+$"})
    assert isinstance(result, AllOf)

    # Test with allOf composition
    result = from_json_schema({"allOf": [{"type": "string"}, {"minLength": 1}]})
    assert isinstance(result, (AllOf, Field))

    # Test with oneOf composition
    result = from_json_schema({"oneOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(result, (OneOf, Field))

    # Test with not constraint
    result = from_json_schema({"not": {"type": "null"}})
    assert isinstance(result, (Not, Field))

    # Test with if-then-else constraint
    result = from_json_schema({"if": {"type": "string"}, "then": {"minLength": 1}})
    assert isinstance(result, (IfThenElse, Field))

    # Test with empty object
    result = from_json_schema({})
    assert isinstance(result, Any)

    # Test with $ref
    test_defs = Definitions()
    test_defs["#/components/schemas/Test"] = String()
    result = from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=test_defs)
    assert isinstance(result, Reference)

    # Test with definitions in data
    schema_data = {
        "type": "object",
        "components": {
            "schemas": {
                "TestSchema": {"type": "string"}
            }
        }
    }
    result = from_json_schema(schema_data)
    assert isinstance(result, (Object, Field))

    # Test with multiple constraints
    result = from_json_schema({
        "type": "string",
        "minLength": 1,
        "pattern": "^[a-z]+$",
        "enum": ["abc", "def"]
    })
    assert isinstance(result, AllOf)

    # Test with nested object schema
    result = from_json_schema({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    })
    assert isinstance(result, AllOf)

    # Test with nested array schema
    result = from_json_schema({
        "type": "array",
        "items": {"type": "string"}
    })
    assert isinstance(result, AllOf)

    # Test with anyOf composition
    result = from_json_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(result, (Union, Field))


# LLM-generated content at query #22
#--------------------------

```python
def test_from_json_schema():
    # Test with boolean schema - True
    result = from_json_schema(True)
    assert isinstance(result, Any)

    # Test with boolean schema - False
    result = from_json_schema(False)
    assert isinstance(result, NeverMatch)

    # Test with simple type constraint
    result = from_json_schema({"type": "string"})
    assert isinstance(result, String)

    # Test with enum
    result = from_json_schema({"enum": [1, 2, 3]})
    assert isinstance(result, Choice)

    # Test with const
    result = from_json_schema({"const": "value"})
    assert isinstance(result, Const)

    # Test with type and enum combined
    result = from_json_schema({"type": "string", "enum": ["a", "b"]})
    assert isinstance(result, AllOf)

    # Test with $ref
    defs = Definitions()
    defs["#/components/schemas/TestSchema"] = String()
    result = from_json_schema({"$ref": "#/components/schemas/TestSchema"}, definitions=defs)
    assert isinstance(result, Reference)

    # Test with allOf
    result = from_json_schema({"allOf": [{"type": "string"}, {"minLength": 1}]})
    assert isinstance(result, AllOf)

    # Test with anyOf
    result = from_json_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(result, Union)

    # Test with oneOf
    result = from_json_schema({"oneOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(result, OneOf)

    # Test with not
    result = from_json_schema({"not": {"type": "string"}})
    assert isinstance(result, Not)

    # Test with if-then-else
    result = from_json_schema(
        {
            "if": {"type": "string"},
            "then": {"minLength": 1},
            "else": {"type": "integer"},
        }
    )
    assert isinstance(result, IfThenElse)

    # Test with empty object
    result = from_json_schema({})
    assert isinstance(result, Any)

    # Test with definitions in components
    result = from_json_schema(
        {
            "components": {
                "schemas": {
                    "StringType": {"type": "string"},
                    "IntType": {"type": "integer"},
                }
            }
        }
    )
    assert isinstance(result, Any)

    # Test with nested properties
    result = from_json_schema(
        {"type": "object", "properties": {"name": {"type": "string"}}}
    )
    assert isinstance(result, Object)

    # Test with array items
    result = from_json_schema({"type": "array", "items": {"type": "string"}})
    assert isinstance(result, Array)

    # Test with multiple constraints
    result = from_json_schema(
        {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
            "pattern": "^[a-z]+$",
        }
    )
    assert isinstance(result, String)

    # Test with numeric constraints
    result = from_json_schema(
        {"type": "number", "minimum": 0, "maximum": 100, "multipleOf": 5}
    )
    assert isinstance(result, Number)

    # Test with custom definitions
    custom_defs = Definitions()
    custom_defs["CustomRef"] = Integer()
    result = from_json_schema(
        {"$ref": "CustomRef"},
        definitions=custom_defs,
    )
    assert isinstance(result, Reference)


# LLM-generated content at query #23
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    result = to_json_schema(Any())
    assert result is True

    # Test NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test String field
    result = to_json_schema(String(allow_null=False))
    assert result["type"] == "string"
    assert "default" not in result

    # Test String field with null
    result = to_json_schema(String(allow_null=True))
    assert result["type"] == ["string", "null"]

    # Test String with constraints
    result = to_json_schema(String(min_length=5, max_length=10, pattern="^[a-z]+$"))
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"

    # Test String with allow_blank=False
    result = to_json_schema(String(allow_blank=False))
    assert result["minLength"] == 1

    # Test Integer field
    result = to_json_schema(Integer(allow_null=False))
    assert result["type"] == "integer"

    # Test Integer with constraints
    result = to_json_schema(Integer(minimum=0, maximum=100, multiple_of=5))
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test Float field
    result = to_json_schema(Float(allow_null=True))
    assert result["type"] == ["number", "null"]

    # Test Float with constraints
    result = to_json_schema(Float(exclusive_minimum=0.5, exclusive_maximum=99.5))
    assert result["exclusiveMinimum"] == 0.5
    assert result["exclusiveMaximum"] == 99.5

    # Test Boolean field
    result = to_json_schema(Boolean(allow_null=False))
    assert result["type"] == "boolean"

    # Test Boolean with null
    result = to_json_schema(Boolean(allow_null=True))
    assert result["type"] == ["boolean", "null"]

    # Test Array field
    result = to_json_schema(Array(items=String(), min_items=1, max_items=10))
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["items"]["type"] == "string"

    # Test Array with unique items
    result = to_json_schema(Array(items=Integer(), unique_items=True))
    assert result["uniqueItems"] is True

    # Test Array with tuple items
    result = to_json_schema(Array(items=[String(), Integer()]))
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array with additional_items as bool
    result = to_json_schema(Array(additional_items=False))
    assert result["additionalItems"] is False

    # Test Array with additional_items as field
    result = to_json_schema(Array(additional_items=String()))
    assert result["additionalItems"]["type"] == "string"

    # Test Object field
    result = to_json_schema(Object(properties={"name": String()}))
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["properties"]["name"]["type"] == "string"

    # Test Object with required fields
    result = to_json_schema(Object(properties={"id": Integer()}, required=["id"]))
    assert result["required"] == ["id"]

    # Test Object with pattern properties
    result = to_json_schema(Object(pattern_properties={"^S_": String()}))
    assert "^S_" in result["patternProperties"]

    # Test Object with additional_properties as bool
    result = to_json_schema(Object(additional_properties=False))
    assert result["additionalProperties"] is False

    # Test Object with additional_properties as field
    result = to_json_schema(Object(additional_properties=String()))
    assert result["additionalProperties"]["type"] == "string"

    # Test Object with property_names
    result = to_json_schema(Object(property_names=String(pattern="^[a-z]+$")))
    assert result["propertyNames"]["pattern"] == "^[a-z]+$"

    # Test Object with min/max properties
    result = to_json_schema(Object(min_properties=1, max_properties=5))
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5

    # Test Choice field
    result = to_json_schema(Choice(choices=[("a", "a"), ("b", "b")]))
    assert result["enum"] == ["a", "b"]

    # Test Const field
    result = to_json_schema(Const(const="fixed"))
    assert result["const"] == "fixed"

    # Test Union field
    result = to_json_schema(Union(any_of=[String(), Integer()]))
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    result = to_json_schema(OneOf(one_of=[String(), Integer()]))
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    result = to_json_schema(AllOf(all_of=[String(), Const(const="test")]))
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Not field
    result = to_json_schema(Not(negated=String()))
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test IfThenElse with all clauses
    result = to_json_schema(IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    ))
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse with only if and then
    result = to_json_schema(IfThenElse(if_clause=String(), then_clause=Integer()))
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test Definitions
    definitions = Definitions({"MyString": String()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "MyString" in result["components"]["schemas"]

    # Test Reference field
    definitions_obj = Definitions({"MyString": String()})
    ref = Reference(to="MyString", definitions=definitions_obj)
    result = to_json_schema(ref)
    assert result["$ref"] == "#/components/schemas/MyString"
    assert "components" in result

    # Test with default value
    result = to_json_schema(String(default="test"))
    assert result["default"] == "test"

    # Test unsupported field type raises error
    class UnsupportedField(Field):
        pass
    
    with_raises = False
    try:
        to_json_schema(UnsupportedField())
    except ValueError as e:
        assert "Cannot convert field type" in str(e)
        with_raises = True
    assert with_raises


# LLM-generated content at query #24
#--------------------------

```python
def test_to_json_schema():
    # Test Any() returns True
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch() returns False
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"
    
    # Test String field with allow_null
    string_null = String(allow_null=True)
    result = to_json_schema(string_null)
    assert result["type"] == ["string", "null"]
    
    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False, exclusive_minimum=0.0, exclusive_maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 1.0
    
    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Boolean field with allow_null
    boolean_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, min_items=1, max_items=5, items=String(), unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] is True
    assert "items" in result
    
    # Test Array with tuple items
    array_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Object field
    object_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=5
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    
    # Test Object with pattern_properties
    object_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(object_pattern)
    assert "patternProperties" in result
    
    # Test Object with additional_properties as bool
    object_additional_bool = Object(additional_properties=False)
    result = to_json_schema(object_additional_bool)
    assert result["additionalProperties"] is False
    
    # Test Object with additional_properties as Field
    object_additional_field = Object(additional_properties=String())
    result = to_json_schema(object_additional_field)
    assert isinstance(result["additionalProperties"], dict)
    
    # Test Object with property_names
    object_prop_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_prop_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test IfThenElse field
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without else clause
    if_then = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(if_then)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test Definitions
    definitions = Definitions({"StringSchema": String(), "IntSchema": Integer()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    
    # Test Reference field
    reference = Reference(to="StringSchema", definitions=Definitions({"StringSchema": String()}))
    result = to_json_schema(reference)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/StringSchema"
    assert "components" in result
    
    # Test invalid field type raises ValueError
    class InvalidField(Field):
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(InvalidField())
    
    # Test Array with additional_items as bool
    array_additional_bool = Array(additional_items=False)
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] is False
    
    # Test Array with additional_items as Field
    array_additional_field = Array(additional_items=String())
    result = to_json_schema(array_additional_field)
    assert isinstance(result["additionalItems"], dict)
    
    # Test String with blank allowed
    string_blank = String(allow_blank=True)
    result = to_json_schema(string_blank)
    assert "minLength" not in result or result.get("minLength") is None


# LLM-generated content at query #25
#--------------------------

```python
import pytest
import re


def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) is True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result

    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=1, exclusive_maximum=99, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 1
    assert result["exclusiveMaximum"] == 99
    assert result["multipleOf"] == 5

    # Test Integer field with allow_null
    int_field_null = Integer(allow_null=True)
    result = to_json_schema(int_field_null)
    assert result["type"] == ["integer", "null"]

    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=100.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 100.0

    # Test Decimal field
    decimal_field = Decimal(allow_null=False)
    result = to_json_schema(decimal_field)
    assert result["type"] == "number"

    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test Boolean field with allow_null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert "items" in result

    # Test Array field with list of items
    array_list_field = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_list_field)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array field with additional_items as bool
    array_additional_bool = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] is False

    # Test Array field with additional_items as Field
    array_additional_field = Array(allow_null=False, additional_items=String())
    result = to_json_schema(array_additional_field)
    assert isinstance(result["additionalItems"], dict)

    # Test Object field
    obj_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]

    # Test Object field with pattern_properties
    obj_pattern = Object(allow_null=False, pattern_properties={"^S_": String()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result

    # Test Object field with additional_properties as bool
    obj_additional_bool = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(obj_additional_bool)
    assert result["additionalProperties"] is False

    # Test Object field with additional_properties as Field
    obj_additional_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(obj_additional_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test Object field with property_names
    obj_property_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_property_names)
    assert "propertyNames" in result

    # Test Schema field
    schema_field = Schema(allow_null=False, fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]

    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Choice(choices=[("a", "A")])])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse field with only if clause
    if_only_field = IfThenElse(if_clause=Choice(choices=[("a", "A")]))
    result = to_json_schema(if_only_field)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result

    # Test with


# LLM-generated content at query #26
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test with String field allowing null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result

    # Test with Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test with Float field
    float_field = Float(allow_null=False, exclusive_minimum=0.5, exclusive_maximum=99.5)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["exclusiveMinimum"] == 0.5
    assert result["exclusiveMaximum"] == 99.5

    # Test with Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=5, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] is True
    assert isinstance(result["items"], dict)

    # Test with Array field with list of items
    array_field_list = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_list)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test with Array field with additional_items as bool
    array_field_additional = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"] is False

    # Test with Array field with additional_items as Field
    array_field_additional_field = Array(allow_null=False, additional_items=String())
    result = to_json_schema(array_field_additional_field)
    assert isinstance(result["additionalItems"], dict)

    # Test with Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]

    # Test with Object field with pattern_properties
    object_field_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result

    # Test with Object field with additional_properties as bool
    object_field_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(object_field_additional)
    assert result["additionalProperties"] is False

    # Test with Object field with additional_properties as Field
    object_field_additional_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(object_field_additional_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test with Object field with property_names
    object_field_property_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_field_property_names)
    assert "propertyNames" in result

    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=100)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without then and else
    if_field = IfThenElse(if_clause=String())
    result = to_json_schema(if_field)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result

    # Test with Reference field
    definitions = Definitions({"MyType": String()})
    ref_field = Reference(to="MyType", definitions=definitions)
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/MyType"
    assert "components" in result
    assert "schemas" in result["components"]

    # Test with Definitions
    definitions = Definitions({"StringType": String(), "IntType": Integer()})
    result = to_json_schema(definitions)
    assert "StringType" in result
    assert "IntType" in result

    # Test with Schema field
    schema_field = Schema(
        allow_null=False,
        


# LLM-generated content at query #27
#--------------------------

```python
def test_to_json_schema():
    # Test Any type returns True
    assert to_json_schema(Any()) is True

    # Test NeverMatch type returns False
    assert to_json_schema(NeverMatch()) is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test String field with allow_blank
    string_blank = String(allow_blank=True, min_length=None)
    result = to_json_schema(string_blank)
    assert "minLength" not in result

    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=1, maximum=100, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 1
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test Integer field with exclusive bounds
    integer_exclusive = Integer(exclusive_minimum=0, exclusive_maximum=10)
    result = to_json_schema(integer_exclusive)
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 10

    # Test Float field
    float_field = Float(allow_null=False, minimum=0.5, maximum=99.9)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.5
    assert result["maximum"] == 99.9

    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test Boolean field with allow_null
    boolean_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_null)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with simple items
    array_field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test Array field with tuple items
    array_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array field with additionalItems
    array_additional_bool = Array(additional_items=False)
    result = to_json_schema(array_additional_bool)
    assert result["additionalItems"] is False

    array_additional_field = Array(additional_items=String())
    result = to_json_schema(array_additional_field)
    assert isinstance(result["additionalItems"], dict)

    # Test Object field
    object_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10

    # Test Object field with pattern_properties
    object_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(object_pattern)
    assert "patternProperties" in result

    # Test Object field with additionalProperties
    object_additional_bool = Object(additional_properties=False)
    result = to_json_schema(object_additional_bool)
    assert result["additionalProperties"] is False

    object_additional_field = Object(additional_properties=String())
    result = to_json_schema(object_additional_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test Object field with property_names
    object_prop_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_prop_names)
    assert "propertyNames" in result

    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse without then/else
    if_only_field = IfThenElse(if_clause=String())
    result = to_json_schema(if_only_field)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result

    # Test with default values
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"

    # Test Reference field
    definitions = Definitions()
    ref_field = Reference(to="MySchema", definitions=definitions)
    result = to_json_schema(ref_field)
    assert "$ref" in result

    # Test Definitions
    defs = Definitions()
    defs["StringDef"] = String()
    defs["IntDef"] = Integer()
    result = to_json_schema(defs)
    assert "components" in result
    assert "schemas" in result["components"]

    # Test invalid field type raises error
    class CustomField(Field):
        pass

    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(CustomField())


# LLM-generated content at query #28
#--------------------------

```python
def test_to_json_schema():
    # Test Any field returns True
    assert to_json_schema(Any()) is True

    # Test NeverMatch field returns False
    assert to_json_schema(NeverMatch()) is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result or result.get("minLength") != 1

    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test Integer field with exclusive bounds
    integer_field_exc = Integer(allow_null=False, exclusive_minimum=0, exclusive_maximum=100)
    result = to_json_schema(integer_field_exc)
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100

    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test Boolean field with null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with items
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test Array field with tuple items
    array_tuple_field = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_tuple_field)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array field with additionalItems
    array_additional = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_additional)
    assert result["additionalItems"] is False

    # Test Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10

    # Test Object field with pattern properties
    object_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String()}
    )
    result = to_json_schema(object_pattern)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]

    # Test Object field with additionalProperties
    object_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(object_additional)
    assert result["additionalProperties"] is False

    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test IfThenElse field
    if_then_else = IfThenElse(
        if_clause=Choice(choices=[("a", "A"), ("b", "B")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse without else clause
    if_then = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String()
    )
    result = to_json_schema(if_then)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test with default values
    field_with_default = String(default="default_value")
    result = to_json_schema(field_with_default)
    assert result.get("default") == "default_value"

    # Test Definitions
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]

    # Test Reference field
    reference_field = Reference(to="User", definitions=Definitions({"User": Object()}))
    result = to_json_schema(reference_field)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/User"

    # Test invalid field type raises ValueError
    class InvalidField(Field):
        pass

    invalid_field = InvalidField()
    try:
        to_json_schema(invalid_field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


# LLM-generated content at query #29
#--------------------------

```python
def test_to_json_schema():
    # Test Any field returns True
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch field returns False
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern=r"\d+", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == r"\d+"
    assert result["format"] == "email"
    
    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True, min_length=None)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result
    
    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=5, exclusive_maximum=95, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5
    assert result["exclusiveMaximum"] == 95
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    
    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test Boolean field with allow_null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field with items
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"
    
    # Test Array field with tuple items
    array_tuple = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Array field with additional_items as bool
    array_additional = Array(allow_null=False, items=String(), additional_items=False)
    result = to_json_schema(array_additional)
    assert result["additionalItems"] is False
    
    # Test Array field with additional_items as Field
    array_additional_field = Array(allow_null=False, items=String(), additional_items=Integer())
    result = to_json_schema(array_additional_field)
    assert result["additionalItems"]["type"] == "integer"
    
    # Test Object field
    obj_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]
    
    # Test Object field with pattern_properties
    obj_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String()}
    )
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]
    
    # Test Object field with additional_properties as bool
    obj_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(obj_additional)
    assert result["additionalProperties"] is False
    
    # Test Object field with additional_properties as Field
    obj_additional_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(obj_additional_field)
    assert result["additionalProperties"]["type"] == "string"
    
    # Test Object field with property_names
    obj_property_names = Object(allow_null=False, property_names=String(min_length=1))
    result = to_json_schema(obj_property_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "option_a"), ("b", "option_b")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    oneof_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(oneof_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    allof_field = AllOf(all_of=[String(), Object()])
    result = to_json_schema(allof_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"
    
    # Test IfThenElse field
    if_then_else = IfThenElse(
        if_clause=Object(properties={"type": Choice(choices=[("a", "a")])}),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without then/else
    if_only = IfThenElse(if_clause=String())
    result = to_json_schema(if_only)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result
    
    # Test with definitions
    definitions = Definitions({"MyString": String()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "MyString" in result["components"]["schemas"]
    
    # Test invalid field type raises ValueError
    class InvalidField(Field):
        pass
    
    invalid_field = InvalidField()
    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_


# LLM-generated content at query #30
#--------------------------

```python
def test_to_json_schema():
    # Test Any type
    result = to_json_schema(Any())
    assert result is True

    # Test NeverMatch type
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="[a-z]+")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "[a-z]+"

    # Test String field with null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100

    # Test Float field
    float_field = Float(allow_null=False, exclusive_minimum=0.5, exclusive_maximum=99.9)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["exclusiveMinimum"] == 0.5
    assert result["exclusiveMaximum"] == 99.9

    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test Boolean field with null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=5)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["items"]["type"] == "string"

    # Test Array with unique items
    array_unique = Array(allow_null=False, items=Integer(), unique_items=True)
    result = to_json_schema(array_unique)
    assert result["uniqueItems"] is True

    # Test Object field
    obj_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]

    # Test Object with additional properties
    obj_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(obj_additional)
    assert result["additionalProperties"] is False

    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Object()])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test IfThenElse field
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse without else clause
    if_then = IfThenElse(if_clause=String(), then_clause=Integer())
    result = to_json_schema(if_then)
    assert "if" in result
    assert "then" in result
    assert "else" not in result

    # Test Reference field with definitions
    ref_field = Reference(to="CustomType", definitions=Definitions())
    result = to_json_schema(ref_field)
    assert "$ref" in result

    # Test with default values
    string_with_default = String(default="test_value")
    result = to_json_schema(string_with_default)
    assert result.get("default") == "test_value"

    # Test Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]

    # Test with definitions at root level
    definitions = Definitions()
    definitions["User"] = Object(properties={"name": String()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]


# LLM-generated content at query #31
#--------------------------

```python
def test_to_json_schema():
    # Test Any field returns True
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch field returns False
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"
    
    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result
    
    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=5, exclusive_maximum=95, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5
    assert result["exclusiveMaximum"] == 95
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    
    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Boolean field with allow_null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"
    
    # Test Array field with tuple items
    array_field_tuple = Array(allow_null=False, items=[String(), Integer()])
    result = to_json_schema(array_field_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Array field with additional_items as bool
    array_field_additional_bool = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_field_additional_bool)
    assert result["additionalItems"] is False
    
    # Test Array field with additional_items as Field
    array_field_additional_field = Array(allow_null=False, additional_items=String())
    result = to_json_schema(array_field_additional_field)
    assert isinstance(result["additionalItems"], dict)
    assert result["additionalItems"]["type"] == "string"
    
    # Test Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]
    
    # Test Object field with pattern_properties
    object_field_pattern = Object(
        allow_null=False,
        pattern_properties={"^S_": String()}
    )
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]
    
    # Test Object field with additional_properties as bool
    object_field_additional_bool = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(object_field_additional_bool)
    assert result["additionalProperties"] is False
    
    # Test Object field with additional_properties as Field
    object_field_additional_field = Object(allow_null=False, additional_properties=String())
    result = to_json_schema(object_field_additional_field)
    assert isinstance(result["additionalProperties"], dict)
    
    # Test Object field with property_names
    object_field_property_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_field_property_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Choice(choices=[("a", "A"), ("b", "B")])])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse field without else_clause
    if_then_field = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test Reference field
    definitions = {"User": Object(properties={"name": String()})}
    reference_field = Reference(to="User", definitions=definitions)
    result = to_json_schema(reference_field)


# LLM-generated content at query #32
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    result = to_json_schema(Any())
    assert result is True

    # Test NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result or result.get("minLength") == 0

    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=1, exclusive_maximum=99, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 1
    assert result["exclusiveMaximum"] == 99
    assert result["multipleOf"] == 5

    # Test Integer field with allow_null
    integer_field_null = Integer(allow_null=True)
    result = to_json_schema(integer_field_null)
    assert result["type"] == ["integer", "null"]

    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test Float field with allow_null
    float_field_null = Float(allow_null=True)
    result = to_json_schema(float_field_null)
    assert result["type"] == ["number", "null"]

    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test Boolean field with allow_null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with items
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test Array field with allow_null
    array_field_null = Array(allow_null=True)
    result = to_json_schema(array_field_null)
    assert result["type"] == ["array", "null"]

    # Test Array field with list of items
    array_field_list = Array(items=[String(), Integer()])
    result = to_json_schema(array_field_list)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array field with additional_items as bool
    array_field_additional_bool = Array(additional_items=False)
    result = to_json_schema(array_field_additional_bool)
    assert result["additionalItems"] is False

    # Test Array field with additional_items as Field
    array_field_additional_field = Array(additional_items=String())
    result = to_json_schema(array_field_additional_field)
    assert isinstance(result["additionalItems"], dict)

    # Test Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]

    # Test Object field with allow_null
    object_field_null = Object(allow_null=True)
    result = to_json_schema(object_field_null)
    assert result["type"] == ["object", "null"]

    # Test Object field with pattern_properties
    object_field_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result

    # Test Object field with additional_properties as bool
    object_field_additional_bool = Object(additional_properties=False)
    result = to_json_schema(object_field_additional_bool)
    assert result["additionalProperties"] is False

    # Test Object field with additional_properties as Field
    object_field_additional_field = Object(additional_properties=String())
    result = to_json_schema(object_field_additional_field)
    assert isinstance(result["additionalProperties"], dict)

    # Test Object field with property_names
    object_field_prop_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_field_prop_names)
    assert "propertyNames" in result

    # Test Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]

    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Object()])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else


# LLM-generated content at query #33
#--------------------------

```python
def test_to_json_schema():
    # Test Any returns True
    assert to_json_schema(Any()) is True

    # Test NeverMatch returns False
    assert to_json_schema(NeverMatch()) is False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String with allow_null
    string_null = String(allow_null=True)
    result = to_json_schema(string_null)
    assert result["type"] == ["string", "null"]

    # Test String with allow_blank
    string_blank = String(allow_blank=True)
    result = to_json_schema(string_blank)
    assert "minLength" not in result

    # Test Integer field
    integer_field = Integer(minimum=0, maximum=100, exclusive_minimum=5, exclusive_maximum=95, multiple_of=5)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5
    assert result["exclusiveMaximum"] == 95
    assert result["multipleOf"] == 5

    # Test Float field
    float_field = Float(allow_null=True)
    result = to_json_schema(float_field)
    assert result["type"] == ["number", "null"]

    # Test Boolean field
    boolean_field = Boolean()
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test Boolean with allow_null
    boolean_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_null)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with items
    array_field = Array(items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert "items" in result

    # Test Array with tuple items
    array_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array with additional_items
    array_additional = Array(items=String(), additional_items=True)
    result = to_json_schema(array_additional)
    assert result["additionalItems"] is True

    # Test Array with additional_items as Field
    array_additional_field = Array(items=String(), additional_items=Integer())
    result = to_json_schema(array_additional_field)
    assert isinstance(result["additionalItems"], dict)

    # Test Object field with properties
    obj_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=5
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5

    # Test Object with pattern_properties
    obj_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result

    # Test Object with additional_properties
    obj_additional = Object(additional_properties=True)
    result = to_json_schema(obj_additional)
    assert result["additionalProperties"] is True

    # Test Object with property_names
    obj_property_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_property_names)
    assert "propertyNames" in result

    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test IfThenElse with all clauses
    if_then_else = IfThenElse(
        if_clause=Choice(choices=[("a", "A")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse with only if clause
    if_only = IfThenElse(if_clause=Choice(choices=[("a", "A")]))
    result = to_json_schema(if_only)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result

    # Test Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]

    # Test with definitions
    definitions = Definitions({
        "User": Object(properties={"name": String()}),
        "Product": Object(properties={"title": String()})
    })
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]

    # Test Reference field
    reference_field = Reference(to="User", definitions=definitions)
    result = to_json_schema(reference_field)
    assert "$ref" in result
    assert "components" in result

    # Test with default values
    string_with_default = String(default="test")
    result = to_json_schema(string_with_default)
    assert result["default"] == "test"

    # Test invalid field type raises ValueError
    class CustomField(Field):
        pass

    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(CustomField())


# LLM-generated content at query #34
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    
    # Test String field with null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String with constraints
    string_constrained = String(
        min_length=2,
        max_length=10,
        pattern="^[a-z]+$",
        format="email"
    )
    result = to_json_schema(string_constrained)
    assert result["minLength"] == 2
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"
    
    # Test Integer field
    integer_field = Integer(allow_null=False)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    
    # Test Integer with constraints
    integer_constrained = Integer(
        minimum=0,
        maximum=100,
        multiple_of=5
    )
    result = to_json_schema(integer_constrained)
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Boolean with null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, items=String())
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"
    
    # Test Array with constraints
    array_constrained = Array(
        items=Integer(),
        min_items=1,
        max_items=10,
        unique_items=True
    )
    result = to_json_schema(array_constrained)
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    
    # Test Object field
    object_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test Object with additional properties
    object_additional = Object(
        properties={"id": Integer()},
        additional_properties=String()
    )
    result = to_json_schema(object_additional)
    assert result["additionalProperties"]["type"] == "string"
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "option_a"), ("b", "option_b")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Object()])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without else
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with definitions
    definitions = Definitions({
        "StringDef": String(),
        "IntegerDef": Integer()
    })
    result = to_json_schema(definitions)
    assert "StringDef" in result
    assert "IntegerDef" in result
    
    # Test field with default value
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"
    
    # Test complex nested structure
    complex_field = Object(
        properties={
            "name": String(min_length=1),
            "items": Array(items=Object(properties={"id": Integer()})),
            "status": Choice(choices=[("active", "Active"), ("inactive", "Inactive")])
        },
        required=["name", "items"]
    )
    result = to_json_schema(complex_field)
    assert result["type"] == "object"
    assert result["required"] == ["name", "items"]
    assert "properties" in result
    assert result["properties"]["items"]["type"] == "array"


# LLM-generated content at query #35
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, allow_blank=True, min_length=None, max_length=10, pattern_regex=None, format=None)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["maxLength"] == 10

    # Test with String field allowing null
    string_field_null = String(allow_null=True, allow_blank=True, min_length=None, max_length=None, pattern_regex=None, format=None)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test with String field with min_length
    string_field_min = String(allow_null=False, allow_blank=False, min_length=5, max_length=None, pattern_regex=None, format=None)
    result = to_json_schema(string_field_min)
    assert result["minLength"] == 5

    # Test with Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100

    # Test with Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test with Boolean field
    boolean_field = Boolean(allow_null=False)
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allowing null
    boolean_field_null = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_null)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, min_items=0, max_items=None, items=None, additional_items=True, unique_items=False)
    result = to_json_schema(array_field)
    assert result["type"] == "array"

    # Test with Array field with items
    array_field_items = Array(allow_null=False, min_items=1, max_items=10, items=String(allow_null=False, allow_blank=True, min_length=None, max_length=None, pattern_regex=None, format=None), additional_items=True, unique_items=False)
    result = to_json_schema(array_field_items)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["items"]["type"] == "string"

    # Test with Object field
    object_field = Object(allow_null=False, properties=None, pattern_properties=None, additional_properties=None, property_names=None, min_properties=None, max_properties=None, required=None)
    result = to_json_schema(object_field)
    assert result["type"] == "object"

    # Test with Object field with properties
    object_field_props = Object(
        allow_null=False,
        properties={
            "name": String(allow_null=False, allow_blank=True, min_length=None, max_length=None, pattern_regex=None, format=None),
            "age": Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)
        },
        pattern_properties=None,
        additional_properties=None,
        property_names=None,
        min_properties=None,
        max_properties=None,
        required=["name"]
    )
    result = to_json_schema(object_field_props)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]

    # Test with Choice field
    choice_field = Choice(choices=[("option1", "option1"), ("option2", "option2")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["option1", "option2"]

    # Test with Const field
    const_field = Const(const=42)
    result = to_json_schema(const_field)
    assert result["const"] == 42

    # Test with Union field
    union_field = Union(any_of=[
        String(allow_null=False, allow_blank=True, min_length=None, max_length=None, pattern_regex=None, format=None),
        Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)
    ])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[
        String(allow_null=False, allow_blank=True, min_length=None, max_length=None, pattern_regex=None, format=None),
        Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None)
    ])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[
        String(allow_null=False, allow_blank=True, min_length=None, max_length=None, pattern_regex=None, format=None),
        Object(allow_null=False, properties=None, pattern_properties=None, additional_properties=None, property_names=None, min_properties=None, max_properties=None, required=None)
    ])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with Not field
    not_field = Not(negated=String(allow_null=False, allow_blank=True, min_length=None, max_length=None, pattern_regex=None, format=None))
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(allow_null=False, allow_blank=True, min_length=None, max_length=None, pattern_regex=None, format=None),
        then_clause=Integer(allow_null=False, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, multiple_of=None),
        else_clause=None
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result

    # Test with Definitions
    definitions = Definitions()
    definitions["MyString"] = String(allow_null=False, allow_blank=True, min_length=None, max_length=None, pattern_regex=None, format=None)
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "MyString" in result["components"]["schemas"]


# LLM-generated content at query #36
#--------------------------

```python
def test_to_json_schema():
    # Test Any type returns True
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch type returns False
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"
    
    # Test String field with allow_null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test String field with allow_blank
    string_field_blank = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field_blank)
    assert "minLength" not in result
    
    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=5, exclusive_maximum=95, multiple_of=5)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 5
    assert result["exclusiveMaximum"] == 95
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float(allow_null=False, minimum=0.0, maximum=1.0)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    
    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"
    
    # Test Boolean field with allow_null
    bool_field_null = Boolean(allow_null=True)
    result = to_json_schema(bool_field_null)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert "items" in result
    
    # Test Array field with additional_items
    array_field_additional = Array(allow_null=False, items=String(), additional_items=False)
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"] is False
    
    # Test Object field
    obj_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    
    # Test Object field with pattern_properties
    obj_field_pattern = Object(
        allow_null=False,
        pattern_properties={"^[a-z]+$": String()}
    )
    result = to_json_schema(obj_field_pattern)
    assert "patternProperties" in result
    
    # Test Object field with additional_properties
    obj_field_additional = Object(allow_null=False, additional_properties=False)
    result = to_json_schema(obj_field_additional)
    assert result["additionalProperties"] is False
    
    # Test Object field with property_names
    obj_field_names = Object(allow_null=False, property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(obj_field_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test IfThenElse field
    if_then_else = IfThenElse(
        if_clause=Choice(choices=[("a", "A"), ("b", "B")]),
        then_clause=String(),
        else_clause=Integer()
    )
    result = to_json_schema(if_then_else)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without then/else
    if_only = IfThenElse(if_clause=String())
    result = to_json_schema(if_only)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test Reference field with definitions
    ref_field = Reference(to="User", definitions={})
    result = to_json_schema(ref_field)
    assert "$ref" in result
    assert result["$ref"] == "#/components/schemas/User"
    
    # Test with root definitions
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    
    # Test Schema field
    schema = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(schema)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]
    
    # Test invalid field type raises ValueError
    class InvalidField(Field):
        pass
    
    with pytest.raises(ValueError, match="Cannot convert field type"):
        to_json_schema(InvalidField())


# LLM-generated content at query #37
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) is True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) is False
    
    # Test String field
    string_field = String()
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    
    # Test String with allow_null
    string_field_nullable = String(allow_null=True)
    result = to_json_schema(string_field_nullable)
    assert result["type"] == ["string", "null"]
    
    # Test String with constraints
    string_field_constrained = String(min_length=5, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_field_constrained)
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test String with format
    string_field_format = String(format="email")
    result = to_json_schema(string_field_format)
    assert result["format"] == "email"
    
    # Test Integer field
    integer_field = Integer()
    result = to_json_schema(integer_field)
    assert result["type"] == "integer"
    
    # Test Integer with constraints
    integer_field_constrained = Integer(minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(integer_field_constrained)
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5
    
    # Test Float field
    float_field = Float()
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    
    # Test Float with exclusive bounds
    float_field_exclusive = Float(exclusive_minimum=0.0, exclusive_maximum=100.0)
    result = to_json_schema(float_field_exclusive)
    assert result["exclusiveMinimum"] == 0.0
    assert result["exclusiveMaximum"] == 100.0
    
    # Test Boolean field
    boolean_field = Boolean()
    result = to_json_schema(boolean_field)
    assert result["type"] == "boolean"
    
    # Test Boolean with allow_null
    boolean_field_nullable = Boolean(allow_null=True)
    result = to_json_schema(boolean_field_nullable)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(items=String())
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"
    
    # Test Array with constraints
    array_field_constrained = Array(items=Integer(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field_constrained)
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    
    # Test Array with tuple items
    array_field_tuple = Array(items=[String(), Integer()])
    result = to_json_schema(array_field_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Array with additional items
    array_field_additional = Array(items=String(), additional_items=Boolean())
    result = to_json_schema(array_field_additional)
    assert result["additionalItems"]["type"] == "boolean"
    
    # Test Array with additional items as boolean
    array_field_additional_bool = Array(items=String(), additional_items=False)
    result = to_json_schema(array_field_additional_bool)
    assert result["additionalItems"] is False
    
    # Test Object field
    object_field = Object(properties={"name": String(), "age": Integer()})
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    
    # Test Object with constraints
    object_field_constrained = Object(
        properties={"name": String()},
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(object_field_constrained)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]
    
    # Test Object with pattern properties
    object_field_pattern = Object(pattern_properties={"^S_": String()})
    result = to_json_schema(object_field_pattern)
    assert "patternProperties" in result
    assert "^S_" in result["patternProperties"]
    
    # Test Object with additional properties
    object_field_additional = Object(additional_properties=String())
    result = to_json_schema(object_field_additional)
    assert result["additionalProperties"]["type"] == "string"
    
    # Test Object with property names
    object_field_property_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_field_property_names)
    assert "propertyNames" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const="constant_value")
    result = to_json_schema(const_field)
    assert result["const"] == "constant_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=100)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"
    
    # Test IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test IfThenElse without else clause
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" not in result
    
    # Test with default values
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"
    
    # Test Definitions
    definitions = Definitions({"MyString": String(), "MyInt": Integer()})
    result = to_json_schema(definitions)
    assert "


# LLM-generated content at query #38
#--------------------------

```python
def test_to_json_schema():
    # Test with Any field
    result = to_json_schema(Any())
    assert result is True

    # Test with NeverMatch field
    result = to_json_schema(NeverMatch())
    assert result is False

    # Test with String field
    string_field = String(allow_null=False, allow_blank=True)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert "minLength" not in result

    # Test with String field with min_length
    string_field = String(allow_null=False, allow_blank=False, min_length=5)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 5

    # Test with String field with max_length
    string_field = String(allow_null=False, max_length=10)
    result = to_json_schema(string_field)
    assert result["maxLength"] == 10

    # Test with String field allow_null
    string_field = String(allow_null=True)
    result = to_json_schema(string_field)
    assert result["type"] == ["string", "null"]

    # Test with Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100

    # Test with Float field
    float_field = Float(allow_null=False)
    result = to_json_schema(float_field)
    assert result["type"] == "number"

    # Test with Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test with Boolean field allow_null
    bool_field = Boolean(allow_null=True)
    result = to_json_schema(bool_field)
    assert result["type"] == ["boolean", "null"]

    # Test with Array field
    array_field = Array(allow_null=False, items=String())
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["items"]["type"] == "string"

    # Test with Array field with min/max items
    array_field = Array(allow_null=False, min_items=1, max_items=5)
    result = to_json_schema(array_field)
    assert result["minItems"] == 1
    assert result["maxItems"] == 5

    # Test with Array field unique_items
    array_field = Array(allow_null=False, unique_items=True)
    result = to_json_schema(array_field)
    assert result["uniqueItems"] is True

    # Test with Object field
    obj_field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()}
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]

    # Test with Object field required
    obj_field = Object(
        allow_null=False,
        properties={"name": String()},
        required=["name"]
    )
    result = to_json_schema(obj_field)
    assert result["required"] == ["name"]

    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]

    # Test with Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"

    # Test with Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test with OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test with AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test with Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with IfThenElse field
    if_then_else_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test with IfThenElse field without then/else
    if_then_else_field = IfThenElse(if_clause=String())
    result = to_json_schema(if_then_else_field)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result

    # Test with Definitions
    definitions = Definitions({"User": Object(properties={"name": String()})})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]

    # Test with default values
    string_field = String(allow_null=False, default="default_value")
    result = to_json_schema(string_field)
    assert result.get("default") == "default_value"

    # Test with pattern
    import re
    string_field = String(allow_null=False, pattern=re.compile(r"^[a-z]+$"))
    result = to_json_schema(string_field)
    assert result["pattern"] == "^[a-z]+$"

    # Test with format
    string_field = String(allow_null=False, format="email")
    result = to_json_schema(string_field)
    assert result["format"] == "email"

    # Test with exclusive_minimum and exclusive_maximum
    int_field = Integer(allow_null=False, exclusive_minimum=0, exclusive_maximum=100)
    result = to_json_schema(int_field)
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100

    # Test with multiple_of
    float_field = Float(allow_null=False, multiple_of=0.5)
    result = to_json_schema(float_field)
    assert result["multipleOf"] == 0.5

    # Test with Array additional_items as boolean
    array_field = Array(allow_null=False, additional_items=False)
    result = to_json_schema(array_field)
    assert result["additionalItems"] is False

    # Test with Array additional_items as Field
    array_field = Array(allow_null=False, additional_items=String())
    result = to_json_schema(array_field)
    assert isinstance(result["additionalItems"], dict)

    # Test with Object pattern_properties
    obj_field = Object(
        allow_null=False,
        pattern_properties={"^S_": String(), "^I_": Integer()}
    )
    result = to_json_schema(obj_field)
    assert "patternProperties" in result

    # Test with Object property_names
    obj_field = Object(
        allow_null=False,
        property_names=String(pattern=re.compile(r"^[a-z]+$"))
    )
    


