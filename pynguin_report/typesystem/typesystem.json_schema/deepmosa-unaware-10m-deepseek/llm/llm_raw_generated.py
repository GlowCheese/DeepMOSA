####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_if_then_else_from_json_schema():
    definitions = Definitions()
    
    # Test basic if-then
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, String)
    assert result.then_clause.min_length == 5
    assert result.else_clause is None
    
    # Test if-then-else
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "integer", "minimum": 0}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, String)
    assert result.then_clause.min_length == 5
    assert isinstance(result.else_clause, Integer)
    assert result.else_clause.minimum == 0
    
    # Test with default value
    data = {
        "if": {"type": "boolean"},
        "then": {"type": "boolean"},
        "else": {"type": "null"},
        "default": True
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == True
    
    # Test with nested schemas
    data = {
        "if": {"type": "object", "properties": {"x": {"type": "integer"}}},
        "then": {"type": "object", "required": ["x"]},
        "else": {"type": "object", "properties": {"y": {"type": "string"}}}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Object)
    assert isinstance(result.then_clause, Object)
    assert isinstance(result.else_clause, Object)
    
    # Test with array condition
    data = {
        "if": {"type": "array", "minItems": 1},
        "then": {"type": "array", "uniqueItems": True}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Array)
    assert result.if_clause.min_items == 1
    assert isinstance(result.then_clause, Array)
    assert result.then_clause.unique_items == True
    
    # Test with number condition
    data = {
        "if": {"type": "number", "minimum": 0},
        "then": {"type": "number", "maximum": 100},
        "else": {"type": "number", "minimum": -100}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Float)
    assert result.if_clause.minimum == 0
    assert isinstance(result.then_clause, Float)
    assert result.then_clause.maximum == 100
    assert isinstance(result.else_clause, Float)
    assert result.else_clause.minimum == -100


# LLM-generated content at query #2
#--------------------------

```python
def test_enum_from_json_schema():
    definitions = Definitions()
    
    # Test basic enum with strings
    data = {"enum": ["red", "green", "blue"]}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [("red", "red"), ("green", "green"), ("blue", "blue")]
    assert field.default is NO_DEFAULT
    
    # Test enum with mixed types
    data = {"enum": [1, 2, 3]}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [(1, 1), (2, 2), (3, 3)]
    
    # Test enum with default value
    data = {"enum": ["a", "b", "c"], "default": "b"}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.default == "b"
    
    # Test enum with complex values
    data = {"enum": [{"key": "value"}, [1, 2, 3], True]}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [
        ({"key": "value"}, {"key": "value"}),
        ([1, 2, 3], [1, 2, 3]),
        (True, True)
    ]
    
    # Test enum with null value
    data = {"enum": [None, "value"]}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [(None, None), ("value", "value")]
    
    # Test enum with single value
    data = {"enum": ["only"]}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [("only", "only")]


# LLM-generated content at query #3
#--------------------------

```python
def test_ref_from_json_schema():
    # Test basic reference creation
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is definitions

    # Test with different reference path
    data = {"$ref": "#/definitions/Address"}
    result = ref_from_json_schema(data, definitions)
    assert result.to == "#/definitions/Address"

    # Test that function raises assertion error for non-standard ref
    data = {"$ref": "http://example.com/schema.json"}
    try:
        ref_from_json_schema(data, definitions)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "Unsupported $ref style" in str(e)

    # Test with empty definitions
    empty_definitions = Definitions()
    data = {"$ref": "#/components/schemas/Test"}
    result = ref_from_json_schema(data, empty_definitions)
    assert result.definitions is empty_definitions


# LLM-generated content at query #4
#--------------------------

```python
def test_ref_from_json_schema():
    # Test basic reference creation
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is definitions

    # Test reference with different path
    data = {"$ref": "#/definitions/Address"}
    result = ref_from_json_schema(data, definitions)
    assert result.to == "#/definitions/Address"

    # Test that it raises assertion error for non-#/ references
    try:
        data = {"$ref": "http://example.com/schema"}
        ref_from_json_schema(data, definitions)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "Unsupported $ref style" in str(e)

    # Test with empty definitions
    empty_definitions = Definitions()
    data = {"$ref": "#/components/schemas/Test"}
    result = ref_from_json_schema(data, empty_definitions)
    assert result.definitions is empty_definitions


# LLM-generated content at query #5
#--------------------------

```python
def test_one_of_from_json_schema():
    # Test basic oneOf conversion
    definitions = Definitions()
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
            {"type": "integer"}
        ],
        "default": "test"
    }
    result_with_default = one_of_from_json_schema(data_with_default, definitions)
    assert result_with_default.default == "test"
    
    # Test oneOf with nested schemas
    data_nested = {
        "oneOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "array", "items": {"type": "number"}}
        ]
    }
    result_nested = one_of_from_json_schema(data_nested, definitions)
    assert isinstance(result_nested, OneOf)
    assert len(result_nested.one_of) == 2
    assert isinstance(result_nested.one_of[0], Object)
    assert isinstance(result_nested.one_of[1], Array)
    
    # Test oneOf with references
    definitions["#/components/schemas/Person"] = Object(properties={"name": String()})
    data_with_ref = {
        "oneOf": [
            {"$ref": "#/components/schemas/Person"},
            {"type": "string"}
        ]
    }
    result_with_ref = one_of_from_json_schema(data_with_ref, definitions)
    assert isinstance(result_with_ref, OneOf)
    assert isinstance(result_with_ref.one_of[0], Reference)
    assert isinstance(result_with_ref.one_of[1], String)
    
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


# LLM-generated content at query #6
#--------------------------

```python
def test_if_then_else_from_json_schema():
    definitions = Definitions()
    
    # Test basic if-then structure
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, String)
    assert result.then_clause.min_length == 5
    assert result.else_clause is None
    
    # Test if-then-else structure
    data = {
        "if": {"type": "number", "minimum": 0},
        "then": {"type": "number", "minimum": 0, "maximum": 100},
        "else": {"type": "number", "maximum": -1}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Float)
    assert result.if_clause.minimum == 0
    assert isinstance(result.then_clause, Float)
    assert result.then_clause.minimum == 0
    assert result.then_clause.maximum == 100
    assert isinstance(result.else_clause, Float)
    assert result.else_clause.maximum == -1
    
    # Test with nested schemas
    data = {
        "if": {"$ref": "#/definitions/Positive"},
        "then": {"type": "string"},
        "else": {"type": "boolean"}
    }
    definitions["#/definitions/Positive"] = Float(minimum=0)
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Reference)
    assert isinstance(result.then_clause, String)
    assert isinstance(result.else_clause, Boolean)
    
    # Test with default value
    data = {
        "if": {"type": "array"},
        "then": {"type": "array", "minItems": 2},
        "else": {"type": "array", "maxItems": 1},
        "default": []
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert result.default == []
    
    # Test with complex condition
    data = {
        "if": {"allOf": [
            {"type": "object"},
            {"required": ["status"]}
        ]},
        "then": {"type": "object", "properties": {"status": {"type": "string"}}}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, AllOf)
    assert isinstance(result.then_clause, Object)


# LLM-generated content at query #7
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    result = to_json_schema(field)
    assert result is True

    # Test NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result is False

    # Test String field
    field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with null allowed
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

    # Test Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100, multiple_of=2)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100
    assert result["multipleOf"] == 2

    # Test Float field
    field = Float(allow_null=True, minimum=0.0, maximum=1.0)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test Boolean field
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with single item schema
    field = Array(allow_null=False, min_items=1, max_items=10, items=String(), unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert "items" in result

    # Test Array field with tuple items
    field = Array(items=[String(), Integer()])
    result = to_json_schema(field)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Array field with additional_items as boolean
    field = Array(additional_items=False)
    result = to_json_schema(field)
    assert result["additionalItems"] is False

    # Test Array field with additional_items as schema
    field = Array(additional_items=String())
    result = to_json_schema(field)
    assert isinstance(result["additionalItems"], dict)

    # Test Object field
    field = Object(
        allow_null=True,
        properties={"name": String(), "age": Integer()},
        pattern_properties={"^test_": Boolean()},
        additional_properties=False,
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    result = to_json_schema(field)
    assert result["type"] == ["object", "null"]
    assert "properties" in result
    assert "patternProperties" in result
    assert result["additionalProperties"] is False
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    assert result["required"] == ["name"]

    # Test Object field with additional_properties as schema
    field = Object(additional_properties=String())
    result = to_json_schema(field)
    assert isinstance(result["additionalProperties"], dict)

    # Test Object field with property_names
    field = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(field)
    assert "propertyNames" in result

    # Test Schema field
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "properties" in result
    assert result["required"] == ["name"]

    # Test Choice field
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test IfThenElse field
    field = IfThenElse(
        if_clause=String(min_length=5),
        then_clause=String(max_length=20),
        else_clause=Integer()
    )
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test IfThenElse field with missing clauses
    field = IfThenElse(if_clause=String())
    result = to_json_schema(field)
    assert "if" in result
    assert "then" not in result
    assert "else" not in result

    # Test Not field
    field = Not(negated=String())
    result = to_json_schema(field)
    assert "not" in result

    # Test Reference field
    definitions = Definitions()
    definitions["User"] = Object(properties={"name": String()})
    field = Reference(to="User", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/User"

    # Test with definitions at root
    definitions = Definitions()
    definitions["User"] = Object(properties={"name": String()})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]

    # Test default value handling
    field = String(default="test")
    result = to_json_schema(field)
    assert result["default"] == "test"

    # Test field with no default
    field = String()
    result = to_json_schema(field)
    assert "default" not in result

    # Test Decimal field (should be treated as number)
    field = Decimal(allow_null=False, minimum=0)
    result = to_json_schema(field)
    assert result["type"] == "number"
    assert result["minimum"] == 0

    # Test error for unsupported field type
    class UnsupportedField(Field):
        pass

    field = UnsupportedField()
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)

    # Test error for regex with non-standard flags
    import re
    field = String(pattern=re.compile("^test$", re.IGNORECASE))
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-standard flags" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {"type": "number", "minimum": 0, "maximum": 10}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert not field.allow_null

    # Test integer type
    data = {"type": "integer", "exclusiveMinimum": 5, "multipleOf": 2}
    field = from_json_schema_type(data, "integer", True, Definitions())
    assert isinstance(field, Integer)
    assert field.exclusive_minimum == 5
    assert field.multiple_of == 2
    assert field.allow_null

    # Test string type
    data = {"type": "string", "minLength": 3, "maxLength": 10, "pattern": "^a.*z$"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 3
    assert field.max_length == 10
    assert field.pattern == "^a.*z$"
    assert not field.allow_blank

    # Test boolean type
    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", True, Definitions())
    assert isinstance(field, Boolean)
    assert field.default is True
    assert field.allow_null

    # Test array type with items as object
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.additional_items is True

    # Test array type with items as list
    data = {"type": "array", "items": [{"type": "string"}, {"type": "number"}]}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert len(field.items) == 2
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Float)

    # Test object type with properties
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)

    # Test object type with patternProperties
    data = {"type": "object", "patternProperties": {"^S_": {"type": "string"}}}
    field = from_json_schema_type(data, "object", True, Definitions())
    assert isinstance(field, Object)
    assert "^S_" in field.pattern_properties
    assert isinstance(field.pattern_properties["^S_"], String)
    assert field.allow_null

    # Test object type with additionalProperties as boolean
    data = {"type": "object", "additionalProperties": False}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert field.additional_properties is False

    # Test object type with propertyNames
    data = {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.property_names, String)
    assert field.property_names.pattern == "^[a-z]+$"

    # Test with default values
    data = {"type": "string", "default": "test"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert field.default == "test"

    # Test with coerce_types=False (default)
    data = {"type": "number"}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert field.coerce_types is False

    # Test invalid type_string raises assertion
    try:
        from_json_schema_type({}, "invalid", False, Definitions())
        assert False, "Should have raised assertion error"
    except AssertionError as e:
        assert "Invalid argument type_string='invalid'" in str(e)


# LLM-generated content at query #9
#--------------------------

```python
def test_all_of_from_json_schema():
    # Test basic allOf with simple schemas
    definitions = Definitions()
    data = {
        "allOf": [
            {"type": "string", "minLength": 1},
            {"type": "string", "maxLength": 10}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert all(isinstance(field, String) for field in result.all_of)
    
    # Test allOf with different types
    data = {
        "allOf": [
            {"type": "string"},
            {"type": "string", "pattern": "^[A-Z]+$"}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert all(isinstance(field, String) for field in result.all_of)
    
    # Test allOf with default value
    data = {
        "allOf": [
            {"type": "integer"},
            {"type": "integer", "minimum": 0}
        ],
        "default": 5
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert result.default == 5
    
    # Test allOf with nested schemas
    data = {
        "allOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "object", "required": ["name"]}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert all(isinstance(field, Object) for field in result.all_of)
    
    # Test allOf with references
    definitions["#/components/schemas/Person"] = Object(properties={"name": String()})
    data = {
        "allOf": [
            {"$ref": "#/components/schemas/Person"},
            {"type": "object", "properties": {"age": {"type": "integer"}}}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], Reference)
    assert isinstance(result.all_of[1], Object)
    
    # Test allOf with array types
    data = {
        "allOf": [
            {"type": "array", "items": {"type": "string"}},
            {"type": "array", "minItems": 1}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert all(isinstance(field, Array) for field in result.all_of)
    
    # Test allOf with mixed constraints
    data = {
        "allOf": [
            {"type": "number", "minimum": 0},
            {"type": "number", "maximum": 100}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert all(isinstance(field, Float) for field in result.all_of)
    
    # Test allOf with single element
    data = {
        "allOf": [
            {"type": "boolean"}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 1
    assert isinstance(result.all_of[0], Boolean)
    
    # Test allOf with empty array (edge case)
    data = {
        "allOf": []
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_all_of_from_json_schema():
    definitions = Definitions()
    
    # Test basic allOf with simple schemas
    data = {
        "allOf": [
            {"type": "string", "minLength": 2},
            {"type": "string", "maxLength": 5}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert all(isinstance(field, String) for field in result.all_of)
    assert result.all_of[0].min_length == 2
    assert result.all_of[1].max_length == 5
    
    # Test allOf with default value
    data_with_default = {
        "allOf": [
            {"type": "integer", "minimum": 0},
            {"type": "integer", "maximum": 100}
        ],
        "default": 50
    }
    result = all_of_from_json_schema(data_with_default, definitions)
    assert result.default == 50
    
    # Test allOf with mixed types
    data_mixed = {
        "allOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "object", "required": ["name"]}
        ]
    }
    result = all_of_from_json_schema(data_mixed, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], Object)
    assert isinstance(result.all_of[1], Object)
    
    # Test allOf with nested references
    definitions["#/components/schemas/Person"] = Object(properties={"name": String()})
    data_with_ref = {
        "allOf": [
            {"$ref": "#/components/schemas/Person"},
            {"type": "object", "properties": {"age": {"type": "integer"}}}
        ]
    }
    result = all_of_from_json_schema(data_with_ref, definitions)
    assert isinstance(result, AllOf)
    assert isinstance(result.all_of[0], Reference)
    assert isinstance(result.all_of[1], Object)
    
    # Test allOf with array types
    data_array = {
        "allOf": [
            {"type": "array", "minItems": 1},
            {"type": "array", "maxItems": 10}
        ]
    }
    result = all_of_from_json_schema(data_array, definitions)
    assert isinstance(result, AllOf)
    assert all(isinstance(field, Array) for field in result.all_of)
    assert result.all_of[0].min_items == 1
    assert result.all_of[1].max_items == 10
    
    # Test allOf with boolean types
    data_boolean = {
        "allOf": [
            {"type": "boolean"},
            {"const": True}
        ]
    }
    result = all_of_from_json_schema(data_boolean, definitions)
    assert isinstance(result, AllOf)
    assert isinstance(result.all_of[0], Boolean)
    assert isinstance(result.all_of[1], Const)
    assert result.all_of[1].const == True


# LLM-generated content at query #11
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schemas
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test basic type conversion
    string_schema = {"type": "string"}
    assert isinstance(from_json_schema(string_schema), String)

    integer_schema = {"type": "integer"}
    assert isinstance(from_json_schema(integer_schema), Integer)

    number_schema = {"type": "number"}
    assert isinstance(from_json_schema(number_schema), Number)

    boolean_schema = {"type": "boolean"}
    assert isinstance(from_json_schema(boolean_schema), Boolean)

    array_schema = {"type": "array"}
    assert isinstance(from_json_schema(array_schema), Array)

    object_schema = {"type": "object"}
    assert isinstance(from_json_schema(object_schema), Object)

    # Test enum conversion
    enum_schema = {"enum": ["a", "b", "c"]}
    result = from_json_schema(enum_schema)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test const conversion
    const_schema = {"const": "fixed_value"}
    result = from_json_schema(const_schema)
    assert isinstance(result, Const)
    assert result.const == "fixed_value"

    # Test allOf conversion
    all_of_schema = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    result = from_json_schema(all_of_schema)
    assert isinstance(result, AllOf)

    # Test anyOf conversion
    any_of_schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(any_of_schema)
    assert isinstance(result, Union)

    # Test oneOf conversion
    one_of_schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    result = from_json_schema(one_of_schema)
    assert isinstance(result, OneOf)

    # Test not conversion
    not_schema = {"not": {"type": "string"}}
    result = from_json_schema(not_schema)
    assert isinstance(result, Not)

    # Test if-then-else conversion
    if_schema = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"}
    }
    result = from_json_schema(if_schema)
    assert isinstance(result, IfThenElse)

    # Test $ref handling
    ref_schema = {"$ref": "#/components/schemas/User"}
    result = from_json_schema(ref_schema)
    assert isinstance(result, Reference)

    # Test multiple constraints
    multi_schema = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    }
    result = from_json_schema(multi_schema)
    assert isinstance(result, AllOf)

    # Test empty schema returns Any
    empty_schema = {}
    assert isinstance(from_json_schema(empty_schema), Any)

    # Test with definitions
    definitions = Definitions()
    definitions["#/components/schemas/User"] = String()
    
    schema_with_defs = {
        "type": "object",
        "properties": {
            "user": {"$ref": "#/components/schemas/User"}
        }
    }
    result = from_json_schema(schema_with_defs, definitions=definitions)
    assert isinstance(result, Object)

    # Test array with items schema
    array_items_schema = {
        "type": "array",
        "items": {"type": "string"}
    }
    result = from_json_schema(array_items_schema)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

    # Test object with properties
    object_props_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
    result = from_json_schema(object_props_schema)
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert "age" in result.properties

    # Test string with format
    format_schema = {
        "type": "string",
        "format": "email"
    }
    result = from_json_schema(format_schema)
    assert isinstance(result, String)
    assert result.format == "email"

    # Test numeric constraints
    numeric_schema = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 5
    }
    result = from_json_schema(numeric_schema)
    assert isinstance(result, Number)

    # Test union type
    union_type_schema = {"type": ["string", "number"]}
    result = from_json_schema(union_type_schema)
    assert isinstance(result, Union)

    # Test array with tuple validation
    tuple_schema = {
        "type": "array",
        "items": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result = from_json_schema(tuple_schema)
    assert isinstance(result, Array)
    assert isinstance(result.items, list)
    assert len(result.items) == 2

    # Test object with additionalProperties
    additional_props_schema = {
        "type": "object",
        "additionalProperties": {"type": "string"}
    }
    result = from_json_schema(additional_props_schema)
    assert isinstance(result, Object)

    # Test object with required properties
    required_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    result = from_json_schema(required_schema)
    assert isinstance(result, Object)

    # Test array with uniqueItems
    unique_schema = {
        "type": "array",
        "uniqueItems": True
    }
    result = from_json_schema(unique_schema)
    assert isinstance(result, Array)
    assert result.unique_items is True


# LLM-generated content at query #12
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    result = to_json_schema(field)
    assert result is True

    # Test NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result is False

    # Test String field
    field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with null allowed
    field = String(allow_null=True, min_length=0, allow_blank=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert "minLength" not in result

    # Test Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100, multiple_of=2)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100
    assert result["multipleOf"] == 2

    # Test Float field
    field = Float(allow_null=True, minimum=0.0, maximum=1.0)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test Boolean field
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with single item schema
    item_field = String()
    field = Array(allow_null=False, items=item_field, min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test Array field with tuple items
    field = Array(items=[String(), Integer()], additional_items=False)
    result = to_json_schema(field)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    assert result["additionalItems"] is False

    # Test Object field
    field = Object(
        allow_null=True,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=2,
        additional_properties=False
    )
    result = to_json_schema(field)
    assert result["type"] == ["object", "null"]
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 2
    assert result["additionalProperties"] is False

    # Test Object field with pattern properties
    field = Object(pattern_properties={"^[a-z]+$": String()})
    result = to_json_schema(field)
    assert "^[a-z]+$" in result["patternProperties"]

    # Test Choice field
    field = Choice(choices=[("A", "Option A"), ("B", "Option B")])
    result = to_json_schema(field)
    assert result["enum"] == ["A", "B"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(field)
    assert len(result["allOf"]) == 2

    # Test IfThenElse field
    field = IfThenElse(
        if_clause=String(min_length=5),
        then_clause=String(max_length=20),
        else_clause=Integer()
    )
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test Not field
    field = Not(negated=String())
    result = to_json_schema(field)
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test Reference field with definitions
    definitions = Definitions({"User": Object(properties={"name": String()})})
    field = Reference(to="User", definitions=definitions)
    result = to_json_schema(field, _definitions={})
    assert result["$ref"] == "#/components/schemas/User"

    # Test Schema field
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]

    # Test with default value
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result["default"] == "default_value"

    # Test error for unsupported field type
    class UnsupportedField(Field):
        pass

    field = UnsupportedField()
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)

    # Test with root definitions
    field = Reference(to="User", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/User"
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]


# LLM-generated content at query #13
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    result = to_json_schema(field)
    assert result == True

    # Test NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result == False

    # Test String field
    field = String(allow_null=True, min_length=5, max_length=10, pattern="^test.*$", format="email")
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["pattern"] == "^test.*$"
    assert result["format"] == "email"

    # Test Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100, multiple_of=5)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100
    assert result["multipleOf"] == 5

    # Test Float field
    field = Float(allow_null=True, minimum=0.0, maximum=1.0, multiple_of=0.1)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    assert result["multipleOf"] == 0.1

    # Test Boolean field
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with single item schema
    field = Array(
        allow_null=False,
        min_items=1,
        max_items=10,
        items=String(),
        additional_items=False,
        unique_items=True
    )
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["items"]["type"] == "string"
    assert result["additionalItems"] == False
    assert result["uniqueItems"] == True

    # Test Array field with tuple items
    field = Array(
        items=[String(), Integer()],
        additional_items=Boolean()
    )
    result = to_json_schema(field)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    assert result["items"][0]["type"] == "string"
    assert result["items"][1]["type"] == "integer"
    assert result["additionalItems"]["type"] == "boolean"

    # Test Object field
    field = Object(
        allow_null=True,
        properties={"name": String(), "age": Integer()},
        pattern_properties={"^test.*$": Boolean()},
        additional_properties=False,
        property_names=String(pattern="^[a-z]+$"),
        min_properties=1,
        max_properties=5,
        required=["name"]
    )
    result = to_json_schema(field)
    assert result["type"] == ["object", "null"]
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert "^test.*$" in result["patternProperties"]
    assert result["additionalProperties"] == False
    assert result["propertyNames"]["pattern"] == "^[a-z]+$"
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    assert "name" in result["required"]

    # Test Schema field
    field = Schema(
        fields={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert "name" in result["required"]

    # Test Choice field
    field = Choice(choices=[("A", "A"), ("B", "B"), ("C", "C")])
    result = to_json_schema(field)
    assert result["enum"] == ["A", "B", "C"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    field = Union(any_of=[String(), Integer(), Boolean()])
    result = to_json_schema(field)
    assert len(result["anyOf"]) == 3
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"
    assert result["anyOf"][2]["type"] == "boolean"

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["oneOf"]) == 2
    assert result["oneOf"][0]["type"] == "string"
    assert result["oneOf"][1]["type"] == "integer"

    # Test AllOf field
    field = AllOf(all_of=[String(min_length=5), String(max_length=10)])
    result = to_json_schema(field)
    assert len(result["allOf"]) == 2
    assert result["allOf"][0]["minLength"] == 5
    assert result["allOf"][1]["maxLength"] == 10

    # Test IfThenElse field
    field = IfThenElse(
        if_clause=String(pattern="^test.*$"),
        then_clause=Integer(minimum=0),
        else_clause=Boolean()
    )
    result = to_json_schema(field)
    assert result["if"]["pattern"] == "^test.*$"
    assert result["then"]["minimum"] == 0
    assert result["else"]["type"] == "boolean"

    # Test Not field
    field = Not(negated=String(pattern="^test.*$"))
    result = to_json_schema(field)
    assert result["not"]["pattern"] == "^test.*$"

    # Test Reference field
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/Person"

    # Test with definitions at root level
    field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(field, _definitions={})
    assert result["$ref"] == "#/components/schemas/Person"

    # Test default values
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result["default"] == "default_value"

    # Test error for unsupported field type
    class UnsupportedField(Field):
        pass

    field = UnsupportedField()
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)

    # Test pattern regex with non-unicode flags raises error
    import re
    field = String(pattern=re.compile("^test.*$", re.IGNORECASE))
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError for non-unicode flags"
    except ValueError as e:
        assert "non-standard flags" in str(e)


# LLM-generated content at query #14
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {"type": "number", "minimum": 0, "maximum": 10}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.allow_null == False

    # Test integer type
    data = {"type": "integer", "exclusiveMinimum": 5, "multipleOf": 2}
    field = from_json_schema_type(data, "integer", True, Definitions())
    assert isinstance(field, Integer)
    assert field.exclusive_minimum == 5
    assert field.multiple_of == 2
    assert field.allow_null == True

    # Test string type
    data = {"type": "string", "minLength": 3, "maxLength": 10, "pattern": "^a.*z$"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 3
    assert field.max_length == 10
    assert field.pattern == "^a.*z$"
    assert field.allow_blank == False

    # Test boolean type
    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", True, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True
    assert field.allow_null == True

    # Test array type with single items schema
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.additional_items == True

    # Test array type with list of items
    data = {"type": "array", "items": [{"type": "string"}, {"type": "number"}]}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, list)
    assert len(field.items) == 2
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Float)

    # Test object type with properties
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]

    # Test object type with patternProperties
    data = {
        "type": "object",
        "patternProperties": {"^S_": {"type": "string"}},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "^S_" in field.pattern_properties
    assert isinstance(field.pattern_properties["^S_"], String)

    # Test object type with additionalProperties as boolean
    data = {"type": "object", "additionalProperties": False}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert field.additional_properties == False

    # Test object type with propertyNames
    data = {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.property_names, String)
    assert field.property_names.pattern == "^[a-z]+$"

    # Test with default values
    data = {"type": "string", "default": "test"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert field.default == "test"

    # Test with allow_null=True
    data = {"type": "number"}
    field = from_json_schema_type(data, "number", True, Definitions())
    assert field.allow_null == True

    # Test array with additionalItems as schema
    data = {"type": "array", "additionalItems": {"type": "string"}}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field.additional_items, String)

    # Test object with additionalProperties as schema
    data = {"type": "object", "additionalProperties": {"type": "number"}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field.additional_properties, Float)


# LLM-generated content at query #15
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {"type": "number", "minimum": 0, "maximum": 10}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert not field.allow_null

    # Test integer type
    data = {"type": "integer", "exclusiveMinimum": 5, "multipleOf": 2}
    field = from_json_schema_type(data, "integer", True, Definitions())
    assert isinstance(field, Integer)
    assert field.exclusive_minimum == 5
    assert field.multiple_of == 2
    assert field.allow_null

    # Test string type
    data = {"type": "string", "minLength": 3, "maxLength": 10, "pattern": "^a.*z$"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 3
    assert field.max_length == 10
    assert field.pattern == "^a.*z$"
    assert not field.allow_blank

    # Test string type with minLength 0
    data = {"type": "string", "minLength": 0}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert field.allow_blank

    # Test boolean type
    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", True, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True
    assert field.allow_null

    # Test array type with single items schema
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.additional_items == True

    # Test array type with list of items schemas
    data = {"type": "array", "items": [{"type": "string"}, {"type": "number"}]}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, list)
    assert len(field.items) == 2
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Float)

    # Test array type with additionalItems as boolean
    data = {"type": "array", "additionalItems": False}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert field.additional_items == False

    # Test array type with additionalItems as schema
    data = {"type": "array", "additionalItems": {"type": "integer"}}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field.additional_items, Integer)

    # Test object type with properties
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)

    # Test object type with patternProperties
    data = {"type": "object", "patternProperties": {"^x_": {"type": "number"}}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert "^x_" in field.pattern_properties
    assert isinstance(field.pattern_properties["^x_"], Float)

    # Test object type with additionalProperties as boolean
    data = {"type": "object", "additionalProperties": False}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert field.additional_properties == False

    # Test object type with additionalProperties as schema
    data = {"type": "object", "additionalProperties": {"type": "boolean"}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field.additional_properties, Boolean)

    # Test object type with propertyNames
    data = {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field.property_names, String)
    assert field.property_names.pattern == "^[a-z]+$"

    # Test object type with required fields
    data = {"type": "object", "required": ["id", "name"]}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert field.required == ["id", "name"]

    # Test with default values
    data = {"type": "string", "default": "test"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert field.default == "test"

    # Test with coerce_types=False
    data = {"type": "integer"}
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert field.coerce_types == False


# LLM-generated content at query #16
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schemas
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test basic type conversion
    schema = {"type": "string"}
    result = from_json_schema(schema)
    assert isinstance(result, String)

    schema = {"type": "integer"}
    result = from_json_schema(schema)
    assert isinstance(result, Integer)

    schema = {"type": "number"}
    result = from_json_schema(schema)
    assert isinstance(result, Number)

    schema = {"type": "boolean"}
    result = from_json_schema(schema)
    assert isinstance(result, Boolean)

    # Test array type
    schema = {"type": "array"}
    result = from_json_schema(schema)
    assert isinstance(result, Array)

    # Test object type
    schema = {"type": "object"}
    result = from_json_schema(schema)
    assert isinstance(result, Object)

    # Test enum
    schema = {"enum": ["a", "b", "c"]}
    result = from_json_schema(schema)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test const
    schema = {"const": "fixed_value"}
    result = from_json_schema(schema)
    assert isinstance(result, Const)
    assert result.const == "fixed_value"

    # Test allOf
    schema = {"allOf": [{"type": "string"}, {"minLength": 3}]}
    result = from_json_schema(schema)
    assert isinstance(result, AllOf)

    # Test anyOf
    schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(schema)
    assert isinstance(result, OneOf)

    # Test oneOf
    schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(schema)
    assert isinstance(result, OneOf)

    # Test not
    schema = {"not": {"type": "string"}}
    result = from_json_schema(schema)
    assert isinstance(result, Not)

    # Test if-then-else
    schema = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    }
    result = from_json_schema(schema)
    assert isinstance(result, IfThenElse)

    # Test $ref
    schema = {"$ref": "#/components/schemas/User"}
    result = from_json_schema(schema)
    assert isinstance(result, Reference)

    # Test multiple constraints
    schema = {
        "type": "string",
        "minLength": 3,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    }
    result = from_json_schema(schema)
    assert isinstance(result, AllOf)

    # Test empty schema (should return Any)
    schema = {}
    result = from_json_schema(schema)
    assert isinstance(result, Any)

    # Test with definitions
    definitions = Definitions()
    schema = {"type": "string"}
    result = from_json_schema(schema, definitions=definitions)
    assert isinstance(result, String)

    # Test with components/schemas
    schema = {
        "components": {
            "schemas": {
                "User": {"type": "string"}
            }
        }
    }
    result = from_json_schema(schema)
    assert isinstance(result, Any)


# LLM-generated content at query #17
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schemas
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test $ref handling
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    result = from_json_schema({"$ref": "#/components/schemas/Test"}, definitions)
    assert isinstance(result, Reference)
    assert result.ref == "#/components/schemas/Test"

    # Test type constraints - string
    result = from_json_schema({"type": "string"})
    assert isinstance(result, String)
    result = from_json_schema({"type": "string", "minLength": 5})
    assert isinstance(result, String)
    assert result.min_length == 5

    # Test type constraints - integer
    result = from_json_schema({"type": "integer"})
    assert isinstance(result, Integer)
    result = from_json_schema({"type": "integer", "minimum": 0})
    assert isinstance(result, Integer)
    assert result.minimum == 0

    # Test type constraints - number
    result = from_json_schema({"type": "number"})
    assert isinstance(result, Number)
    result = from_json_schema({"type": "number", "maximum": 100})
    assert isinstance(result, Number)
    assert result.maximum == 100

    # Test type constraints - boolean
    result = from_json_schema({"type": "boolean"})
    assert isinstance(result, Boolean)

    # Test type constraints - array
    result = from_json_schema({"type": "array"})
    assert isinstance(result, Array)
    result = from_json_schema({"type": "array", "items": {"type": "string"}})
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

    # Test type constraints - object
    result = from_json_schema({"type": "object"})
    assert isinstance(result, Object)
    result = from_json_schema({
        "type": "object",
        "properties": {"name": {"type": "string"}}
    })
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert isinstance(result.properties["name"], String)

    # Test enum
    result = from_json_schema({"enum": ["a", "b", "c"]})
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test const
    result = from_json_schema({"const": "fixed_value"})
    assert isinstance(result, Const)
    assert result.const == "fixed_value"

    # Test allOf
    result = from_json_schema({
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    })
    assert isinstance(result, AllOf)
    assert len(result.fields) == 2

    # Test anyOf
    result = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

    # Test oneOf
    result = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2

    # Test not
    result = from_json_schema({"not": {"type": "string"}})
    assert isinstance(result, Not)
    assert isinstance(result.negated, String)

    # Test if/then/else
    result = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    })
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, String)
    assert isinstance(result.else_clause, Integer)

    # Test multiple constraints combined
    result = from_json_schema({
        "type": "string",
        "enum": ["a", "b", "c"],
        "minLength": 1
    })
    assert isinstance(result, AllOf)
    assert len(result.fields) == 2

    # Test empty schema returns Any
    result = from_json_schema({})
    assert isinstance(result, Any)

    # Test definitions from components
    data = {
        "components": {
            "schemas": {
                "TestSchema": {"type": "string", "minLength": 5}
            }
        }
    }
    result = from_json_schema(data)
    assert isinstance(result, Any)


# LLM-generated content at query #18
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schemas
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test basic type conversion
    string_schema = {"type": "string"}
    assert isinstance(from_json_schema(string_schema), String)

    number_schema = {"type": "number"}
    assert isinstance(from_json_schema(number_schema), Number)

    integer_schema = {"type": "integer"}
    assert isinstance(from_json_schema(integer_schema), Integer)

    boolean_schema = {"type": "boolean"}
    assert isinstance(from_json_schema(boolean_schema), Boolean)

    array_schema = {"type": "array"}
    assert isinstance(from_json_schema(array_schema), Array)

    object_schema = {"type": "object"}
    assert isinstance(from_json_schema(object_schema), Object)

    # Test with constraints
    min_length_schema = {"type": "string", "minLength": 5}
    field = from_json_schema(min_length_schema)
    assert isinstance(field, String)
    assert field.min_length == 5

    # Test enum
    enum_schema = {"enum": ["a", "b", "c"]}
    field = from_json_schema(enum_schema)
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test const
    const_schema = {"const": "fixed_value"}
    field = from_json_schema(const_schema)
    assert isinstance(field, Const)
    assert field.const == "fixed_value"

    # Test allOf
    all_of_schema = {
        "allOf": [
            {"type": "string", "minLength": 3},
            {"type": "string", "maxLength": 10}
        ]
    }
    field = from_json_schema(all_of_schema)
    assert isinstance(field, AllOf)

    # Test anyOf
    any_of_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    field = from_json_schema(any_of_schema)
    assert isinstance(field, OneOf)

    # Test oneOf
    one_of_schema = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    field = from_json_schema(one_of_schema)
    assert isinstance(field, OneOf)

    # Test not
    not_schema = {"not": {"type": "string"}}
    field = from_json_schema(not_schema)
    assert isinstance(field, Not)

    # Test if-then-else
    if_then_else_schema = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "number"}
    }
    field = from_json_schema(if_then_else_schema)
    assert isinstance(field, IfThenElse)

    # Test $ref
    ref_schema = {"$ref": "#/components/schemas/User"}
    field = from_json_schema(ref_schema)
    assert isinstance(field, Reference)

    # Test multiple constraints
    multi_schema = {
        "type": "string",
        "minLength": 3,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    }
    field = from_json_schema(multi_schema)
    assert isinstance(field, String)
    assert field.min_length == 3
    assert field.max_length == 10
    assert field.pattern == re.compile("^[a-z]+$")

    # Test empty schema returns Any
    empty_schema = {}
    assert isinstance(from_json_schema(empty_schema), Any)

    # Test with definitions
    definitions = Definitions()
    schema_with_defs = {
        "components": {
            "schemas": {
                "User": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        }
    }
    field = from_json_schema(schema_with_defs, definitions)
    assert isinstance(field, Any)


# LLM-generated content at query #19
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    any_field = Any()
    result = to_json_schema(any_field)
    assert result is True

    # Test NeverMatch field
    never_field = NeverMatch()
    result = to_json_schema(never_field)
    assert result is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10

    # Test String field with null allowed
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

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
    object_field = Object(
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=5
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5

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
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test Reference field
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    ref_field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/Person"

    # Test Definitions conversion
    definitions = Definitions()
    definitions["User"] = Object(properties={"id": Integer()})
    definitions["Profile"] = Object(properties={"user": Reference(to="User", definitions=definitions)})
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    assert "Profile" in result["components"]["schemas"]

    # Test default value handling
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"

    # Test pattern properties
    object_with_pattern = Object(
        pattern_properties={r"^test_": String()}
    )
    result = to_json_schema(object_with_pattern)
    assert "patternProperties" in result
    assert r"^test_" in result["patternProperties"]

    # Test additional properties
    object_with_additional = Object(additional_properties=False)
    result = to_json_schema(object_with_additional)
    assert result["additionalProperties"] is False

    # Test property names constraint
    object_with_names = Object(property_names=String(pattern="^[a-z]+$"))
    result = to_json_schema(object_with_names)
    assert "propertyNames" in result
    assert result["propertyNames"]["pattern"] == "^[a-z]+$"

    # Test exclusive minimum/maximum
    number_field = Float(exclusive_minimum=0, exclusive_maximum=100)
    result = to_json_schema(number_field)
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100

    # Test multipleOf
    integer_field = Integer(multiple_of=2)
    result = to_json_schema(integer_field)
    assert result["multipleOf"] == 2

    # Test unique items
    array_unique = Array(items=String(), unique_items=True)
    result = to_json_schema(array_unique)
    assert result["uniqueItems"] is True

    # Test format
    string_with_format = String(format="email")
    result = to_json_schema(string_with_format)
    assert result["format"] == "email"

    # Test IfThenElse field
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test Schema field (special Object type)
    schema_field = Schema(fields={"id": Integer()}, required=["id"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "id" in result["properties"]
    assert result["required"] == ["id"]

    # Test Decimal field
    decimal_field = Decimal(minimum=0, maximum=100)
    result = to_json_schema(decimal_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0
    assert result["maximum"] == 100

    # Test with nested definitions
    inner_definitions = Definitions()
    inner_definitions["Inner"] = String()
    outer_field = Object(
        properties={"inner": Reference(to="Inner", definitions=inner_definitions)}
    )
    result = to_json_schema(outer_field, _definitions=inner_definitions)
    assert "Inner" in result.get("components", {}).get("schemas", {})


# LLM-generated content at query #20
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schemas
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test basic type constraints
    string_schema = {"type": "string"}
    result = from_json_schema(string_schema)
    assert isinstance(result, String)

    integer_schema = {"type": "integer"}
    result = from_json_schema(integer_schema)
    assert isinstance(result, Integer)

    number_schema = {"type": "number"}
    result = from_json_schema(number_schema)
    assert isinstance(result, Number)

    boolean_schema = {"type": "boolean"}
    result = from_json_schema(boolean_schema)
    assert isinstance(result, Boolean)

    array_schema = {"type": "array"}
    result = from_json_schema(array_schema)
    assert isinstance(result, Array)

    object_schema = {"type": "object"}
    result = from_json_schema(object_schema)
    assert isinstance(result, Object)

    # Test enum constraint
    enum_schema = {"enum": ["a", "b", "c"]}
    result = from_json_schema(enum_schema)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test const constraint
    const_schema = {"const": "fixed_value"}
    result = from_json_schema(const_schema)
    assert isinstance(result, Const)
    assert result.const == "fixed_value"

    # Test allOf constraint
    all_of_schema = {"allOf": [{"type": "string"}, {"minLength": 1}]}
    result = from_json_schema(all_of_schema)
    assert isinstance(result, AllOf)

    # Test anyOf constraint
    any_of_schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(any_of_schema)
    assert isinstance(result, Union)

    # Test oneOf constraint
    one_of_schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(one_of_schema)
    assert isinstance(result, OneOf)

    # Test not constraint
    not_schema = {"not": {"type": "string"}}
    result = from_json_schema(not_schema)
    assert isinstance(result, Not)

    # Test if-then-else constraint
    if_schema = {
        "if": {"type": "string"},
        "then": {"minLength": 1},
        "else": {"type": "integer"}
    }
    result = from_json_schema(if_schema)
    assert isinstance(result, IfThenElse)

    # Test multiple constraints
    multi_schema = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10
    }
    result = from_json_schema(multi_schema)
    assert isinstance(result, AllOf)

    # Test empty schema (should return Any)
    empty_schema = {}
    result = from_json_schema(empty_schema)
    assert isinstance(result, Any)

    # Test with definitions
    definitions = Definitions()
    ref_schema = {"$ref": "#/definitions/MyType"}
    result = from_json_schema(ref_schema, definitions=definitions)
    assert isinstance(result, Reference)

    # Test with components/schemas
    component_schema = {
        "components": {
            "schemas": {
                "User": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        }
    }
    result = from_json_schema(component_schema)
    assert isinstance(result, Any)


# LLM-generated content at query #21
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    result = to_json_schema(field)
    assert result == True

    # Test NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result == False

    # Test String field
    field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"

    # Test String field with null allowed
    field = String(allow_null=True, allow_blank=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert "minLength" not in result

    # Test Integer field
    field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=False)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == True
    assert "exclusiveMaximum" not in result

    # Test Float field
    field = Float(multiple_of=0.5)
    result = to_json_schema(field)
    assert result["type"] == "number"
    assert result["multipleOf"] == 0.5

    # Test Boolean field
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with single item schema
    field = Array(items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    assert "items" in result

    # Test Array field with tuple items
    field = Array(items=[String(), Integer()])
    result = to_json_schema(field)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Test Object field
    field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=2
    )
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 2

    # Test Object field with pattern properties
    field = Object(pattern_properties={"^[a-z]+$": Integer()})
    result = to_json_schema(field)
    assert "^[a-z]+$" in result["patternProperties"]

    # Test Choice field
    field = Choice(choices=[("A", "Option A"), ("B", "Option B")])
    result = to_json_schema(field)
    assert result["enum"] == ["A", "B"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(field)
    assert len(result["allOf"]) == 2

    # Test IfThenElse field
    field = IfThenElse(
        if_clause=String(min_length=5),
        then_clause=Integer(minimum=10),
        else_clause=Boolean()
    )
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test Not field
    field = Not(negated=String())
    result = to_json_schema(field)
    assert "not" in result

    # Test Reference field
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/Person"

    # Test with definitions
    person_schema = Object(properties={"name": String()})
    result = to_json_schema(person_schema, _definitions={"Person": person_schema})
    assert "components" not in result

    # Test root with definitions
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "Person" in result["components"]["schemas"]

    # Test default value handling
    field = String(default="test")
    result = to_json_schema(field)
    assert result["default"] == "test"

    # Test Schema field (subclass of Object)
    class PersonSchema(Schema):
        name = String()
        age = Integer()

    field = PersonSchema()
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert "name" in result["required"]

    # Test Decimal field
    field = Decimal(minimum=0, maximum=100)
    result = to_json_schema(field)
    assert result["type"] == "number"
    assert result["minimum"] == 0
    assert result["maximum"] == 100

    # Test field with no type conversion
    class CustomField(Field):
        pass

    field = CustomField()
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


# LLM-generated content at query #22
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test String field with null allowed
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=False)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == True
    assert "exclusiveMaximum" not in result
    
    # Test Float field
    float_field = Float(multiple_of=0.5)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["multipleOf"] == 0.5
    
    # Test Boolean field
    bool_field = Boolean(allow_null=True)
    result = to_json_schema(bool_field)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(
        items=String(),
        min_items=1,
        max_items=10,
        unique_items=True,
        additional_items=False
    )
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    assert result["additionalItems"] == False
    assert "items" in result
    
    # Test Object field
    object_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=5
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    
    # Test Object field with pattern properties
    object_field_pattern = Object(
        pattern_properties={"^test_": String()},
        additional_properties=False
    )
    result = to_json_schema(object_field_pattern)
    assert "^test_" in result["patternProperties"]
    assert result["additionalProperties"] == False
    
    # Test Choice field
    choice_field = Choice(choices=[("A", "Option A"), ("B", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["A", "B"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    oneof_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(oneof_field)
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    allof_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(allof_field)
    assert len(result["allOf"]) == 2
    
    # Test IfThenElse field
    ifthenelse_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(ifthenelse_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test Reference field with definitions
    definitions = Definitions({"User": Object(properties={"name": String()})})
    reference_field = Reference(to="User", definitions=definitions)
    result = to_json_schema(reference_field)
    assert result["$ref"] == "#/components/schemas/User"
    
    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test with default value
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"
    
    # Test root definitions
    user_schema = Object(properties={"name": String()})
    result = to_json_schema(user_schema, _definitions={"User": user_schema})
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    
    # Test error for unsupported field type
    class UnsupportedField(Field):
        pass
    
    unsupported_field = UnsupportedField()
    try:
        to_json_schema(unsupported_field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ref_from_json_schema():
    # Test basic reference creation
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/User"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/User"
    assert result.definitions is definitions

    # Test reference with different path
    data = {"$ref": "#/definitions/Address"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/definitions/Address"
    assert result.definitions is definitions

    # Test that function raises AssertionError for non-#/ references
    try:
        data = {"$ref": "http://example.com/schema.json"}
        ref_from_json_schema(data, definitions)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "Unsupported $ref style" in str(e)

    # Test with empty definitions
    empty_definitions = Definitions()
    data = {"$ref": "#/components/schemas/Product"}
    result = ref_from_json_schema(data, empty_definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/Product"
    assert result.definitions is empty_definitions


# LLM-generated content at query #2
#--------------------------

```python
def test_if_then_else_from_json_schema():
    definitions = Definitions()
    
    # Test basic if-then structure
    data = {
        "if": {"type": "string", "minLength": 5},
        "then": {"type": "string", "pattern": "^[A-Z].*"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert result.if_clause.min_length == 5
    assert isinstance(result.then_clause, String)
    assert result.then_clause.pattern is not None
    assert result.else_clause is None
    
    # Test if-then-else structure
    data = {
        "if": {"type": "number", "minimum": 0},
        "then": {"type": "number", "maximum": 100},
        "else": {"type": "number", "maximum": -1}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Float)
    assert result.if_clause.minimum == 0
    assert isinstance(result.then_clause, Float)
    assert result.then_clause.maximum == 100
    assert isinstance(result.else_clause, Float)
    assert result.else_clause.maximum == -1
    
    # Test with nested schemas
    data = {
        "if": {"allOf": [{"type": "object"}, {"required": ["active"]}]},
        "then": {"type": "object", "properties": {"status": {"type": "string"}}},
        "else": {"type": "object", "properties": {"error": {"type": "string"}}}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, AllOf)
    assert isinstance(result.then_clause, Object)
    assert isinstance(result.else_clause, Object)
    
    # Test with default value
    data = {
        "if": {"type": "boolean"},
        "then": {"type": "string", "const": "yes"},
        "else": {"type": "string", "const": "no"},
        "default": "maybe"
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert result.default == "maybe"
    
    # Test with only if clause (no then/else)
    data = {
        "if": {"type": "array", "minItems": 1}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Array)
    assert result.if_clause.min_items == 1
    assert result.then_clause is None
    assert result.else_clause is None
    
    # Test with complex nested conditions
    data = {
        "if": {"anyOf": [{"type": "string"}, {"type": "number"}]},
        "then": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        "else": {"not": {"type": "boolean"}}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Union)
    assert isinstance(result.then_clause, OneOf)
    assert isinstance(result.else_clause, Not)
    
    # Test with references in definitions
    definitions["#/components/schemas/User"] = Object(properties={"name": String()})
    data = {
        "if": {"$ref": "#/components/schemas/User"},
        "then": {"type": "string"},
        "else": {"type": "number"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, Reference)
    assert result.if_clause.to == "#/components/schemas/User"
    assert isinstance(result.then_clause, String)
    assert isinstance(result.else_clause, Float)


# LLM-generated content at query #3
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schemas
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test basic type conversion
    schema = {"type": "string"}
    result = from_json_schema(schema)
    assert isinstance(result, String)

    schema = {"type": "integer"}
    result = from_json_schema(schema)
    assert isinstance(result, Integer)

    schema = {"type": "number"}
    result = from_json_schema(schema)
    assert isinstance(result, Number)

    schema = {"type": "boolean"}
    result = from_json_schema(schema)
    assert isinstance(result, Boolean)

    schema = {"type": "array"}
    result = from_json_schema(schema)
    assert isinstance(result, Array)

    schema = {"type": "object"}
    result = from_json_schema(schema)
    assert isinstance(result, Object)

    # Test enum conversion
    schema = {"enum": ["a", "b", "c"]}
    result = from_json_schema(schema)
    assert isinstance(result, Choice)
    assert result.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test const conversion
    schema = {"const": "fixed_value"}
    result = from_json_schema(schema)
    assert isinstance(result, Const)
    assert result.const == "fixed_value"

    # Test allOf conversion
    schema = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    result = from_json_schema(schema)
    assert isinstance(result, AllOf)

    # Test anyOf conversion
    schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(schema)
    assert isinstance(result, Union)

    # Test oneOf conversion
    schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(schema)
    assert isinstance(result, OneOf)

    # Test not conversion
    schema = {"not": {"type": "string"}}
    result = from_json_schema(schema)
    assert isinstance(result, Not)

    # Test if-then-else conversion
    schema = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    }
    result = from_json_schema(schema)
    assert isinstance(result, IfThenElse)

    # Test combination of constraints
    schema = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10
    }
    result = from_json_schema(schema)
    assert isinstance(result, AllOf)

    # Test empty schema returns Any
    schema = {}
    result = from_json_schema(schema)
    assert isinstance(result, Any)

    # Test with definitions
    schema = {
        "$ref": "#/components/schemas/User",
        "components": {
            "schemas": {
                "User": {"type": "string"}
            }
        }
    }
    result = from_json_schema(schema)
    assert isinstance(result, Reference)

    # Test array with items schema
    schema = {
        "type": "array",
        "items": {"type": "string"}
    }
    result = from_json_schema(schema)
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

    # Test object with properties
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
    result = from_json_schema(schema)
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert "age" in result.properties

    # Test string with pattern
    schema = {
        "type": "string",
        "pattern": "^[a-z]+$"
    }
    result = from_json_schema(schema)
    assert isinstance(result, AllOf)
    assert any(isinstance(c, String) for c in result.fields)

    # Test numeric constraints
    schema = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 5
    }
    result = from_json_schema(schema)
    assert isinstance(result, AllOf)

    # Test array constraints
    schema = {
        "type": "array",
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True
    }
    result = from_json_schema(schema)
    assert isinstance(result, AllOf)

    # Test object constraints
    schema = {
        "type": "object",
        "minProperties": 1,
        "maxProperties": 5,
        "required": ["id"]
    }
    result = from_json_schema(schema)
    assert isinstance(result, AllOf)


# LLM-generated content at query #4
#--------------------------

```python
def test_from_json_schema():
    # Test boolean schemas
    assert isinstance(from_json_schema(True), Any)
    assert isinstance(from_json_schema(False), NeverMatch)

    # Test basic type conversion
    string_schema = {"type": "string"}
    assert isinstance(from_json_schema(string_schema), String)

    integer_schema = {"type": "integer"}
    assert isinstance(from_json_schema(integer_schema), Integer)

    number_schema = {"type": "number"}
    assert isinstance(from_json_schema(number_schema), Number)

    boolean_schema = {"type": "boolean"}
    assert isinstance(from_json_schema(boolean_schema), Boolean)

    array_schema = {"type": "array"}
    assert isinstance(from_json_schema(array_schema), Array)

    object_schema = {"type": "object"}
    assert isinstance(from_json_schema(object_schema), Object)

    # Test enum conversion
    enum_schema = {"enum": ["a", "b", "c"]}
    field = from_json_schema(enum_schema)
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test const conversion
    const_schema = {"const": "fixed_value"}
    field = from_json_schema(const_schema)
    assert isinstance(field, Const)
    assert field.const == "fixed_value"

    # Test allOf conversion
    all_of_schema = {"allOf": [{"type": "string"}, {"minLength": 3}]}
    field = from_json_schema(all_of_schema)
    assert isinstance(field, AllOf)

    # Test anyOf conversion
    any_of_schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    field = from_json_schema(any_of_schema)
    assert isinstance(field, Union)

    # Test oneOf conversion
    one_of_schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    field = from_json_schema(one_of_schema)
    assert isinstance(field, OneOf)

    # Test not conversion
    not_schema = {"not": {"type": "string"}}
    field = from_json_schema(not_schema)
    assert isinstance(field, Not)

    # Test if-then-else conversion
    if_schema = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    }
    field = from_json_schema(if_schema)
    assert isinstance(field, IfThenElse)

    # Test $ref handling
    ref_schema = {"$ref": "#/components/schemas/User"}
    field = from_json_schema(ref_schema)
    assert isinstance(field, Reference)

    # Test multiple constraints
    multi_schema = {
        "type": "string",
        "minLength": 3,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    }
    field = from_json_schema(multi_schema)
    assert isinstance(field, AllOf)

    # Test empty schema returns Any
    empty_schema = {}
    assert isinstance(from_json_schema(empty_schema), Any)

    # Test with definitions
    definitions = Definitions()
    schema_with_defs = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        }
    }
    field = from_json_schema(schema_with_defs, definitions=definitions)
    assert isinstance(field, Object)

    # Test array with items schema
    array_items_schema = {
        "type": "array",
        "items": {"type": "string"}
    }
    field = from_json_schema(array_items_schema)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)

    # Test object with properties
    object_props_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    field = from_json_schema(object_props_schema)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert "name" in field.required

    # Test numeric constraints
    numeric_schema = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 2
    }
    field = from_json_schema(numeric_schema)
    assert isinstance(field, AllOf)

    # Test string with format
    format_schema = {
        "type": "string",
        "format": "email"
    }
    field = from_json_schema(format_schema)
    assert isinstance(field, String)
    assert field.format == "email"

    # Test array with tuple validation
    tuple_schema = {
        "type": "array",
        "items": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    field = from_json_schema(tuple_schema)
    assert isinstance(field, Array)
    assert isinstance(field.items, list)
    assert len(field.items) == 2
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Integer)


# LLM-generated content at query #5
#--------------------------

```python
def test_enum_from_json_schema():
    definitions = Definitions()
    
    # Test basic enum with string values
    data = {"enum": ["red", "green", "blue"]}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [("red", "red"), ("green", "green"), ("blue", "blue")]
    assert field.default == NO_DEFAULT
    
    # Test enum with numeric values
    data = {"enum": [1, 2, 3]}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [(1, 1), (2, 2), (3, 3)]
    
    # Test enum with mixed types
    data = {"enum": ["text", 42, True, None]}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [("text", "text"), (42, 42), (True, True), (None, None)]
    
    # Test enum with default value
    data = {"enum": ["a", "b", "c"], "default": "b"}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.default == "b"
    
    # Test enum with empty list (should work but might be edge case)
    data = {"enum": []}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == []
    
    # Test enum with single value
    data = {"enum": ["only"]}
    field = enum_from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [("only", "only")]


# LLM-generated content at query #6
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {"type": "number", "minimum": 0, "maximum": 10}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.allow_null == False

    # Test integer type
    data = {"type": "integer", "exclusiveMinimum": 5, "multipleOf": 2}
    field = from_json_schema_type(data, "integer", True, Definitions())
    assert isinstance(field, Integer)
    assert field.exclusive_minimum == 5
    assert field.multiple_of == 2
    assert field.allow_null == True

    # Test string type
    data = {"type": "string", "minLength": 3, "pattern": "^[A-Z]+$"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 3
    assert field.pattern == "^[A-Z]+$"
    assert field.allow_null == False

    # Test boolean type
    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", True, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True
    assert field.allow_null == True

    # Test array type with list items
    definitions = Definitions()
    data = {"type": "array", "items": [{"type": "string"}, {"type": "number"}]}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, list)
    assert len(field.items) == 2
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Float)

    # Test array type with single item schema
    data = {"type": "array", "items": {"type": "integer"}}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, Integer)

    # Test object type with properties
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert "age" in field.properties
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]

    # Test object type with patternProperties
    data = {
        "type": "object",
        "patternProperties": {"^S_": {"type": "string"}},
    }
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert "^S_" in field.pattern_properties
    assert isinstance(field.pattern_properties["^S_"], String)

    # Test object type with additionalProperties as boolean
    data = {"type": "object", "additionalProperties": False}
    field = from_json_schema_type(data, "object", False, definitions)
    assert field.additional_properties == False

    # Test object type with additionalProperties as schema
    data = {"type": "object", "additionalProperties": {"type": "number"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field.additional_properties, Float)

    # Test with allow_null=True
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, definitions)
    assert field.allow_null == True

    # Test with default value
    data = {"type": "integer", "default": 42}
    field = from_json_schema_type(data, "integer", False, definitions)
    assert field.default == 42

    # Test array with additionalItems as boolean
    data = {"type": "array", "additionalItems": False}
    field = from_json_schema_type(data, "array", False, definitions)
    assert field.additional_items == False

    # Test array with additionalItems as schema
    data = {"type": "array", "additionalItems": {"type": "boolean"}}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field.additional_items, Boolean)

    # Test object with propertyNames
    data = {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field.property_names, String)
    assert field.property_names.pattern == "^[a-z]+$"


# LLM-generated content at query #7
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    result = to_json_schema(field)
    assert result is True

    # Test NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result is False

    # Test String field
    field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with null allowed
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

    # Test Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100, multiple_of=2)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100
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

    # Test Array field with single item schema
    item_field = String()
    field = Array(allow_null=False, items=item_field, min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test Array field with tuple items
    field = Array(items=[String(), Integer()], additional_items=False)
    result = to_json_schema(field)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    assert result["additionalItems"] is False

    # Test Object field
    properties = {"name": String(), "age": Integer()}
    field = Object(allow_null=False, properties=properties, required=["name"], additional_properties=False)
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["additionalProperties"] is False

    # Test Object field with pattern properties
    pattern_properties = {"^[a-z]+$": Integer()}
    field = Object(pattern_properties=pattern_properties)
    result = to_json_schema(field)
    assert "^[a-z]+$" in result["patternProperties"]

    # Test Choice field
    field = Choice(choices=[("A", "A"), ("B", "B")])
    result = to_json_schema(field)
    assert result["enum"] == ["A", "B"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(field)
    assert len(result["allOf"]) == 2

    # Test IfThenElse field
    if_clause = String(pattern="^[a-z]+$")
    then_clause = Integer(minimum=0)
    else_clause = Boolean()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause, else_clause=else_clause)
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test Not field
    negated = String(pattern="^[0-9]+$")
    field = Not(negated=negated)
    result = to_json_schema(field)
    assert "not" in result
    assert result["not"]["type"] == "string"

    # Test Reference field
    target = String()
    definitions = Definitions({"User": target})
    field = Reference(to="User", definitions=definitions)
    result = to_json_schema(field, _definitions={})
    assert result["$ref"] == "#/components/schemas/User"

    # Test Definitions conversion
    definitions = Definitions({
        "User": Object(properties={"name": String()}),
        "Group": Object(properties={"users": Array(items=Reference(to="User", definitions={}))})
    })
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    assert "Group" in result["components"]["schemas"]

    # Test default value inclusion
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result["default"] == "default_value"

    # Test Schema field (subclass of Object)
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    field = UserSchema()
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]

    # Test error for unsupported field type
    class UnsupportedField(Field):
        pass
    
    field = UnsupportedField()
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_type_from_json_schema():
    # Test with single type string
    data = {"type": "string"}
    definitions = Definitions()
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    assert result.allow_null == False

    # Test with multiple type strings
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Number)
    assert result.allow_null == False

    # Test with null allowed
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, String)
    assert result.allow_null == True

    # Test with only null type
    data = {"type": "null"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)
    assert result.const is None
    assert result.allow_null == True

    # Test with empty type array
    data = {"type": []}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, NeverMatch)

    # Test with empty type array and null
    data = {"type": ["null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Const)
    assert result.const is None
    assert result.allow_null == True

    # Test with integer type
    data = {"type": "integer"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Integer)
    assert result.allow_null == False

    # Test with number type
    data = {"type": "number"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Number)
    assert result.allow_null == False

    # Test with boolean type
    data = {"type": "boolean"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Boolean)
    assert result.allow_null == False

    # Test with array type
    data = {"type": "array"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Array)
    assert result.allow_null == False

    # Test with object type
    data = {"type": "object"}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Object)
    assert result.allow_null == False

    # Test with multiple types including null
    data = {"type": ["string", "integer", "null"]}
    result = type_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Integer)
    assert result.allow_null == True


# LLM-generated content at query #9
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    any_field = Any()
    assert to_json_schema(any_field) == True
    
    # Test NeverMatch field
    never_field = NeverMatch()
    assert to_json_schema(never_field) == False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    
    # Test String field with null allowed
    string_null_field = String(allow_null=True)
    result = to_json_schema(string_null_field)
    assert result["type"] == ["string", "null"]
    
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
    bool_field = Boolean(allow_null=True)
    result = to_json_schema(bool_field)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=10)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert "items" in result
    
    # Test Object field
    object_field = Object(
        properties={"name": String()},
        required=["name"]
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]
    
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
    
    # Test AllOf field
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(all_of_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(one_of_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test IfThenElse field
    if_then_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    result = to_json_schema(if_then_field)
    assert "if" in result
    assert "then" in result
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test Reference field
    ref_field = Reference(to="MySchema", definitions={"MySchema": String()})
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/MySchema"
    
    # Test with definitions
    definitions = {"MySchema": String()}
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "MySchema" in result["components"]["schemas"]
    
    # Test default values
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"
    
    # Test pattern regex
    string_with_pattern = String(pattern_regex=re.compile(r"^\d+$"))
    result = to_json_schema(string_with_pattern)
    assert result["pattern"] == r"^\d+$"
    
    # Test exclusive minimum/maximum
    int_with_exclusive = Integer(exclusive_minimum=0, exclusive_maximum=100)
    result = to_json_schema(int_with_exclusive)
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100
    
    # Test multipleOf
    int_with_multiple = Integer(multiple_of=2)
    result = to_json_schema(int_with_multiple)
    assert result["multipleOf"] == 2
    
    # Test uniqueItems
    array_unique = Array(unique_items=True)
    result = to_json_schema(array_unique)
    assert result["uniqueItems"] == True
    
    # Test additionalProperties as bool
    object_bool_additional = Object(additional_properties=False)
    result = to_json_schema(object_bool_additional)
    assert result["additionalProperties"] == False
    
    # Test additionalProperties as Field
    object_field_additional = Object(additional_properties=String())
    result = to_json_schema(object_field_additional)
    assert "additionalProperties" in result
    assert isinstance(result["additionalProperties"], dict)
    
    # Test propertyNames
    object_with_names = Object(property_names=String(min_length=1))
    result = to_json_schema(object_with_names)
    assert "propertyNames" in result
    
    # Test patternProperties
    object_with_pattern = Object(pattern_properties={r"^\d+$": Integer()})
    result = to_json_schema(object_with_pattern)
    assert "patternProperties" in result
    assert r"^\d+$" in result["patternProperties"]
    
    # Test minProperties/maxProperties
    object_with_size = Object(min_properties=1, max_properties=10)
    result = to_json_schema(object_with_size)
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 10
    
    # Test Schema field (subclass of Object)
    schema_field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test Decimal field
    decimal_field = Decimal(minimum=0.0, maximum=1.0)
    result = to_json_schema(decimal_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0


# LLM-generated content at query #10
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10)
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    
    # Test nullable String field
    nullable_string = String(allow_null=True)
    result = to_json_schema(nullable_string)
    assert result["type"] == ["string", "null"]
    
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
    bool_field = Boolean(allow_null=True)
    result = to_json_schema(bool_field)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert "items" in result
    
    # Test Array with list items
    array_list_items = Array(items=[String(), Integer()])
    result = to_json_schema(array_list_items)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Object field
    obj_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"]
    )
    result = to_json_schema(obj_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test Object with pattern properties
    obj_pattern = Object(pattern_properties={r"\d+": Integer()})
    result = to_json_schema(obj_pattern)
    assert "patternProperties" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("A", "A"), ("B", "B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["A", "B"]
    
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
    allof_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(allof_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test IfThenElse field
    ifthen_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(ifthen_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test Reference field
    ref_field = Reference(to="MySchema", definitions=Definitions({"MySchema": String()}))
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/MySchema"
    
    # Test with definitions
    definitions = Definitions({
        "Person": Object(properties={"name": String()}),
        "Address": Object(properties={"street": String()})
    })
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "Person" in result["components"]["schemas"]
    assert "Address" in result["components"]["schemas"]
    
    # Test default values
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"
    
    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test pattern regex conversion
    string_with_pattern = String(pattern_regex=re.compile(r"^\d+$"))
    result = to_json_schema(string_with_pattern)
    assert result["pattern"] == r"^\d+$"
    
    # Test unique items in Array
    array_unique = Array(unique_items=True)
    result = to_json_schema(array_unique)
    assert result["uniqueItems"] == True
    
    # Test additionalProperties as boolean
    obj_bool_additional = Object(additional_properties=False)
    result = to_json_schema(obj_bool_additional)
    assert result["additionalProperties"] == False
    
    # Test additionalProperties as Field
    obj_field_additional = Object(additional_properties=String())
    result = to_json_schema(obj_field_additional)
    assert isinstance(result["additionalProperties"], dict)
    
    # Test propertyNames
    obj_with_names = Object(property_names=String(min_length=1))
    result = to_json_schema(obj_with_names)
    assert "propertyNames" in result
    
    # Test Decimal field (should be treated as number)
    decimal_field = Decimal()
    result = to_json_schema(decimal_field)
    assert result["type"] == "number"


# LLM-generated content at query #11
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test String field with null allowed
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=True)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == True
    assert result["exclusiveMaximum"] == True
    
    # Test Float field
    float_field = Float(minimum=0.0, maximum=1.0, multiple_of=0.1)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    assert result["multipleOf"] == 0.1
    
    # Test Boolean field
    bool_field = Boolean(allow_null=True)
    result = to_json_schema(bool_field)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(
        items=String(),
        min_items=1,
        max_items=10,
        unique_items=True,
        additional_items=False
    )
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    assert result["additionalItems"] == False
    assert "items" in result
    
    # Test Object field
    object_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=5,
        additional_properties=False
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    assert result["additionalProperties"] == False
    
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
    oneof_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(oneof_field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    allof_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(allof_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2
    
    # Test IfThenElse field
    ifthenelse_field = IfThenElse(
        if_clause=String(min_length=1),
        then_clause=Integer(minimum=0),
        else_clause=String(max_length=10)
    )
    result = to_json_schema(ifthenelse_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test Reference field with definitions
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    ref_field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/Person"
    
    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test with default value
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"
    
    # Test pattern properties in Object
    object_with_pattern = Object(
        pattern_properties={"^[a-z]+$": String()}
    )
    result = to_json_schema(object_with_pattern)
    assert "patternProperties" in result
    assert "^[a-z]+$" in result["patternProperties"]
    
    # Test property names in Object
    object_with_names = Object(
        property_names=String(pattern="^[a-z]+$")
    )
    result = to_json_schema(object_with_names)
    assert "propertyNames" in result
    
    # Test array with tuple items
    array_with_tuple = Array(
        items=[String(), Integer()],
        additional_items=String()
    )
    result = to_json_schema(array_with_tuple)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    assert "additionalItems" in result
    
    # Test error case for unsupported regex flags
    import re
    try:
        string_with_flags = String(pattern=re.compile("^[a-z]+$", re.IGNORECASE))
        to_json_schema(string_with_flags)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-standard flags" in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    result = to_json_schema(field)
    assert result is True

    # Test NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result is False

    # Test String field
    field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with null
    field = String(allow_null=True, min_length=0, allow_blank=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert "minLength" not in result

    # Test Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100, multiple_of=2)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100
    assert result["multipleOf"] == 2

    # Test Float field
    field = Float(allow_null=True, minimum=0.0, maximum=1.0)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test Boolean field
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

    # Test Array field with single item schema
    item_field = String()
    field = Array(allow_null=False, items=item_field, min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] is True
    assert result["items"]["type"] == "string"

    # Test Array field with tuple items
    field = Array(items=[String(), Integer()], additional_items=False)
    result = to_json_schema(field)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    assert result["additionalItems"] is False

    # Test Object field
    properties = {"name": String(), "age": Integer()}
    field = Object(allow_null=True, properties=properties, additional_properties=False, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == ["object", "null"]
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["additionalProperties"] is False
    assert result["required"] == ["name"]

    # Test Object field with pattern properties
    field = Object(pattern_properties={"^test_": String()})
    result = to_json_schema(field)
    assert "^test_" in result["patternProperties"]

    # Test Choice field
    field = Choice(choices=[("A", "Option A"), ("B", "Option B")])
    result = to_json_schema(field)
    assert result["enum"] == ["A", "B"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(field)
    assert len(result["allOf"]) == 2

    # Test IfThenElse field
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test Not field
    field = Not(negated=String())
    result = to_json_schema(field)
    assert "not" in result

    # Test Reference field
    definitions = Definitions({"User": Object(properties={"name": String()})})
    field = Reference(to="User", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/User"

    # Test with definitions
    user_schema = Object(properties={"name": String()})
    result = to_json_schema(user_schema)
    assert "components" not in result

    # Test Schema field (special Object type)
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]

    # Test default value handling
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result["default"] == "default_value"

    # Test Decimal field
    field = Decimal(allow_null=True, minimum=0, maximum=100)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]
    assert result["minimum"] == 0
    assert result["maximum"] == 100

    # Test error for unsupported field type
    class UnsupportedField(Field):
        pass

    field = UnsupportedField()
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


# LLM-generated content at query #13
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test String field with null allowed
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=False, exclusive_maximum=False)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    
    # Test Float field
    float_field = Float(minimum=0.0, maximum=1.0, multiple_of=0.1)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    assert result["multipleOf"] == 0.1
    
    # Test Boolean field
    bool_field = Boolean(allow_null=True)
    result = to_json_schema(bool_field)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=10, unique_items=True)
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    assert "items" in result
    
    # Test Array field with list items
    array_field_list = Array(items=[String(), Integer()])
    result = to_json_schema(array_field_list)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    
    # Test Object field
    object_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"],
        additional_properties=False
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["additionalProperties"] == False
    
    # Test Object field with pattern properties
    object_field_pattern = Object(
        pattern_properties={"^test_": String()}
    )
    result = to_json_schema(object_field_pattern)
    assert "^test_" in result["patternProperties"]
    
    # Test Choice field
    choice_field = Choice(choices=[("A", "Option A"), ("B", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["A", "B"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    oneof_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(oneof_field)
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    allof_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(allof_field)
    assert len(result["allOf"]) == 2
    
    # Test IfThenElse field
    ifthen_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(ifthen_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test Reference field
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    ref_field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/Person"
    
    # Test with definitions at root
    schema_field = Object(properties={"person": Reference(to="Person", definitions=definitions)})
    result = to_json_schema(schema_field)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "Person" in result["components"]["schemas"]
    
    # Test default value handling
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"
    
    # Test Schema field (special case)
    schema_field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]
    
    # Test Decimal field
    decimal_field = Decimal(minimum=0, maximum=100)
    result = to_json_schema(decimal_field)
    assert result["type"] == "number"
    assert result["minimum"] == 0
    assert result["maximum"] == 100


# LLM-generated content at query #14
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    any_field = Any()
    result = to_json_schema(any_field)
    assert result is True

    # Test NeverMatch field
    never_field = NeverMatch()
    result = to_json_schema(never_field)
    assert result is False

    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"

    # Test String field with null allowed
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]

    # Test Integer field
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=2)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 2

    # Test Float field
    float_field = Float(allow_null=True, exclusive_minimum=0, exclusive_maximum=1)
    result = to_json_schema(float_field)
    assert result["type"] == ["number", "null"]
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 1

    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    result = to_json_schema(bool_field)
    assert result["type"] == "boolean"

    # Test Array field
    array_field = Array(
        allow_null=False,
        min_items=1,
        max_items=10,
        items=String(),
        additional_items=False,
        unique_items=True
    )
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["additionalItems"] is False
    assert result["uniqueItems"] is True
    assert "items" in result

    # Test Object field
    object_field = Object(
        allow_null=True,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=2
    )
    result = to_json_schema(object_field)
    assert result["type"] == ["object", "null"]
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 2

    # Test Choice field
    choice_field = Choice(choices=[("A", "Option A"), ("B", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["A", "B"]

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
    allof_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(allof_field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test Reference field
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    ref_field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/Person"

    # Test IfThenElse field
    ifthen_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(ifthen_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result

    # Test with definitions
    person_def = Object(properties={"name": String()})
    result = to_json_schema(person_def, _definitions={"Person": person_def})
    assert "components" not in result

    # Test root with definitions
    result = to_json_schema(person_def)
    assert "components" in result
    assert "schemas" in result["components"]

    # Test default value handling
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"


# LLM-generated content at query #15
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    result = to_json_schema(field)
    assert result == True

    # Test NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result == False

    # Test String field
    field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with null allowed
    field = String(allow_null=True, allow_blank=False)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 1

    # Test Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=5)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["multipleOf"] == 5

    # Test Float field
    field = Float(allow_null=True, exclusive_minimum=0, exclusive_maximum=1)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 1

    # Test Boolean field
    field = Boolean(allow_null=False)
    result = to_json_schema(field)
    assert result["type"] == "boolean"

    # Test Array field
    field = Array(
        allow_null=False,
        min_items=1,
        max_items=10,
        items=String(),
        additional_items=False,
        unique_items=True
    )
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["additionalItems"] == False
    assert result["uniqueItems"] == True
    assert "items" in result

    # Test Object field
    field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        additional_properties=False,
        min_properties=1,
        max_properties=2
    )
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["additionalProperties"] == False
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 2

    # Test Choice field
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(field)
    assert result["enum"] == ["a", "b"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "anyOf" in result
    assert len(result["anyOf"]) == 2

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert "oneOf" in result
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(field)
    assert "allOf" in result
    assert len(result["allOf"]) == 2

    # Test IfThenElse field
    field = IfThenElse(
        if_clause=String(min_length=5),
        then_clause=String(max_length=10),
        else_clause=Integer()
    )
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test Not field
    field = Not(negated=String())
    result = to_json_schema(field)
    assert "not" in result

    # Test Reference field
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/Person"

    # Test with definitions
    field = Object(properties={"person": Reference(to="Person", definitions=definitions)})
    result = to_json_schema(field, _definitions={})
    assert "components" in result
    assert "schemas" in result["components"]
    assert "Person" in result["components"]["schemas"]

    # Test default value handling
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result["default"] == "default_value"

    # Test Schema field
    class PersonSchema(Schema):
        name = String()
        age = Integer()
    
    field = PersonSchema()
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]


# LLM-generated content at query #16
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    result = to_json_schema(field)
    assert result is True

    # Test NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result is False

    # Test String field
    field = String(allow_null=True, min_length=5, max_length=10, pattern="^test.*$", format="email")
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["pattern"] == "^test.*$"
    assert result["format"] == "email"

    # Test Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100, multiple_of=5)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100
    assert result["multipleOf"] == 5

    # Test Float field
    field = Float(allow_null=True, minimum=0.0, maximum=1.0)
    result = to_json_schema(field)
    assert result["type"] == ["number", "null"]
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0

    # Test Boolean field
    field = Boolean(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["boolean", "null"]

    # Test Array field
    field = Array(
        allow_null=False,
        min_items=1,
        max_items=10,
        items=String(),
        additional_items=False,
        unique_items=True
    )
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["items"]["type"] == "string"
    assert result["additionalItems"] is False
    assert result["uniqueItems"] is True

    # Test Object field
    field = Object(
        allow_null=True,
        properties={"name": String(), "age": Integer()},
        pattern_properties={"^test.*$": String()},
        additional_properties=False,
        property_names=String(pattern="^[a-z]+$"),
        min_properties=1,
        max_properties=5,
        required=["name"]
    )
    result = to_json_schema(field)
    assert result["type"] == ["object", "null"]
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert "^test.*$" in result["patternProperties"]
    assert result["additionalProperties"] is False
    assert result["propertyNames"]["pattern"] == "^[a-z]+$"
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    assert "name" in result["required"]

    # Test Schema field
    field = Schema(fields={"name": String(), "age": Integer()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert "name" in result["required"]

    # Test Choice field
    field = Choice(choices=[("A", "A"), ("B", "B")])
    result = to_json_schema(field)
    assert result["enum"] == ["A", "B"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    field = AllOf(all_of=[String(min_length=5), String(max_length=10)])
    result = to_json_schema(field)
    assert len(result["allOf"]) == 2

    # Test IfThenElse field
    field = IfThenElse(
        if_clause=String(pattern="^test.*$"),
        then_clause=String(min_length=10),
        else_clause=String(max_length=5)
    )
    result = to_json_schema(field)
    assert "if" in result
    assert "then" in result
    assert "else" in result

    # Test Not field
    field = Not(negated=String(pattern="^test.*$"))
    result = to_json_schema(field)
    assert "not" in result

    # Test Reference field
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/Person"

    # Test with definitions
    field = Object(properties={"person": Reference(to="Person", definitions=definitions)})
    result = to_json_schema(field)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "Person" in result["components"]["schemas"]

    # Test default value
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result["default"] == "default_value"

    # Test error for unsupported field type
    class UnsupportedField(Field):
        pass

    field = UnsupportedField()
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


# LLM-generated content at query #17
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test String field with null
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=False)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == True
    assert "exclusiveMaximum" not in result
    
    # Test Float field
    float_field = Float(multiple_of=0.5)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["multipleOf"] == 0.5
    
    # Test Boolean field
    bool_field = Boolean(allow_null=True)
    result = to_json_schema(bool_field)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(
        items=String(),
        min_items=1,
        max_items=10,
        unique_items=True
    )
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["uniqueItems"] == True
    assert "items" in result
    assert result["items"]["type"] == "string"
    
    # Test Array field with list items
    array_field_list = Array(
        items=[String(), Integer()],
        additional_items=False
    )
    result = to_json_schema(array_field_list)
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    assert result["additionalItems"] == False
    
    # Test Object field
    object_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"],
        additional_properties=False
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["additionalProperties"] == False
    
    # Test Object field with pattern properties
    object_field_pattern = Object(
        pattern_properties={"^test_": String()}
    )
    result = to_json_schema(object_field_pattern)
    assert "^test_" in result["patternProperties"]
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["a", "b"]
    
    # Test Const field
    const_field = Const(const=42)
    result = to_json_schema(const_field)
    assert result["const"] == 42
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"
    
    # Test OneOf field
    oneof_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(oneof_field)
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    allof_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(allof_field)
    assert len(result["allOf"]) == 2
    
    # Test IfThenElse field
    ifthen_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    result = to_json_schema(ifthen_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    assert result["not"]["type"] == "string"
    
    # Test Reference field
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    ref_field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/Person"
    
    # Test with definitions at root
    schema_field = Object(properties={"person": ref_field})
    result = to_json_schema(schema_field)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "Person" in result["components"]["schemas"]
    
    # Test default values
    string_with_default = String(default="hello")
    result = to_json_schema(string_with_default)
    assert result["default"] == "hello"
    
    # Test Schema field (subclass of Object)
    schema_field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]


# LLM-generated content at query #18
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    result = to_json_schema(field)
    assert result == True

    # Test NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result == False

    # Test String field
    field = String(allow_null=True, min_length=5, max_length=10, pattern="^test.*$", format="email")
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]
    assert result["minLength"] == 5
    assert result["maxLength"] == 10
    assert result["pattern"] == "^test.*$"
    assert result["format"] == "email"

    # Test Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100, multiple_of=5)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100
    assert result["multipleOf"] == 5

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
    field = Array(
        allow_null=True,
        min_items=1,
        max_items=10,
        items=String(),
        additional_items=False,
        unique_items=True
    )
    result = to_json_schema(field)
    assert result["type"] == ["array", "null"]
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["additionalItems"] == False
    assert result["uniqueItems"] == True
    assert result["items"]["type"] == "string"

    # Test Object field
    field = Object(
        allow_null=False,
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=5,
        additional_properties=False
    )
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 5
    assert result["additionalProperties"] == False

    # Test Choice field
    field = Choice(choices=[("A", "Option A"), ("B", "Option B")])
    result = to_json_schema(field)
    assert result["enum"] == ["A", "B"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(field)
    assert len(result["oneOf"]) == 2
    assert result["oneOf"][0]["type"] == "string"
    assert result["oneOf"][1]["type"] == "integer"

    # Test AllOf field
    field = AllOf(all_of=[String(min_length=5), String(max_length=10)])
    result = to_json_schema(field)
    assert len(result["allOf"]) == 2
    assert result["allOf"][0]["minLength"] == 5
    assert result["allOf"][1]["maxLength"] == 10

    # Test IfThenElse field
    field = IfThenElse(
        if_clause=String(min_length=5),
        then_clause=Integer(minimum=10),
        else_clause=Boolean()
    )
    result = to_json_schema(field)
    assert result["if"]["type"] == "string"
    assert result["if"]["minLength"] == 5
    assert result["then"]["type"] == "integer"
    assert result["then"]["minimum"] == 10
    assert result["else"]["type"] == "boolean"

    # Test Not field
    field = Not(negated=String())
    result = to_json_schema(field)
    assert result["not"]["type"] == "string"

    # Test Reference field
    definitions = Definitions()
    definitions["Person"] = Object(properties={"name": String()})
    field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/Person"

    # Test with definitions
    field = Reference(to="Person", definitions=definitions)
    result = to_json_schema(field, _definitions={})
    assert result["$ref"] == "#/components/schemas/Person"

    # Test Schema field
    field = Schema(fields={"name": String()}, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["required"] == ["name"]

    # Test default values
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result["default"] == "default_value"

    # Test error for unsupported field type
    class UnsupportedField(Field):
        pass

    field = UnsupportedField()
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)

    # Test pattern regex with non-unicode flags error
    import re
    field = String(pattern_regex=re.compile("^test.*$", re.IGNORECASE))
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-standard flags" in str(e)


# LLM-generated content at query #19
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    field = Any()
    result = to_json_schema(field)
    assert result == True

    # Test NeverMatch field
    field = NeverMatch()
    result = to_json_schema(field)
    assert result == False

    # Test String field
    field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$", format="email")
    result = to_json_schema(field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    assert result["format"] == "email"

    # Test String field with null
    field = String(allow_null=True)
    result = to_json_schema(field)
    assert result["type"] == ["string", "null"]

    # Test Integer field
    field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=0, exclusive_maximum=100, multiple_of=2)
    result = to_json_schema(field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == 0
    assert result["exclusiveMaximum"] == 100
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
    item_field = String()
    field = Array(allow_null=False, min_items=1, max_items=10, items=item_field, additional_items=False, unique_items=True)
    result = to_json_schema(field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 10
    assert result["items"]["type"] == "string"
    assert result["additionalItems"] == False
    assert result["uniqueItems"] == True

    # Test Array field with list items
    item_fields = [String(), Integer()]
    field = Array(items=item_fields)
    result = to_json_schema(field)
    assert len(result["items"]) == 2
    assert result["items"][0]["type"] == "string"
    assert result["items"][1]["type"] == "integer"

    # Test Object field
    properties = {"name": String(), "age": Integer()}
    field = Object(allow_null=False, properties=properties, additional_properties=False, required=["name"], min_properties=1, max_properties=2)
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["additionalProperties"] == False
    assert result["required"] == ["name"]
    assert result["minProperties"] == 1
    assert result["maxProperties"] == 2

    # Test Object field with pattern properties
    pattern_properties = {"^test_": String()}
    field = Object(pattern_properties=pattern_properties)
    result = to_json_schema(field)
    assert "^test_" in result["patternProperties"]
    assert result["patternProperties"]["^test_"]["type"] == "string"

    # Test Choice field
    field = Choice(choices=[("A", "A"), ("B", "B")])
    result = to_json_schema(field)
    assert result["enum"] == ["A", "B"]

    # Test Const field
    field = Const(const="fixed_value")
    result = to_json_schema(field)
    assert result["const"] == "fixed_value"

    # Test Union field
    fields = [String(), Integer()]
    field = Union(any_of=fields)
    result = to_json_schema(field)
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["type"] == "string"
    assert result["anyOf"][1]["type"] == "integer"

    # Test OneOf field
    field = OneOf(one_of=fields)
    result = to_json_schema(field)
    assert len(result["oneOf"]) == 2

    # Test AllOf field
    field = AllOf(all_of=fields)
    result = to_json_schema(field)
    assert len(result["allOf"]) == 2

    # Test IfThenElse field
    if_field = Boolean()
    then_field = String()
    else_field = Integer()
    field = IfThenElse(if_clause=if_field, then_clause=then_field, else_clause=else_field)
    result = to_json_schema(field)
    assert result["if"]["type"] == "boolean"
    assert result["then"]["type"] == "string"
    assert result["else"]["type"] == "integer"

    # Test Not field
    negated_field = String()
    field = Not(negated=negated_field)
    result = to_json_schema(field)
    assert result["not"]["type"] == "string"

    # Test Reference field
    target_field = String()
    definitions = Definitions({"TestSchema": target_field})
    field = Reference(to="TestSchema", definitions=definitions)
    result = to_json_schema(field)
    assert result["$ref"] == "#/components/schemas/TestSchema"

    # Test with definitions at root level
    field = Reference(to="TestSchema", definitions=definitions)
    result = to_json_schema(field, _definitions={})
    assert "components" in result
    assert "schemas" in result["components"]
    assert "TestSchema" in result["components"]["schemas"]

    # Test Schema field (treated as Object)
    fields = {"name": String(), "age": Integer()}
    field = Schema(fields=fields, required=["name"])
    result = to_json_schema(field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]

    # Test default value handling
    field = String(default="default_value")
    result = to_json_schema(field)
    assert result["default"] == "default_value"

    # Test Decimal field (treated as number)
    field = Decimal(allow_null=False, minimum=0, maximum=10)
    result = to_json_schema(field)
    assert result["type"] == "number"
    assert result["minimum"] == 0
    assert result["maximum"] == 10

    # Test error for unsupported field type
    class UnsupportedField(Field):
        pass

    field = UnsupportedField()
    try:
        to_json_schema(field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


# LLM-generated content at query #20
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True
    
    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False
    
    # Test String field
    string_field = String(allow_null=False, min_length=1, max_length=10, pattern="^[a-z]+$")
    result = to_json_schema(string_field)
    assert result["type"] == "string"
    assert result["minLength"] == 1
    assert result["maxLength"] == 10
    assert result["pattern"] == "^[a-z]+$"
    
    # Test String field with null allowed
    string_field_null = String(allow_null=True)
    result = to_json_schema(string_field_null)
    assert result["type"] == ["string", "null"]
    
    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=False)
    result = to_json_schema(int_field)
    assert result["type"] == "integer"
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["exclusiveMinimum"] == True
    assert "exclusiveMaximum" not in result
    
    # Test Float field
    float_field = Float(multiple_of=0.5)
    result = to_json_schema(float_field)
    assert result["type"] == "number"
    assert result["multipleOf"] == 0.5
    
    # Test Boolean field
    bool_field = Boolean(allow_null=True)
    result = to_json_schema(bool_field)
    assert result["type"] == ["boolean", "null"]
    
    # Test Array field
    array_field = Array(
        items=String(),
        min_items=1,
        max_items=5,
        unique_items=True,
        additional_items=False
    )
    result = to_json_schema(array_field)
    assert result["type"] == "array"
    assert result["minItems"] == 1
    assert result["maxItems"] == 5
    assert result["uniqueItems"] == True
    assert result["additionalItems"] == False
    assert "items" in result
    
    # Test Object field
    object_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"],
        additional_properties=False
    )
    result = to_json_schema(object_field)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["required"] == ["name"]
    assert result["additionalProperties"] == False
    
    # Test Schema field
    schema_field = Schema(fields={"id": Integer()}, required=["id"])
    result = to_json_schema(schema_field)
    assert result["type"] == "object"
    assert "id" in result["properties"]
    assert result["required"] == ["id"]
    
    # Test Choice field
    choice_field = Choice(choices=[("A", "Option A"), ("B", "Option B")])
    result = to_json_schema(choice_field)
    assert result["enum"] == ["A", "B"]
    
    # Test Const field
    const_field = Const(const="fixed_value")
    result = to_json_schema(const_field)
    assert result["const"] == "fixed_value"
    
    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    result = to_json_schema(union_field)
    assert len(result["anyOf"]) == 2
    
    # Test OneOf field
    oneof_field = OneOf(one_of=[String(), Integer()])
    result = to_json_schema(oneof_field)
    assert len(result["oneOf"]) == 2
    
    # Test AllOf field
    allof_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    result = to_json_schema(allof_field)
    assert len(result["allOf"]) == 2
    
    # Test IfThenElse field
    ifthen_field = IfThenElse(
        if_clause=String(min_length=5),
        then_clause=String(max_length=20),
        else_clause=Integer()
    )
    result = to_json_schema(ifthen_field)
    assert "if" in result
    assert "then" in result
    assert "else" in result
    
    # Test Not field
    not_field = Not(negated=String())
    result = to_json_schema(not_field)
    assert "not" in result
    
    # Test Reference field with definitions
    definitions = Definitions({"User": Object(properties={"name": String()})})
    ref_field = Reference(to="User", definitions=definitions)
    result = to_json_schema(ref_field)
    assert result["$ref"] == "#/components/schemas/User"
    
    # Test with definitions at root level
    user_field = Object(properties={"name": String()})
    result = to_json_schema(user_field, _definitions={"User": user_field})
    assert "components" not in result
    
    # Test root level with definitions
    result = to_json_schema(definitions)
    assert "components" in result
    assert "schemas" in result["components"]
    assert "User" in result["components"]["schemas"]
    
    # Test default values
    string_with_default = String(default="default_value")
    result = to_json_schema(string_with_default)
    assert result["default"] == "default_value"
    
    # Test pattern regex with unicode flags (should work)
    string_with_pattern = String(pattern_regex=re.compile("^[a-z]+$", re.UNICODE))
    result = to_json_schema(string_with_pattern)
    assert result["pattern"] == "^[a-z]+$"
    
    # Test pattern regex with non-unicode flags (should raise error)
    string_with_bad_pattern = String(pattern_regex=re.compile("^[a-z]+$", re.IGNORECASE))
    try:
        to_json_schema(string_with_bad_pattern)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-standard flags" in str(e)
    
    # Test unknown field type
    class UnknownField(Field):
        pass
    
    unknown_field = UnknownField()
    try:
        to_json_schema(unknown_field)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert field type" in str(e)


