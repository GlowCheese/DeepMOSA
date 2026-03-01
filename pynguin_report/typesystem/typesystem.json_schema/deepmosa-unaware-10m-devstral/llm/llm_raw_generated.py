####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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
    const_field = from_json_schema({"const": "value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "value"

    # Test allOf
    all_of_field = from_json_schema({
        "allOf": [
            {"type": "string"},
            {"minLength": 5}
        ]
    })
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.constraints) == 2

    # Test anyOf
    any_of_field = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.options) == 2

    # Test oneOf
    one_of_field = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.options) == 2

    # Test not
    not_field = from_json_schema({
        "not": {"type": "string"}
    })
    assert isinstance(not_field, Not)
    assert isinstance(not_field.negated, String)

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
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "string"})
    ref_field = from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=definitions)
    assert isinstance(ref_field, Reference)
    assert ref_field.ref == "#/components/schemas/Test"

    # Test combined constraints
    combined_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10
    })
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.constraints) == 3

    # Test default case (no constraints)
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #4
#--------------------------

```python
def test_ref_from_json_schema():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/TestSchema"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/TestSchema"
    assert result.definitions == definitions


# LLM-generated content at query #5
#--------------------------

```python
def test_all_of_from_json_schema():
    definitions = Definitions()

    # Test with simple allOf
    data = {
        "allOf": [
            {"type": "string", "minLength": 2},
            {"type": "string", "maxLength": 10}
        ]
    }
    field = all_of_from_json_schema(data, definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2
    assert isinstance(field.all_of[0], String)
    assert isinstance(field.all_of[1], String)

    # Test with nested allOf
    data = {
        "allOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "object", "properties": {"age": {"type": "integer"}}}
        ]
    }
    field = all_of_from_json_schema(data, definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2
    assert isinstance(field.all_of[0], Object)
    assert isinstance(field.all_of[1], Object)

    # Test with default value
    data = {
        "allOf": [{"type": "string"}],
        "default": "test"
    }
    field = all_of_from_json_schema(data, definitions)
    assert field.default == "test"

    # Test with reference in allOf
    definitions["#/components/schemas/Test"] = String()
    data = {
        "allOf": [
            {"$ref": "#/components/schemas/Test"},
            {"type": "string", "minLength": 1}
        ]
    }
    field = all_of_from_json_schema(data, definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2
    assert isinstance(field.all_of[0], Reference)
    assert isinstance(field.all_of[1], String)


# LLM-generated content at query #6
#--------------------------

```python
def test_if_then_else_from_json_schema():
    definitions = Definitions()

    # Test with if, then, and else clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "number"},
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, String)
    assert isinstance(field.else_clause, Number)

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, String)
    assert field.else_clause is None

    # Test with only if clause
    data = {
        "if": {"type": "string"},
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
        "default": "default_value",
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert field.default == "default_value"


# LLM-generated content at query #7
#--------------------------

```python
def test_to_json_schema():
    # Test Any
    assert to_json_schema(Any()) == True

    # Test NeverMatch
    assert to_json_schema(NeverMatch()) == False

    # Test String
    field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(field) == expected

    # Test Integer
    field = Integer(minimum=0, maximum=100, exclusive_maximum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMaximum": True
    }
    assert to_json_schema(field) == expected

    # Test Float
    field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(field) == expected

    # Test Boolean
    field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(field) == expected

    # Test Array
    field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(field) == expected

    # Test Object
    field = Object(properties={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(field) == expected

    # Test Choice
    field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(field) == expected

    # Test Const
    field = Const(const="fixed")
    expected = {
        "const": "fixed"
    }
    assert to_json_schema(field) == expected

    # Test Union
    field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(field) == expected

    # Test AllOf
    field = AllOf(all_of=[String(), Const(const="test")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "test"}]
    }
    assert to_json_schema(field) == expected

    # Test OneOf
    field = OneOf(one_of=[String(), Integer()])
    expected = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(field) == expected

    # Test Not
    field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(field) == expected

    # Test IfThenElse
    field = IfThenElse(if_clause=String(), then_clause=Integer())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    assert to_json_schema(field) == expected

    # Test Reference
    definitions = Definitions({"Test": String()})
    field = Reference(to="Test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(field) == expected

    # Test Schema
    field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(field) == expected


# LLM-generated content at query #8
#--------------------------

```python
def test_ref_from_json_schema():
    definitions = Definitions()
    definitions["#/test"] = String()

    data = {"$ref": "#/test"}
    field = ref_from_json_schema(data, definitions)

    assert isinstance(field, Reference)
    assert field.to == "#/test"
    assert field.definitions is definitions


# LLM-generated content at query #9
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type
    data = {"type": "string"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)

    # Test multiple types (union)
    data = {"type": ["string", "integer"]}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Integer)

    # Test nullable type
    data = {"type": "string", "nullable": True}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)
    assert result.allow_null is True

    # Test nullable union
    data = {"type": ["string", "integer"], "nullable": True}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test no type with nullable
    data = {"nullable": True}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test no type without nullable
    data = {}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, NeverMatch)

    # Test with constraints
    data = {"type": "string", "minLength": 5}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5

    # Test array type
    data = {"type": "array", "items": {"type": "string"}}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

    # Test object type
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert isinstance(result.properties["name"], String)


# LLM-generated content at query #10
#--------------------------

```python
def test_ref_from_json_schema():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/Test"}
    field = ref_from_json_schema(data, definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/Test"
    assert field.definitions is definitions


# LLM-generated content at query #11
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

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #12
#--------------------------

```python
def test_ref_from_json_schema():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/Test"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/Test"
    assert result.definitions == definitions


# LLM-generated content at query #13
#--------------------------

```python
def test_ref_from_json_schema():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/TestSchema"}
    definitions["#/components/schemas/TestSchema"] = String()

    field = ref_from_json_schema(data, definitions=definitions)

    assert isinstance(field, Reference)
    assert field.to == "#/components/schemas/TestSchema"
    assert field.definitions == definitions


# LLM-generated content at query #14
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
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items is False
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
        "patternProperties": {
            "^S_": {"type": "string"},
            "^I_": {"type": "integer"}
        },
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test"}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert isinstance(field.pattern_properties["^S_"], String)
    assert isinstance(field.pattern_properties["^I_"], Integer)
    assert field.additional_properties is False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_to_json_schema():
    # Test Any
    assert to_json_schema(Any()) == True

    # Test NeverMatch
    assert to_json_schema(NeverMatch()) == False

    # Test String
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer
    int_field = Integer(minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected

    # Test Float
    float_field = Float(minimum=0.0, maximum=1.0, multiple_of=0.1)
    expected = {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "multipleOf": 0.1
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean
    bool_field = Boolean()
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array
    array_field = Array(items=String(), min_items=1, max_items=5)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5
    }
    assert to_json_schema(array_field) == expected

    # Test Object
    object_field = Object(properties={"name": String(), "age": Integer()})
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
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

    # Test Not
    not_field = Not(negated=String())
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected

    # Test IfThenElse
    if_field = IfThenElse(if_clause=String(), then_clause=Integer())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    assert to_json_schema(if_field) == expected

    # Test Reference
    definitions = Definitions()
    ref_field = Reference(to="test", definitions=definitions)
    expected = {"$ref": "#/components/schemas/test"}
    assert to_json_schema(ref_field) == expected

    # Test Schema
    schema_field = Schema(fields={"name": String()})
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}}
    }
    assert to_json_schema(schema_field) == expected


# LLM-generated content at query #17
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
    obj_field = Object(properties={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
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
    ref_field = Reference(to="Test", target=String())
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

    # Test Definitions
    definitions = Definitions({"Test": String()})
    expected = {
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #18
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
    int_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=True, multiple_of=2)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "multipleOf": 2
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(allow_null=True, minimum=0.0, maximum=1.0)
    expected = {
        "type": ["number", "null"],
        "minimum": 0.0,
        "maximum": 1.0
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(allow_null=True, min_items=1, max_items=5, items=String(), additional_items=False, unique_items=True)
    expected = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"},
        "additionalItems": False,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String()},
        pattern_properties={"^S_": String()},
        additional_properties=False,
        property_names=String(pattern="[A-Z]+"),
        min_properties=1,
        max_properties=5,
        required=["name"]
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "patternProperties": {"^S_": {"type": "string"}},
        "additionalProperties": False,
        "propertyNames": {"type": "string", "pattern": "[A-Z]+"},
        "minProperties": 1,
        "maxProperties": 5,
        "required": ["name"]
    }
    assert to_json_schema(object_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")], default="a")
    expected = {"enum": ["a", "b"], "default": "a"}
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="fixed", default="fixed")
    expected = {"const": "fixed", "default": "fixed"}
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
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
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

    # Test Definitions
    definitions = Definitions({
        "StringField": String(),
        "IntField": Integer()
    })
    expected = {
        "components": {
            "schemas": {
                "StringField": {"type": "string"},
                "IntField": {"type": "integer"}
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #19
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
        "pattern": "^[a-zA-Z0-9_]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9_]+$"
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
        "default": 50,
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
        "default": 50,
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
        "default": "test",
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

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 10,
        "default": {"name": "test"},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.default == {"name": "test"}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #21
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
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test empty type with allow_null
    data = {}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, NeverMatch)

    # Test empty type with allow_null=True
    data = {"type": ["null"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test complex type with constraints
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[A-Za-z]+$"
    }
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.pattern == "^[A-Za-z]+$"

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        },
        "required": ["name"]
    }
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert "age" in result.properties
    assert "name" in result.required

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5
    }
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    assert result.min_items == 1
    assert result.max_items == 5


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
        "maxLength": 10,
        "pattern": "^[A-Za-z]+$",
        "default": "hello",
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
        "maxItems": 5,
        "uniqueItems": True,
        "default": ["a", "b"],
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items is True
    assert field.default == ["a", "b"]

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
def test_type_from_json_schema():
    # Test single type string
    data = {"type": "string"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)

    # Test multiple type strings
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

    # Test allow_null with single type
    data = {"type": "string", "nullable": True}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert any(isinstance(field, String) for field in result.any_of)
    assert any(isinstance(field, Const) and field.value is None for field in result.any_of)

    # Test allow_null with no type
    data = {"nullable": True}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test allow_null with no type and nullable=False
    data = {"nullable": False}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, NeverMatch)

    # Test array type
    data = {"type": "array"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Array)

    # Test object type
    data = {"type": "object"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Object)

    # Test number type
    data = {"type": "number"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Number)

    # Test integer type
    data = {"type": "integer"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Integer)

    # Test boolean type
    data = {"type": "boolean"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Boolean)


# LLM-generated content at query #24
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type
    data = {"type": "string"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)

    # Test multiple types (Union)
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert any(isinstance(field, String) for field in result.any_of)
    assert any(isinstance(field, Number) for field in result.any_of)

    # Test nullable type
    data = {"type": "string", "nullable": True}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)
    assert result.allow_null is True

    # Test no type with nullable
    data = {"nullable": True}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test no type without nullable
    data = {}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, NeverMatch)

    # Test with constraints
    data = {"type": "string", "minLength": 5}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5


# LLM-generated content at query #25
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
        "minLength": 1,
        "maxLength": 100,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 1
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
        "properties": {"name": {"type": "string"}},
        "patternProperties": {"^S_": {"type": "string"}},
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test"}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.pattern_properties["^S_"], String)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}

    # Test with allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #26
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
    assert isinstance(combined_field, String)
    assert combined_field.min_length == 5
    assert combined_field.max_length == 10

    # Test multiple constraints
    multi_field = from_json_schema({
        "type": "string",
        "enum": ["a", "b"],
        "minLength": 1
    })
    assert isinstance(multi_field, AllOf)
    assert len(multi_field.schemas) == 2


# LLM-generated content at query #27
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
    data = {"type": "array", "items": {"type": "string"}}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)

    # Test object type
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
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


# LLM-generated content at query #28
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, allow_null=True)
    expected = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, allow_null=False)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5, allow_null=True)
    expected = {
        "type": ["number", "null"],
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean(allow_null=True)
    expected = {
        "type": ["boolean", "null"]
    }
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, allow_null=False)
    expected = {
        "type": "array",
        "minItems": 1,
        "items": {"type": "string"}
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
        properties={"name": String()},
        required=["name"],
        allow_null=True
    )
    expected = {
        "type": ["object", "null"],
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

    # Test Schema field
    schema_field = Schema(
        fields={"name": String()},
        required=["name"],
        allow_null=True
    )
    expected = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected


# LLM-generated content at query #29
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
    result = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.default == 50.5

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 50
    }
    result = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[A-Za-z]+$",
        "default": "hello"
    }
    result = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.pattern == "^[A-Za-z]+$"
    assert result.default == "hello"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    result = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(result, Boolean)
    assert result.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "default": ["a", "b"]
    }
    result = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    assert result.min_items == 1
    assert result.max_items == 5
    assert result.unique_items == True
    assert result.default == ["a", "b"]

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
    result = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(result, Object)
    assert isinstance(result.properties["name"], String)
    assert isinstance(result.properties["age"], Integer)
    assert result.required == ["name"]
    assert result.default == {"name": "John", "age": 30}


# LLM-generated content at query #30
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

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #31
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    any_field = Any()
    assert to_json_schema(any_field) == True

    # Test NeverMatch field
    never_match = NeverMatch()
    assert to_json_schema(never_match) == False

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


# LLM-generated content at query #32
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
    const_schema = {"const": "test"}
    assert isinstance(from_json_schema(const_schema), Const)
    assert from_json_schema(const_schema).value == "test"

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

    # Test combined constraints
    combined_schema = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    }
    result = from_json_schema(combined_schema)
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.pattern == "^[a-z]+$"

    # Test no constraints (should return Any)
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #33
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
        "exclusiveMinimum": True,
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
        max_properties=5
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
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
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected


# LLM-generated content at query #34
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
    const_field = from_json_schema({"const": "value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "value"

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
    definitions["#/components/schemas/Test"] = String()
    ref_field = from_json_schema(
        {"$ref": "#/components/schemas/Test"},
        definitions=definitions
    )
    assert isinstance(ref_field, Reference)
    assert ref_field.ref == "#/components/schemas/Test"

    # Test multiple constraints
    multi_constraint_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    })
    assert isinstance(multi_constraint_field, AllOf)
    assert len(multi_constraint_field.schemas) == 4

    # Test no constraints
    any_field = from_json_schema({})
    assert isinstance(any_field, Any)


# LLM-generated content at query #35
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
    assert isinstance(from_json_schema({"type": ["string", "integer"]}), Union)

    # Test enum constraint
    assert isinstance(from_json_schema({"enum": ["a", "b", "c"]}), Choice)

    # Test const constraint
    assert isinstance(from_json_schema({"const": "value"}), Const)

    # Test allOf constraint
    assert isinstance(from_json_schema({"allOf": [{"type": "string"}, {"minLength": 1}]}), AllOf)

    # Test anyOf constraint
    assert isinstance(from_json_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]}), Union)

    # Test oneOf constraint
    assert isinstance(from_json_schema({"oneOf": [{"type": "string"}, {"type": "integer"}]}), OneOf)

    # Test not constraint
    assert isinstance(from_json_schema({"not": {"type": "string"}}), Not)

    # Test if-then-else constraint
    assert isinstance(from_json_schema({"if": {"type": "string"}, "then": {"minLength": 1}, "else": {"minLength": 0}}), IfThenElse)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "string"})
    assert isinstance(from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=definitions), Reference)

    # Test combined constraints
    assert isinstance(from_json_schema({"type": "string", "enum": ["a", "b", "c"]}), AllOf)

    # Test no constraints
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #36
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
    field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=True, multiple_of=2)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "multipleOf": 2
    }
    assert to_json_schema(field) == expected

    # Test Float
    field = Float(allow_null=True, minimum=0.0, maximum=1.0)
    expected = {
        "type": ["number", "null"],
        "minimum": 0.0,
        "maximum": 1.0
    }
    assert to_json_schema(field) == expected

    # Test Boolean
    field = Boolean(allow_null=False)
    expected = {"type": "boolean"}
    assert to_json_schema(field) == expected

    # Test Array
    field = Array(allow_null=True, items=String(), min_items=1, max_items=10, unique_items=True)
    expected = {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True
    }
    assert to_json_schema(field) == expected

    # Test Object
    field = Object(
        allow_null=False,
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
    assert to_json_schema(field) == expected

    # Test Choice
    field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {"enum": ["a", "b"]}
    assert to_json_schema(field) == expected

    # Test Const
    field = Const(const="fixed_value")
    expected = {"const": "fixed_value"}
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
    definitions = Definitions({"Test": String()})
    field = Reference(to="Test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {"schemas": {"Test": {"type": "string"}}}
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


# LLM-generated content at query #37
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
    object_field = Object(
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
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(all_of_field) == expected

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

    # Test null handling
    string_field_null = String(allow_null=True)
    expected = {"type": ["string", "null"]}
    assert to_json_schema(string_field_null) == expected


# LLM-generated content at query #38
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
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=2)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 2
    }
    assert to_json_schema(int_field) == expected

    # Test Boolean field
    bool_field = Boolean(allow_null=True)
    expected = {"type": ["boolean", "null"]}
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(allow_null=False, items=String(), min_items=1, max_items=5)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    obj_field = Object(
        allow_null=True,
        properties={"name": String()},
        required=["name"]
    )
    expected = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(obj_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {"enum": ["a", "b"]}
    assert to_json_schema(choice_field) == expected

    # Test Const field
    const_field = Const(const="value")
    expected = {"const": "value"}
    assert to_json_schema(const_field) == expected

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(union_field) == expected

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(all_of_field) == expected

    # Test Reference field
    definitions = Definitions()
    ref_field = Reference(to="test", definitions=definitions)
    expected = {"$ref": "#/components/schemas/test"}
    assert to_json_schema(ref_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()})
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}}
    }
    assert to_json_schema(schema_field) == expected


# LLM-generated content at query #39
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
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=2)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 2
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(allow_null=True, exclusive_minimum=0, exclusive_maximum=1)
    expected = {
        "type": ["number", "null"],
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 1
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(allow_null=True, min_items=1, max_items=5, items=String(), additional_items=False)
    expected = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"},
        "additionalItems": False,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
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
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(all_of_field) == expected

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(one_of_field) == expected

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
    ref_field = Reference(to="test", definitions={"test": String()})
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


# LLM-generated content at query #40
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
        "pattern": "^[A-Za-z]+$",
        "default": "hello",
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
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["hello"],
    }
    field = from_json_schema_type(data, "array", False, Definitions())
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
            "age": {"type": "integer"},
        },
        "required": ["name"],
        "default": {"name": "John", "age": 30},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}


# LLM-generated content at query #41
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
        "additionalProperties": False,
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.additional_properties == False
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #42
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
    field = from_json_schema({"enum": ["a", "b", "c"]})
    assert isinstance(field, Choice)
    assert field.choices == ["a", "b", "c"]

    # Test const constraint
    field = from_json_schema({"const": "test"})
    assert isinstance(field, Const)
    assert field.value == "test"

    # Test allOf constraint
    field = from_json_schema({"allOf": [{"type": "string"}, {"minLength": 5}]})
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2

    # Test anyOf constraint
    field = from_json_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2

    # Test oneOf constraint
    field = from_json_schema({"oneOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2

    # Test not constraint
    field = from_json_schema({"not": {"type": "string"}})
    assert isinstance(field, Not)
    assert isinstance(field.not_, String)

    # Test if/then/else constraint
    field = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    })
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_, String)
    assert isinstance(field.then, String)
    assert isinstance(field.else_, Integer)

    # Test $ref constraint
    definitions = Definitions()
    definitions["#/components/schemas/test"] = {"type": "string"}
    field = from_json_schema({"$ref": "#/components/schemas/test"}, definitions=definitions)
    assert isinstance(field, Reference)

    # Test multiple constraints
    field = from_json_schema({"type": "string", "minLength": 5, "maxLength": 10})
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2

    # Test no constraints
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #43
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


# LLM-generated content at query #44
#--------------------------

```python
def test_from_json_schema_type():
    # Test number type
    data = {"type": "number", "minimum": 0, "maximum": 10}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10

    # Test integer type
    data = {"type": "integer", "minimum": 0, "maximum": 10}
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10

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
    data = {"type": "array", "items": {"type": "string"}}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)

    # Test object type
    data = {"type": "object", "properties": {"name": {"type": "string"}}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)


# LLM-generated content at query #45
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
        "properties": {"name": {"type": "string"}},
        "patternProperties": {"^S_": {"type": "string"}},
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test"}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.pattern_properties["^S_"], String)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}


# LLM-generated content at query #46
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
        "minProperties": 1,
        "maxProperties": 2,
        "required": ["name"],
        "default": {"name": "test", "age": 25}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 25}


# LLM-generated content at query #47
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
        "additionalItems": False,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.additional_items is False
    assert field.unique_items is True
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
    assert field.additional_properties is False
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert isinstance(field, String)
    assert field.allow_null is True


# LLM-generated content at query #48
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

    # Test Reference field
    definitions = Definitions()
    ref_field = Reference(to="Test", definitions=definitions, target=String())
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(ref_field) == expected


# LLM-generated content at query #49
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
    assert isinstance(from_json_schema({"type": ["string", "integer"]}), Union)

    # Test enum constraint
    result = from_json_schema({"enum": ["a", "b", "c"]})
    assert isinstance(result, Choice)
    assert result.choices == ["a", "b", "c"]

    # Test const constraint
    result = from_json_schema({"const": "value"})
    assert isinstance(result, Const)
    assert result.value == "value"

    # Test allOf constraint
    result = from_json_schema({
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    })
    assert isinstance(result, AllOf)
    assert len(result.schemas) == 2

    # Test anyOf constraint
    result = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(result, Union)
    assert len(result.schemas) == 2

    # Test oneOf constraint
    result = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(result, OneOf)
    assert len(result.schemas) == 2

    # Test not constraint
    result = from_json_schema({
        "not": {"type": "string"}
    })
    assert isinstance(result, Not)
    assert isinstance(result.schema, String)

    # Test if-then-else constraint
    result = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
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
    assert result.ref == "#/components/schemas/Test"

    # Test multiple constraints
    result = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-z]+$"
    })
    assert isinstance(result, AllOf)
    assert len(result.schemas) == 4

    # Test no constraints
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #50
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
    const_schema = {"const": "value"}
    const_field = from_json_schema(const_schema)
    assert isinstance(const_field, Const)
    assert const_field.value == "value"

    # Test allOf
    all_of_schema = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    all_of_field = from_json_schema(all_of_schema)
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.schemas) == 2

    # Test anyOf
    any_of_schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    any_of_field = from_json_schema(any_of_schema)
    assert isinstance(any_of_field, Union)
    assert len(any_of_field.schemas) == 2

    # Test oneOf
    one_of_schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    one_of_field = from_json_schema(one_of_schema)
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.schemas) == 2

    # Test not
    not_schema = {"not": {"type": "string"}}
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
    assert isinstance(combined_field, String)
    assert combined_field.min_length == 5
    assert combined_field.max_length == 10
    assert combined_field.pattern == "^[a-z]+$"

    # Test no constraints
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #51
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
    object_field = Object(
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
    assert to_json_schema(object_field) == expected

    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
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
    definitions = Definitions()
    ref_field = Reference(to="TestRef", definitions=definitions)
    expected = {"$ref": "#/components/schemas/TestRef"}
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
    definitions = Definitions()
    definitions["Test"] = String()
    result = to_json_schema(definitions)
    assert result == {
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }


# LLM-generated content at query #52
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


# LLM-generated content at query #53
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected_string = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected_string

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True)
    expected_int = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True
    }
    assert to_json_schema(int_field) == expected_int

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected_float = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected_float

    # Test Boolean field
    bool_field = Boolean()
    expected_bool = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected_bool

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected_array = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected_array

    # Test Object field
    object_field = Object(
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=5
    )
    expected_object = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 5
    }
    assert to_json_schema(object_field) == expected_object

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected_choice = {"enum": ["a", "b"]}
    assert to_json_schema(choice_field) == expected_choice

    # Test Const field
    const_field = Const(const="fixed_value")
    expected_const = {"const": "fixed_value"}
    assert to_json_schema(const_field) == expected_const

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected_union = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(union_field) == expected_union

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected_all_of = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(all_of_field) == expected_all_of

    # Test Reference field
    definitions = Definitions({"Test": String()})
    ref_field = Reference(to="Test", definitions=definitions)
    expected_ref = {
        "$ref": "#/components/schemas/Test",
        "components": {"schemas": {"Test": {"type": "string"}}}
    }
    assert to_json_schema(ref_field) == expected_ref

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected_schema

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected_if = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected_if

    # Test Not field
    not_field = Not(negated=String())
    expected_not = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected_not

    # Test with definitions
    definitions = Definitions({
        "StringField": String(),
        "IntField": Integer()
    })
    expected_definitions = {
        "components": {
            "schemas": {
                "StringField": {"type": "string"},
                "IntField": {"type": "integer"}
            }
        }
    }
    assert to_json_schema(definitions) == expected_definitions


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_any_of_from_json_schema():
    # Test with simple anyOf schema
    schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    result = any_of_from_json_schema(schema, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Number)

    # Test with nested anyOf schema
    schema = {"anyOf": [{"type": "string"}, {"anyOf": [{"type": "number"}, {"type": "integer"}]}]}
    result = any_of_from_json_schema(schema, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Union)
    assert len(result.any_of[1].any_of) == 2
    assert isinstance(result.any_of[1].any_of[0], Number)
    assert isinstance(result.any_of[1].any_of[1], Integer)

    # Test with default value
    schema = {"anyOf": [{"type": "string"}, {"type": "number"}], "default": "test"}
    result = any_of_from_json_schema(schema, definitions=Definitions())
    assert isinstance(result, Union)
    assert result.default == "test"

    # Test with empty anyOf
    schema = {"anyOf": []}
    result = any_of_from_json_schema(schema, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 0


# LLM-generated content at query #2
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
    data_with_default = {"enum": [1, 2, 3], "default": 2}
    field_with_default = enum_from_json_schema(data_with_default, definitions=Definitions())
    assert isinstance(field_with_default, Choice)
    assert field_with_default.choices == [(1, 1), (2, 2), (3, 3)]
    assert field_with_default.default == 2

    # Test with mixed types in enum
    data_mixed = {"enum": [True, "text", 123]}
    field_mixed = enum_from_json_schema(data_mixed, definitions=Definitions())
    assert isinstance(field_mixed, Choice)
    assert field_mixed.choices == [(True, True), ("text", "text"), (123, 123)]


# LLM-generated content at query #3
#--------------------------

```python
def test_type_from_json_schema():
    # Test with single type
    data = {"type": "string"}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, String)

    # Test with multiple types (union)
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Number)

    # Test with nullable type
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test with empty type (allow_null)
    data = {}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, NeverMatch)

    # Test with empty type and allow_null
    data = {"type": ["null"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test with complex type constraints
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[A-Za-z]+$"
    }
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10
    assert result.pattern == "^[A-Za-z]+$"

    # Test with numeric type constraints
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 5
    }
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Number)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.multiple_of == 5

    # Test with object type constraints
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        },
        "required": ["name"]
    }
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert "age" in result.properties
    assert isinstance(result.properties["name"], String)
    assert isinstance(result.properties["age"], Number)
    assert "name" in result.required

    # Test with array type constraints
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True
    }
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    assert result.min_items == 1
    assert result.max_items == 10
    assert result.unique_items is True


# LLM-generated content at query #4
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
    assert field.all_of[0].min_length == 5
    assert isinstance(field.all_of[1], String)
    assert field.all_of[1].max_length == 10
    assert field.default == "test"


# LLM-generated content at query #5
#--------------------------

```python
def test_enum_from_json_schema():
    # Test with simple enum values
    data = {"enum": ["a", "b", "c"]}
    field = enum_from_json_schema(data, definitions=Definitions())
    assert isinstance(field, Choice)
    assert field.choices == [("a", "a"), ("b", "b"), ("c", "c")]
    assert field.default == NO_DEFAULT

    # Test with default value
    data_with_default = {"enum": [1, 2, 3], "default": 2}
    field_with_default = enum_from_json_schema(data_with_default, definitions=Definitions())
    assert field_with_default.choices == [(1, 1), (2, 2), (3, 3)]
    assert field_with_default.default == 2

    # Test with mixed types in enum
    data_mixed = {"enum": [True, False, None]}
    field_mixed = enum_from_json_schema(data_mixed, definitions=Definitions())
    assert field_mixed.choices == [(True, True), (False, False), (None, None)]


# LLM-generated content at query #6
#--------------------------

```python
def test_if_then_else_from_json_schema():
    definitions = Definitions()

    # Test with all clauses
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

    # Test with only if and then
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert isinstance(result.then_clause, Number)
    assert result.else_clause is None

    # Test with only if and else
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert result.then_clause is None
    assert isinstance(result.else_clause, Boolean)

    # Test with only if
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


# LLM-generated content at query #7
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
    ref_field = Reference(to="Test", target=String(), definitions={"Test": String()})
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {
            "schemas": {"Test": {"type": "string"}}
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


# LLM-generated content at query #8
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
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Number)
    assert isinstance(field.else_clause, Boolean)

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"}
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Number)
    assert field.else_clause is None

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"}
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert field.then_clause is None
    assert isinstance(field.else_clause, Boolean)

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
        "then": {"type": "number"},
        "else": {"type": "boolean"},
        "default": 42
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert field.default == 42


# LLM-generated content at query #9
#--------------------------

```python
def test_ref_from_json_schema():
    definitions = Definitions()
    definitions["#/test"] = String()

    data = {"$ref": "#/test"}
    field = ref_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/test"
    assert field.definitions == definitions

    data = {"$ref": "#/another"}
    definitions["#/another"] = Integer()
    field = ref_from_json_schema(data, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.to == "#/another"
    assert field.definitions == definitions


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
            {"type": "string", "minLength": 1},
            {"type": "number", "minimum": 0},
            {"type": "boolean"}
        ],
        "default": "default_value"
    }
    result = one_of_from_json_schema(schema, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 3
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Number)
    assert isinstance(result.one_of[2], Boolean)
    assert result.default == "default_value"

    # Test with reference in oneOf
    definitions["#/components/schemas/TestRef"] = from_json_schema({"type": "string"})
    schema = {
        "oneOf": [
            {"$ref": "#/components/schemas/TestRef"},
            {"type": "number"}
        ]
    }
    result = one_of_from_json_schema(schema, definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Reference)
    assert isinstance(result.one_of[1], Number)


# LLM-generated content at query #12
#--------------------------

```python
def test_one_of_from_json_schema():
    # Test with simple oneOf schema
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result = one_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Number)

    # Test with oneOf schema containing references
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    data = {
        "oneOf": [
            {"$ref": "#/components/schemas/Test"},
            {"type": "number"}
        ]
    }
    result = one_of_from_json_schema(data, definitions=definitions)
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Reference)
    assert isinstance(result.one_of[1], Number)

    # Test with oneOf schema containing default value
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test"
    }
    result = one_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert result.default == "test"


# LLM-generated content at query #13
#--------------------------

```python
def test_one_of_from_json_schema():
    # Test basic oneOf with simple types
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }
    result = one_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Number)

    # Test oneOf with nested objects
    data = {
        "oneOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "object", "properties": {"b": {"type": "number"}}}
        ]
    }
    result = one_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Object)
    assert isinstance(result.one_of[1], Object)

    # Test oneOf with default value
    data = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test"
    }
    result = one_of_from_json_schema(data, definitions=Definitions())
    assert result.default == "test"

    # Test oneOf with complex nested schemas
    data = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            },
            {
                "type": "array",
                "items": {"type": "string"}
            }
        ]
    }
    result = one_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], Object)
    assert isinstance(result.one_of[1], Array)


# LLM-generated content at query #14
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
    result = one_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, OneOf)
    assert len(result.one_of) == 2
    assert isinstance(result.one_of[0], String)
    assert isinstance(result.one_of[1], Number)

    # Test oneOf with default value
    data_with_default = {
        "oneOf": [
            {"type": "string"},
            {"type": "number"}
        ],
        "default": "test"
    }
    result_with_default = one_of_from_json_schema(data_with_default, definitions=Definitions())
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
    result_complex = one_of_from_json_schema(data_complex, definitions=Definitions())
    assert isinstance(result_complex, OneOf)
    assert len(result_complex.one_of) == 2
    assert isinstance(result_complex.one_of[0], Object)
    assert isinstance(result_complex.one_of[1], Array)

    # Test oneOf with references
    definitions = Definitions()
    definitions["#/components/schemas/Person"] = from_json_schema({
        "type": "object",
        "properties": {"name": {"type": "string"}}
    })
    data_with_ref = {
        "oneOf": [
            {"$ref": "#/components/schemas/Person"},
            {"type": "string"}
        ]
    }
    result_with_ref = one_of_from_json_schema(data_with_ref, definitions=definitions)
    assert isinstance(result_with_ref, OneOf)
    assert len(result_with_ref.one_of) == 2
    assert isinstance(result_with_ref.one_of[0], Reference)
    assert isinstance(result_with_ref.one_of[1], String)


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
        "multipleOf": 10,
        "default": 50
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 10
    assert field.default == 50

    # Test integer type
    data = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 100,
        "multipleOf": 10,
        "default": 50
    }
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100
    assert field.multiple_of == 10
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
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 30}


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
    assert field.allow_null is False

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
    assert field.allow_null is False

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
    assert field.allow_null is False

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default is True
    assert field.allow_null is False

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
    assert field.allow_null is False

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
    assert field.allow_null is False

    # Test allow_null parameter
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #17
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
    all_of_schema = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    result = from_json_schema(all_of_schema)
    assert isinstance(result, AllOf)
    assert len(result.schemas) == 2

    # Test anyOf schema
    any_of_schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(any_of_schema)
    assert isinstance(result, OneOf)
    assert len(result.schemas) == 2

    # Test oneOf schema
    one_of_schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    result = from_json_schema(one_of_schema)
    assert isinstance(result, OneOf)
    assert len(result.schemas) == 2

    # Test not schema
    not_schema = {"not": {"type": "string"}}
    result = from_json_schema(not_schema)
    assert isinstance(result, Not)
    assert isinstance(result.schema, String)

    # Test if-then-else schema
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

    # Test reference schema
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
        "pattern": "^[a-z]+$"
    }
    result = from_json_schema(combined_schema)
    assert isinstance(result, AllOf)
    assert len(result.schemas) == 4

    # Test empty schema (should return Any)
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #18
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
    int_field = Integer(minimum=0, maximum=100)
    expected_int_schema = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected_int_schema

    # Test Float field
    float_field = Float(minimum=0.0, maximum=1.0)
    expected_float_schema = {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0
    }
    assert to_json_schema(float_field) == expected_float_schema

    # Test Boolean field
    bool_field = Boolean()
    expected_bool_schema = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected_bool_schema

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=10)
    expected_array_schema = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10
    }
    assert to_json_schema(array_field) == expected_array_schema

    # Test Object field
    object_field = Object(properties={"name": String(), "age": Integer()})
    expected_object_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
    assert to_json_schema(object_field) == expected_object_schema

    # Test Choice field
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected_choice_schema = {"enum": ["a", "b"]}
    assert to_json_schema(choice_field) == expected_choice_schema

    # Test Const field
    const_field = Const(const="fixed_value")
    expected_const_schema = {"const": "fixed_value"}
    assert to_json_schema(const_field) == expected_const_schema

    # Test Union field
    union_field = Union(any_of=[String(), Integer()])
    expected_union_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    assert to_json_schema(union_field) == expected_union_schema

    # Test AllOf field
    all_of_field = AllOf(all_of=[String(), Integer()])
    expected_all_of_schema = {
        "allOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    assert to_json_schema(all_of_field) == expected_all_of_schema

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected_one_of_schema = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    assert to_json_schema(one_of_field) == expected_one_of_schema

    # Test Not field
    not_field = Not(negated=String())
    expected_not_schema = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected_not_schema

    # Test IfThenElse field
    if_then_else_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    expected_if_then_else_schema = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_then_else_field) == expected_if_then_else_schema

    # Test Reference field
    definitions = Definitions({"Person": Object(properties={"name": String()})})
    ref_field = Reference(to="Person", definitions=definitions)
    expected_ref_schema = {
        "$ref": "#/components/schemas/Person",
        "components": {
            "schemas": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
    }
    assert to_json_schema(ref_field) == expected_ref_schema

    # Test Schema field
    schema_field = Schema(fields={"name": String(), "age": Integer()})
    expected_schema_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
    assert to_json_schema(schema_field) == expected_schema_schema


# LLM-generated content at query #19
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
    const_field = from_json_schema({"const": "value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "value"

    # Test allOf
    all_of_field = from_json_schema({
        "allOf": [
            {"type": "string"},
            {"minLength": 5}
        ]
    })
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.constraints) == 2

    # Test anyOf
    any_of_field = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.options) == 2

    # Test oneOf
    one_of_field = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.options) == 2

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

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "string"})
    ref_field = from_json_schema(
        {"$ref": "#/components/schemas/Test"},
        definitions=definitions
    )
    assert isinstance(ref_field, Reference)

    # Test combined constraints
    combined_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10
    })
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.constraints) == 2

    # Test no constraints
    any_field = from_json_schema({})
    assert isinstance(any_field, Any)


# LLM-generated content at query #20
#--------------------------

```python
def test_to_json_schema():
    # Test Any type
    assert to_json_schema(Any()) == True

    # Test NeverMatch type
    assert to_json_schema(NeverMatch()) == False

    # Test String type
    string_field = String(min_length=1, max_length=10, allow_null=True)
    expected = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10
    }
    assert to_json_schema(string_field) == expected

    # Test Integer type
    int_field = Integer(minimum=0, maximum=100, allow_null=False)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected

    # Test Float type
    float_field = Float(multiple_of=0.5, allow_null=True)
    expected = {
        "type": ["number", "null"],
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean type
    bool_field = Boolean(allow_null=True)
    expected = {
        "type": ["boolean", "null"]
    }
    assert to_json_schema(bool_field) == expected

    # Test Array type
    array_field = Array(items=String(), min_items=1, allow_null=False)
    expected = {
        "type": "array",
        "minItems": 1,
        "items": {"type": "string"}
    }
    assert to_json_schema(array_field) == expected

    # Test Object type
    object_field = Object(properties={"name": String()}, allow_null=True)
    expected = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}}
    }
    assert to_json_schema(object_field) == expected

    # Test Choice type
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected

    # Test Const type
    const_field = Const(const="fixed")
    expected = {
        "const": "fixed"
    }
    assert to_json_schema(const_field) == expected

    # Test Union type
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected

    # Test AllOf type
    all_of_field = AllOf(all_of=[String(), Const("test")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "test"}]
    }
    assert to_json_schema(all_of_field) == expected

    # Test Reference type
    definitions = Definitions()
    ref_field = Reference(to="test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/test",
        "components": {"schemas": {}}
    }
    assert to_json_schema(ref_field) == expected

    # Test Schema type
    schema_field = Schema(fields={"name": String()}, allow_null=True)
    expected = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}}
    }
    assert to_json_schema(schema_field) == expected

    # Test IfThenElse type
    if_field = IfThenElse(if_clause=String(), then_clause=Integer())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not type
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_to_json_schema():
    # Test Any
    assert to_json_schema(Any()) == True

    # Test NeverMatch
    assert to_json_schema(NeverMatch()) == False

    # Test String
    field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(field) == expected

    # Test Integer
    field = Integer(minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(field) == expected

    # Test Float
    field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(field) == expected

    # Test Boolean
    field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(field) == expected

    # Test Array
    field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(field) == expected

    # Test Object
    field = Object(properties={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(field) == expected

    # Test Choice
    field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(field) == expected

    # Test Const
    field = Const(const="value")
    expected = {
        "const": "value"
    }
    assert to_json_schema(field) == expected

    # Test Union
    field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(field) == expected

    # Test OneOf
    field = OneOf(one_of=[String(), Integer()])
    expected = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(field) == expected

    # Test AllOf
    field = AllOf(all_of=[String(), Integer()])
    expected = {
        "allOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(field) == expected

    # Test IfThenElse
    field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(field) == expected

    # Test Not
    field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(field) == expected

    # Test Reference
    definitions = Definitions()
    field = Reference(to="test", target=String(), definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/test",
        "components": {
            "schemas": {
                "test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(field) == expected

    # Test Schema
    field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(field) == expected


# LLM-generated content at query #22
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
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float type
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean type
    bool_field = Boolean()
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array type
    array_field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object type
    object_field = Object(
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
    assert to_json_schema(object_field) == expected

    # Test Choice type
    choice_field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected

    # Test Const type
    const_field = Const(const="fixed_value")
    expected = {
        "const": "fixed_value"
    }
    assert to_json_schema(const_field) == expected

    # Test Union type
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected

    # Test AllOf type
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "test"}]
    }
    assert to_json_schema(all_of_field) == expected

    # Test Reference type
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

    # Test Schema type
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

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
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected

    # Test with definitions
    definitions = Definitions({
        "Person": Object(properties={"name": String()})
    })
    expected = {
        "components": {
            "schemas": {
                "Person": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #23
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
    schema = {"enum": ["a", "b", "c"]}
    field = from_json_schema(schema)
    assert isinstance(field, Choice)
    assert field.choices == ["a", "b", "c"]

    # Test const constraint
    schema = {"const": "fixed_value"}
    field = from_json_schema(schema)
    assert isinstance(field, Const)
    assert field.value == "fixed_value"

    # Test allOf constraint
    schema = {
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    }
    field = from_json_schema(schema)
    assert isinstance(field, AllOf)
    assert len(field.schemas) == 2

    # Test anyOf constraint
    schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    field = from_json_schema(schema)
    assert isinstance(field, OneOf)
    assert len(field.schemas) == 2

    # Test oneOf constraint
    schema = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }
    field = from_json_schema(schema)
    assert isinstance(field, OneOf)
    assert len(field.schemas) == 2

    # Test not constraint
    schema = {
        "not": {"type": "string"}
    }
    field = from_json_schema(schema)
    assert isinstance(field, Not)
    assert isinstance(field.schema, String)

    # Test if-then-else constraint
    schema = {
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"maxLength": 10}
    }
    field = from_json_schema(schema)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_schema, String)
    assert isinstance(field.then_schema, String)
    assert isinstance(field.else_schema, String)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    schema = {"$ref": "#/components/schemas/Test"}
    field = from_json_schema(schema, definitions=definitions)
    assert isinstance(field, Reference)
    assert field.ref == "#/components/schemas/Test"

    # Test combined constraints
    schema = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
        "pattern": "^[a-zA-Z]+$"
    }
    field = from_json_schema(schema)
    assert isinstance(field, AllOf)
    assert len(field.schemas) == 4

    # Test no constraints
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #24
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
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "minProperties": 1,
        "maxProperties": 2,
        "required": ["name"],
        "default": {"name": "test", "age": 25}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 25}


# LLM-generated content at query #25
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
    object_field = Object(
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

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
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
    definitions = Definitions()
    ref_field = Reference(to="Test", definitions=definitions)
    definitions["Test"] = String()
    expected = {
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        },
        "$ref": "#/components/schemas/Test"
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


# LLM-generated content at query #26
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
        "maxLength": 100,
        "pattern": "^[A-Za-z]+$",
        "default": "hello"
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
        "default": ["a", "b"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["a", "b"]

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


# LLM-generated content at query #27
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

    # Test default values
    data = {"type": "string", "default": "test"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert field.default == "test"


# LLM-generated content at query #28
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(field) == expected

    # Test Integer field
    field = Integer(minimum=0, maximum=100, exclusive_minimum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "exclusiveMinimum": True,
        "maximum": 100
    }
    assert to_json_schema(field) == expected

    # Test Float field
    field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(field) == expected

    # Test Boolean field
    field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(field) == expected

    # Test Array field
    field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(field) == expected

    # Test Object field
    field = Object(
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
    assert to_json_schema(field) == expected

    # Test Choice field
    field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(field) == expected

    # Test Const field
    field = Const(const="value")
    expected = {
        "const": "value"
    }
    assert to_json_schema(field) == expected

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(field) == expected

    # Test AllOf field
    field = AllOf(all_of=[String(), Const(const="value")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "value"}]
    }
    assert to_json_schema(field) == expected

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    expected = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(field) == expected

    # Test Not field
    field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(field) == expected

    # Test IfThenElse field
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

    # Test Reference field
    definitions = Definitions({"Test": String()})
    field = Reference(to="Test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(field) == expected

    # Test Schema field
    field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(field) == expected

    # Test Definitions
    definitions = Definitions({
        "StringField": String(),
        "IntegerField": Integer()
    })
    expected = {
        "components": {
            "schemas": {
                "StringField": {"type": "string"},
                "IntegerField": {"type": "integer"}
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #29
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
        "maxProperties": 10,
        "default": {"name": "John", "age": 30}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #30
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
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, exclusive_minimum=True, multiple_of=2)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "multipleOf": 2
    }
    assert to_json_schema(integer_field) == expected

    # Test Float
    float_field = Float(allow_null=True, minimum=0.0, maximum=1.0, exclusive_maximum=True)
    expected = {
        "type": ["number", "null"],
        "minimum": 0.0,
        "maximum": 1.0,
        "exclusiveMaximum": True
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean
    boolean_field = Boolean(allow_null=False)
    expected = {"type": "boolean"}
    assert to_json_schema(boolean_field) == expected

    # Test Array
    array_field = Array(allow_null=True, min_items=1, max_items=5, items=String(), additional_items=False, unique_items=True)
    expected = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"},
        "additionalItems": False,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object
    object_field = Object(
        allow_null=False,
        properties={"name": String()},
        pattern_properties={"^S_": String()},
        additional_properties=True,
        property_names=String(),
        min_properties=1,
        max_properties=10,
        required=["name"]
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "patternProperties": {"^S_": {"type": "string"}},
        "additionalProperties": True,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"]
    }
    assert to_json_schema(object_field) == expected

    # Test Choice
    choice_field = Choice(choices=[("a", "a"), ("b", "b")], default="a")
    expected = {"enum": ["a", "b"], "default": "a"}
    assert to_json_schema(choice_field) == expected

    # Test Const
    const_field = Const(const="fixed_value", default="fixed_value")
    expected = {"const": "fixed_value", "default": "fixed_value"}
    assert to_json_schema(const_field) == expected

    # Test Union
    union_field = Union(any_of=[String(), Integer()])
    expected = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(union_field) == expected

    # Test OneOf
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(one_of_field) == expected

    # Test AllOf
    all_of_field = AllOf(all_of=[String(), Integer()])
    expected = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    assert to_json_schema(all_of_field) == expected

    # Test IfThenElse
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

    # Test Not
    not_field = Not(negated=String())
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected

    # Test Reference
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

    # Test Schema
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected


# LLM-generated content at query #31
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
        "uniqueItems": True,
        "default": ["a", "b"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items == True
    assert field.default == ["a", "b"]

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


# LLM-generated content at query #32
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
    integer_field = Integer(minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(integer_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    boolean_field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(boolean_field) == expected

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
    object_field = Object(
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
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "test"}]
    }
    assert to_json_schema(all_of_field) == expected

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(one_of_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected

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

    # Test Reference field
    definitions = Definitions()
    reference_field = Reference(to="test", definitions=definitions)
    definitions["test"] = String()
    expected = {
        "$ref": "#/components/schemas/test",
        "components": {
            "schemas": {
                "test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(reference_field) == expected

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected


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


# LLM-generated content at query #34
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
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1

    # Test object type
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
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

    # Test default values
    data = {"type": "string", "default": "test"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert field.default == "test"

    # Test invalid type_string
    with pytest.raises(AssertionError):
        from_json_schema_type({"type": "invalid"}, "invalid", False, Definitions())


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


# LLM-generated content at query #36
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
        "properties": {"name": {"type": "string"}},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test"}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #37
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
    object_field = Object(
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
    if_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

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

    # Test null types
    string_field_null = String(allow_null=True)
    expected = {"type": ["string", "null"]}
    assert to_json_schema(string_field_null) == expected

    # Test default values
    string_field_default = String(default="default_value")
    expected = {"type": "string", "default": "default_value"}
    assert to_json_schema(string_field_default) == expected


# LLM-generated content at query #38
#--------------------------

```python
def test_to_json_schema():
    # Test Any
    assert to_json_schema(Any()) == True

    # Test NeverMatch
    assert to_json_schema(NeverMatch()) == False

    # Test String
    field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(field) == expected

    # Test Integer
    field = Integer(minimum=0, maximum=100, exclusive_maximum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMaximum": True
    }
    assert to_json_schema(field) == expected

    # Test Float
    field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(field) == expected

    # Test Boolean
    field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(field) == expected

    # Test Array
    field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(field) == expected

    # Test Object
    field = Object(
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
    assert to_json_schema(field) == expected

    # Test Choice
    field = Choice(choices=[("a", "a"), ("b", "b")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(field) == expected

    # Test Const
    field = Const(const="fixed")
    expected = {
        "const": "fixed"
    }
    assert to_json_schema(field) == expected

    # Test Union
    field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(field) == expected

    # Test AllOf
    field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    expected = {
        "allOf": [{"type": "string", "minLength": 1}, {"type": "string", "maxLength": 10}]
    }
    assert to_json_schema(field) == expected

    # Test OneOf
    field = OneOf(one_of=[String(), Integer()])
    expected = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(field) == expected

    # Test Not
    field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
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

    # Test Reference
    definitions = Definitions({"Test": String()})
    field = Reference(to="Test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(field) == expected

    # Test Schema
    field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(field) == expected

    # Test allow_null
    field = String(allow_null=True)
    expected = {
        "type": ["string", "null"]
    }
    assert to_json_schema(field) == expected


# LLM-generated content at query #39
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
        "additionalItems": False,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.additional_items is False
    assert field.unique_items is True
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
    assert field.additional_properties is False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 25}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_valid_types():
    # Test with a single string type
    data = {"type": "string"}
    types, allow_null = get_valid_types(data)
    assert types == {"string"}
    assert allow_null is False

    # Test with multiple types
    data = {"type": ["string", "number"]}
    types, allow_null = get_valid_types(data)
    assert types == {"string", "number"}
    assert allow_null is False

    # Test with null type included
    data = {"type": ["string", "null"]}
    types, allow_null = get_valid_types(data)
    assert types == {"string"}
    assert allow_null is True

    # Test with no type specified
    data = {}
    types, allow_null = get_valid_types(data)
    assert types == {"boolean", "object", "array", "number", "string"}
    assert allow_null is False

    # Test with integer and number types
    data = {"type": ["integer", "number"]}
    types, allow_null = get_valid_types(data)
    assert types == {"number"}
    assert allow_null is False


# LLM-generated content at query #2
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
    data_with_default = {"enum": [1, 2, 3], "default": 2}
    field_with_default = enum_from_json_schema(data_with_default, definitions=Definitions())
    assert isinstance(field_with_default, Choice)
    assert field_with_default.choices == [(1, 1), (2, 2), (3, 3)]
    assert field_with_default.default == 2

    # Test with mixed types in enum
    data_mixed = {"enum": [True, False, None]}
    field_mixed = enum_from_json_schema(data_mixed, definitions=Definitions())
    assert isinstance(field_mixed, Choice)
    assert field_mixed.choices == [(True, True), (False, False), (None, None)]
    assert field_mixed.default == NO_DEFAULT


# LLM-generated content at query #3
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
    data_with_default = {"enum": [1, 2, 3], "default": 2}
    field_with_default = enum_from_json_schema(data_with_default, definitions=Definitions())
    assert isinstance(field_with_default, Choice)
    assert field_with_default.choices == [(1, 1), (2, 2), (3, 3)]
    assert field_with_default.default == 2

    # Test with mixed types in enum
    data_mixed = {"enum": [True, "text", 123]}
    field_mixed = enum_from_json_schema(data_mixed, definitions=Definitions())
    assert isinstance(field_mixed, Choice)
    assert field_mixed.choices == [(True, True), ("text", "text"), (123, 123)]


# LLM-generated content at query #4
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
    assert isinstance(result.any_of[1], Float)

    # Test with nested anyOf schema
    data = {
        "anyOf": [
            {"type": "string", "minLength": 1},
            {"type": "number", "minimum": 0}
        ],
        "default": "test"
    }
    result = any_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert result.any_of[0].min_length == 1
    assert result.any_of[1].minimum == 0
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
    assert isinstance(result.any_of[1], Float)

    # Test with empty anyOf
    data = {"anyOf": []}
    result = any_of_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_all_of_from_json_schema():
    definitions = Definitions()

    # Test with simple allOf
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

    # Test with nested allOf
    data = {
        "allOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "object", "properties": {"age": {"type": "integer"}}}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert len(result.all_of) == 2
    assert isinstance(result.all_of[0], Object)
    assert isinstance(result.all_of[1], Object)

    # Test with default value
    data = {
        "allOf": [{"type": "string"}],
        "default": "test"
    }
    result = all_of_from_json_schema(data, definitions)
    assert result.default == "test"

    # Test with reference in allOf
    definitions["#/components/schemas/Test"] = String()
    data = {
        "allOf": [
            {"$ref": "#/components/schemas/Test"},
            {"type": "string", "minLength": 1}
        ]
    }
    result = all_of_from_json_schema(data, definitions)
    assert isinstance(result, AllOf)
    assert isinstance(result.all_of[0], Reference)
    assert isinstance(result.all_of[1], String)


# LLM-generated content at query #6
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with if, then, and else clauses
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
    assert isinstance(field.then_clause, Number)
    assert isinstance(field.else_clause, Boolean)
    assert field.default == 42

    # Test with only if and then clauses
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "default": 3.14
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert isinstance(field.then_clause, Number)
    assert field.else_clause is None
    assert field.default == 3.14

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"},
        "default": True
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert field.then_clause is None
    assert isinstance(field.else_clause, Boolean)
    assert field.default == True

    # Test with only if clause
    data = {
        "if": {"type": "string"},
        "default": "test"
    }
    field = if_then_else_from_json_schema(data, definitions)
    assert isinstance(field, IfThenElse)
    assert isinstance(field.if_clause, String)
    assert field.then_clause is None
    assert field.else_clause is None
    assert field.default == "test"


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

    # Test oneOf with definitions
    definitions = Definitions()
    definitions["#/components/schemas/StringSchema"] = String()
    data_with_ref = {
        "oneOf": [
            {"$ref": "#/components/schemas/StringSchema"},
            {"type": "number"}
        ]
    }
    field_with_ref = one_of_from_json_schema(data_with_ref, definitions=definitions)
    assert isinstance(field_with_ref, OneOf)
    assert len(field_with_ref.one_of) == 2
    assert isinstance(field_with_ref.one_of[0], Reference)
    assert isinstance(field_with_ref.one_of[1], Number)


# LLM-generated content at query #8
#--------------------------

```python
def test_ref_from_json_schema():
    definitions = Definitions()
    data = {"$ref": "#/components/schemas/Test"}
    result = ref_from_json_schema(data, definitions)
    assert isinstance(result, Reference)
    assert result.to == "#/components/schemas/Test"
    assert result.definitions == definitions


# LLM-generated content at query #9
#--------------------------

```python
def test_if_then_else_from_json_schema():
    # Test with all clauses present
    data = {
        "if": {"type": "string"},
        "then": {"type": "number"},
        "else": {"type": "boolean"}
    }
    definitions = Definitions()
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

    # Test with only if and else clauses
    data = {
        "if": {"type": "string"},
        "else": {"type": "boolean"}
    }
    result = if_then_else_from_json_schema(data, definitions)
    assert isinstance(result, IfThenElse)
    assert isinstance(result.if_clause, String)
    assert result.then_clause is None
    assert isinstance(result.else_clause, Boolean)

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


# LLM-generated content at query #10
#--------------------------

```python
def test_one_of_from_json_schema():
    definitions = Definitions()

    # Test with simple oneOf schema
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
    assert isinstance(field.one_of[1], Number)

    # Test with nested oneOf schema
    data = {
        "oneOf": [
            {"type": "string", "minLength": 1},
            {"type": "number", "minimum": 0},
            {"type": "boolean"}
        ],
        "default": "default_value"
    }
    field = one_of_from_json_schema(data, definitions)
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 3
    assert isinstance(field.one_of[0], String)
    assert isinstance(field.one_of[1], Number)
    assert isinstance(field.one_of[2], Boolean)
    assert field.default == "default_value"

    # Test with reference in oneOf
    definitions["#/components/schemas/TestRef"] = String()
    data = {
        "oneOf": [
            {"$ref": "#/components/schemas/TestRef"},
            {"type": "number"}
        ]
    }
    field = one_of_from_json_schema(data, definitions)
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 2
    assert isinstance(field.one_of[0], Reference)
    assert isinstance(field.one_of[1], Number)

    # Test with empty oneOf (should raise an error or return empty OneOf)
    data = {"oneOf": []}
    field = one_of_from_json_schema(data, definitions)
    assert isinstance(field, OneOf)
    assert len(field.one_of) == 0


# LLM-generated content at query #11
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
        "maxLength": 100,
        "pattern": "^[A-Za-z]+$",
        "default": "hello"
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
        "default": ["a", "b"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items is True
    assert field.default == ["a", "b"]

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
def test_from_json_schema_type():
    definitions = Definitions()

    # Test number type
    data = {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "default": 50.5
    }
    field = from_json_schema_type(data, "number", False, definitions)
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
        "maxItems": 5,
        "uniqueItems": True,
        "default": ["item1"]
    }
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items == True
    assert field.default == ["item1"]

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

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, definitions)
    assert field.allow_null == True


# LLM-generated content at query #13
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type
    data = {"type": "string"}
    assert isinstance(type_from_json_schema(data, Definitions()), String)

    # Test multiple types (Union)
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

    # Test nullable type
    data = {"type": "string", "nullable": True}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Union)
    assert any(isinstance(field, String) for field in result.any_of)
    assert any(isinstance(field, Const) and field.value is None for field in result.any_of)

    # Test no type with nullable
    data = {"nullable": True}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test no type without nullable
    data = {}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, NeverMatch)

    # Test complex nested types
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

    # Test array type
    data = {"type": "array", "items": {"type": "string"}}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)


# LLM-generated content at query #14
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
        "allOf": [{"type": "string"}, {"minLength": 5}]
    })
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.schemas) == 2

    # Test anyOf
    any_of_field = from_json_schema({
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    })
    assert isinstance(any_of_field, Union)
    assert len(any_of_field.schemas) == 2

    # Test oneOf
    one_of_field = from_json_schema({
        "oneOf": [{"type": "string"}, {"type": "integer"}]
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
    definitions["#/components/schemas/Test"] = from_json_schema({"type": "string"})
    ref_field = from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=definitions)
    assert isinstance(ref_field, Reference)
    assert ref_field.ref == "#/components/schemas/Test"

    # Test combined constraints
    combined_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10
    })
    assert isinstance(combined_field, String)
    assert combined_field.min_length == 5
    assert combined_field.max_length == 10

    # Test multiple constraints
    multi_constraint_field = from_json_schema({
        "type": "string",
        "enum": ["a", "b"],
        "const": "a"
    })
    assert isinstance(multi_constraint_field, AllOf)
    assert len(multi_constraint_field.schemas) == 3


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
    const_field = from_json_schema({"const": "value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "value"

    # Test allOf
    all_of_field = from_json_schema({
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    })
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.constraints) == 2

    # Test anyOf
    any_of_field = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.options) == 2

    # Test oneOf
    one_of_field = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.options) == 2

    # Test not
    not_field = from_json_schema({
        "not": {"type": "string"}
    })
    assert isinstance(not_field, Not)
    assert isinstance(not_field.schema, String)

    # Test if-then-else
    if_then_else_field = from_json_schema({
        "if": {"type": "string"},
        "then": {"type": "string", "minLength": 5},
        "else": {"type": "integer"}
    })
    assert isinstance(if_then_else_field, IfThenElse)
    assert isinstance(if_then_else_field.if_schema, String)
    assert isinstance(if_then_else_field.then_schema, String)
    assert isinstance(if_then_else_field.else_schema, Integer)

    # Test reference
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

    # Test no constraints
    any_field = from_json_schema({})
    assert isinstance(any_field, Any)


# LLM-generated content at query #17
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
    data = {"type": "string", "minLength": 5, "maxLength": 10, "format": "email"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10
    assert field.format == "email"

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
        "minProperties": 1,
        "maxProperties": 5
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 5

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #18
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
    int_field = Integer(minimum=0, maximum=100, exclusive_maximum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMaximum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float type
    float_field = Float(multiple_of=0.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean type
    bool_field = Boolean()
    expected = {
        "type": "boolean"
    }
    assert to_json_schema(bool_field) == expected

    # Test Array type
    array_field = Array(items=String(), min_items=1, max_items=5, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object type
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

    # Test Choice type
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    expected = {
        "enum": ["a", "b"]
    }
    assert to_json_schema(choice_field) == expected

    # Test Const type
    const_field = Const(const="fixed_value")
    expected = {
        "const": "fixed_value"
    }
    assert to_json_schema(const_field) == expected

    # Test Union type
    union_field = Union(any_of=[String(), Integer()])
    expected = {
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(union_field) == expected

    # Test AllOf type
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    expected = {
        "allOf": [
            {"type": "string", "minLength": 1},
            {"type": "string", "maxLength": 10}
        ]
    }
    assert to_json_schema(all_of_field) == expected

    # Test OneOf type
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(one_of_field) == expected

    # Test Not type
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected

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

    # Test Reference type
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

    # Test Schema type
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test Definitions type
    definitions = Definitions({
        "Person": Object(properties={"name": String()}),
        "Address": Object(properties={"street": String()})
    })
    expected = {
        "components": {
            "schemas": {
                "Person": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                },
                "Address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}}
                }
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #19
#--------------------------

```python
def test_to_json_schema():
    # Test Any
    assert to_json_schema(Any()) == True

    # Test NeverMatch
    assert to_json_schema(NeverMatch()) == False

    # Test String
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern=r"^[a-z]+$", format="email")
    expected = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "^[a-z]+$",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer
    int_field = Integer(allow_null=False, minimum=0, maximum=100)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
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
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"}
    }
    assert to_json_schema(array_field) == expected

    # Test Object
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
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    expected = {"allOf": [{"type": "string", "minLength": 1}, {"type": "string", "maxLength": 10}]}
    assert to_json_schema(all_of_field) == expected

    # Test Reference
    definitions = Definitions()
    ref_field = Reference(to="TestSchema", definitions=definitions)
    expected = {"$ref": "#/components/schemas/TestSchema"}
    assert to_json_schema(ref_field) == expected

    # Test Schema with definitions
    schema = Schema(fields={"name": String()}, allow_null=True)
    definitions = {"TestSchema": schema}
    expected = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}},
        "components": {
            "schemas": {
                "TestSchema": {
                    "type": ["object", "null"],
                    "properties": {"name": {"type": "string"}}
                }
            }
        }
    }
    assert to_json_schema(definitions) == expected


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
    assert "name" in field.properties
    assert "age" in field.properties
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.default == {"name": "test", "age": 25}


# LLM-generated content at query #21
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type
    data = {"type": "string"}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)

    # Test multiple types (Union)
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Number)

    # Test nullable type
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test empty type with allow_null
    data = {}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test empty type without allow_null
    data = {"type": []}
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, NeverMatch)

    # Test object type with properties
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        }
    }
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Object)
    assert "name" in result.properties
    assert "age" in result.properties
    assert isinstance(result.properties["name"], String)
    assert isinstance(result.properties["age"], Number)

    # Test array type with items
    data = {
        "type": "array",
        "items": {"type": "string"}
    }
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)

    # Test with additional constraints
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10
    }
    result = type_from_json_schema(data, definitions=Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 10


# LLM-generated content at query #22
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type
    data = {"type": "string"}
    field = type_from_json_schema(data, Definitions())
    assert isinstance(field, String)

    # Test multiple types (Union)
    data = {"type": ["string", "number"]}
    field = type_from_json_schema(data, Definitions())
    assert isinstance(field, Union)
    assert len(field.any_of) == 2
    assert any(isinstance(f, String) for f in field.any_of)
    assert any(isinstance(f, Number) for f in field.any_of)

    # Test nullable type
    data = {"type": "string", "nullable": True}
    field = type_from_json_schema(data, Definitions())
    assert isinstance(field, String)
    assert field.allow_null is True

    # Test no type with nullable
    data = {"nullable": True}
    field = type_from_json_schema(data, Definitions())
    assert isinstance(field, Const)
    assert field.value is None

    # Test no type without nullable
    data = {}
    field = type_from_json_schema(data, Definitions())
    assert isinstance(field, NeverMatch)

    # Test with definitions
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = {"type": "integer"}
    data = {"$ref": "#/components/schemas/Test"}
    field = type_from_json_schema(data, definitions)
    assert isinstance(field, Integer)


# LLM-generated content at query #23
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
    assert field.default == True

    # Test array type
    data = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["item1"],
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["item1"]

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
        "default": {"name": "John"},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.required == ["name"]
    assert field.default == {"name": "John"}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #24
#--------------------------

```python
def test_type_from_json_schema():
    # Test single type
    data = {"type": "string"}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, String)

    # Test multiple types (Union)
    data = {"type": ["string", "number"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Union)
    assert len(result.any_of) == 2

    # Test nullable type
    data = {"type": ["string", "null"]}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Union)
    assert result.allow_null is True

    # Test no type (allow_null only)
    data = {}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, NeverMatch)

    # Test with allow_null=True and no type
    data = {"type": "null"}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Const)
    assert result.value is None

    # Test with constraints
    data = {"type": "string", "minLength": 5}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5

    # Test with multiple constraints
    data = {"type": "number", "minimum": 0, "maximum": 100}
    result = type_from_json_schema(data, Definitions())
    assert isinstance(result, Number)
    assert result.minimum == 0
    assert result.maximum == 100


# LLM-generated content at query #25
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
        "default": 50,
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
        "default": 50,
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
        "default": "test",
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

    # Test object type
    data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "minProperties": 1,
        "maxProperties": 2,
        "required": ["name"],
        "default": {"name": "test", "age": 30},
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.properties["age"], Integer)
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 30}


# LLM-generated content at query #26
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
    object_field = Object(
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
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    expected = {
        "allOf": [
            {"type": "string", "minLength": 1},
            {"type": "string", "maxLength": 10}
        ]
    }
    assert to_json_schema(all_of_field) == expected

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

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=String(min_length=1),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string", "minLength": 1},
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


# LLM-generated content at query #27
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
        "exclusiveMinimum": True,
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
        max_properties=5
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
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

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(one_of_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=Boolean())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Reference field
    definitions = Definitions()
    ref_field = Reference(to="test_ref", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/test_ref",
        "components": {"schemas": {}}
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
    definitions = Definitions()
    definitions["test"] = String()
    result = to_json_schema(definitions)
    expected = {
        "components": {
            "schemas": {
                "test": {"type": "string"}
            }
        }
    }
    assert result == expected


# LLM-generated content at query #28
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
    const_field = from_json_schema({"const": "value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "value"

    # Test allOf
    all_of_field = from_json_schema({"allOf": [{"type": "string"}, {"minLength": 5}]})
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.constraints) == 2

    # Test anyOf
    any_of_field = from_json_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.options) == 2

    # Test oneOf
    one_of_field = from_json_schema({"oneOf": [{"type": "string"}, {"type": "integer"}]})
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.options) == 2

    # Test not
    not_field = from_json_schema({"not": {"type": "string"}})
    assert isinstance(not_field, Not)
    assert isinstance(not_field.constraint, String)

    # Test if-then-else
    if_then_else_field = from_json_schema({
        "if": {"type": "string"},
        "then": {"minLength": 5},
        "else": {"type": "integer"}
    })
    assert isinstance(if_then_else_field, IfThenElse)
    assert isinstance(if_then_else_field.if_constraint, String)
    assert isinstance(if_then_else_field.then_constraint, String)
    assert isinstance(if_then_else_field.else_constraint, Integer)

    # Test reference
    definitions = Definitions()
    definitions["#/components/schemas/Test"] = String()
    ref_field = from_json_schema({"$ref": "#/components/schemas/Test"}, definitions=definitions)
    assert isinstance(ref_field, Reference)
    assert ref_field.reference == "#/components/schemas/Test"

    # Test multiple constraints
    multi_constraint_field = from_json_schema({
        "type": "string",
        "minLength": 5,
        "maxLength": 10
    })
    assert isinstance(multi_constraint_field, String)
    assert multi_constraint_field.min_length == 5
    assert multi_constraint_field.max_length == 10

    # Test combined constraints
    combined_field = from_json_schema({
        "type": "string",
        "enum": ["a", "b"],
        "minLength": 1
    })
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.constraints) == 2


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
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items is False
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
    assert field.additional_properties is False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 30}

    # Test allow_null
    field = from_json_schema_type({"type": "string"}, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #30
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
    assert field.default is True

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
    assert field.unique_items is True
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


# LLM-generated content at query #31
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
        "additionalItems": {"type": "number"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["test"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert isinstance(field.additional_items, Float)
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
        "additionalProperties": {"type": "boolean"},
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
    assert isinstance(field.additional_properties, Boolean)
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test", "age": 30}

    # Test invalid type
    with pytest.raises(AssertionError):
        from_json_schema_type({}, "invalid", False, Definitions())


# LLM-generated content at query #32
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
    result = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(result, Float)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.exclusive_minimum == 0
    assert result.exclusive_maximum == 100
    assert result.multiple_of == 2
    assert result.default == 50

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
    result = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(result, Integer)
    assert result.minimum == 0
    assert result.maximum == 100
    assert result.exclusive_minimum == 0
    assert result.exclusive_maximum == 100
    assert result.multiple_of == 2
    assert result.default == 50

    # Test string type
    data = {
        "type": "string",
        "minLength": 5,
        "maxLength": 100,
        "format": "email",
        "pattern": "^[a-zA-Z0-9]+$",
        "default": "test"
    }
    result = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(result, String)
    assert result.min_length == 5
    assert result.max_length == 100
    assert result.format == "email"
    assert result.pattern == "^[a-zA-Z0-9]+$"
    assert result.default == "test"

    # Test boolean type
    data = {
        "type": "boolean",
        "default": True
    }
    result = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(result, Boolean)
    assert result.default == True

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
    result = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(result, Array)
    assert isinstance(result.items, String)
    assert result.additional_items == False
    assert result.min_items == 1
    assert result.max_items == 10
    assert result.unique_items == True
    assert result.default == ["test"]

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
    result = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(result, Object)
    assert isinstance(result.properties["name"], String)
    assert isinstance(result.properties["age"], Integer)
    assert isinstance(result.pattern_properties["^S_"], String)
    assert isinstance(result.pattern_properties["^I_"], Integer)
    assert result.additional_properties == False
    assert isinstance(result.property_names, String)
    assert result.min_properties == 1
    assert result.max_properties == 10
    assert result.required == ["name"]
    assert result.default == {"name": "test", "age": 30}

    # Test invalid type_string
    try:
        from_json_schema_type({}, "invalid", False, Definitions())
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid'"


# LLM-generated content at query #34
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
        "properties": {"name": {"type": "string"}},
        "patternProperties": {"^S_": {"type": "string"}},
        "additionalProperties": False,
        "propertyNames": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 10,
        "required": ["name"],
        "default": {"name": "test"}
    }
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert isinstance(field.properties["name"], String)
    assert isinstance(field.pattern_properties["^S_"], String)
    assert field.additional_properties == False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}

    # Test invalid type
    with pytest.raises(AssertionError):
        from_json_schema_type({}, "invalid", False, Definitions())


# LLM-generated content at query #35
#--------------------------

```python
def test_to_json_schema():
    # Test Any
    assert to_json_schema(Any()) == True

    # Test NeverMatch
    assert to_json_schema(NeverMatch()) == False

    # Test String
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer
    int_field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=True)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True
    }
    assert to_json_schema(int_field) == expected

    # Test Float
    float_field = Float(minimum=0.0, maximum=1.0, multiple_of=0.1)
    expected = {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "multipleOf": 0.1
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean
    bool_field = Boolean()
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array
    array_field = Array(items=String(), min_items=1, max_items=10, unique_items=True)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object
    object_field = Object(
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
    all_of_field = AllOf(all_of=[String(), Integer()])
    expected = {"allOf": [{"type": "string"}, {"type": "integer"}]}
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
    definitions = Definitions({"Test": String()})
    ref_field = Reference(to="Test", definitions=definitions)
    expected = {"$ref": "#/components/schemas/Test"}
    result = to_json_schema(ref_field)
    assert result == expected
    assert result["$ref"] in definitions

    # Test Schema
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test Definitions
    definitions = Definitions({"Test": String()})
    expected = {
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #36
#--------------------------

```python
def test_to_json_schema():
    # Test Any type
    assert to_json_schema(Any()) == True

    # Test NeverMatch type
    assert to_json_schema(NeverMatch()) == False

    # Test String type
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer type
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=2)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 2
    }
    assert to_json_schema(int_field) == expected

    # Test Float type
    float_field = Float(allow_null=True, exclusive_minimum=0, exclusive_maximum=1)
    expected = {
        "type": ["number", "null"],
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 1
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean type
    bool_field = Boolean(allow_null=False)
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array type
    array_field = Array(allow_null=True, items=String(), additional_items=False, min_items=1, max_items=5)
    expected = {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 5
    }
    assert to_json_schema(array_field) == expected

    # Test Object type
    object_field = Object(
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

    # Test AllOf type
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(all_of_field) == expected

    # Test Reference type
    definitions = Definitions({"Test": String()})
    ref_field = Reference(to="Test", definitions=definitions)
    expected = {"$ref": "#/components/schemas/Test"}
    result = to_json_schema(ref_field)
    assert result == expected
    assert result["$ref"] in result["components"]["schemas"]

    # Test Schema type
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected

    # Test Definitions
    definitions = Definitions({
        "StringField": String(),
        "IntField": Integer()
    })
    expected = {
        "components": {
            "schemas": {
                "StringField": {"type": "string"},
                "IntField": {"type": "integer"}
            }
        }
    }
    assert to_json_schema(definitions) == expected


# LLM-generated content at query #37
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
        "default": 50.5
    }
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 100
    assert field.exclusive_minimum is True
    assert field.exclusive_maximum is True
    assert field.multiple_of == 2
    assert field.default == 50.5

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
        "default": "test@example.com"
    }
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 100
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9]+$"
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
        "additionalItems": False,
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True,
        "default": ["item1", "item2"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.additional_items is False
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items is True
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
    assert field.additional_properties is False
    assert isinstance(field.property_names, String)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}

    # Test allow_null
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null is True


# LLM-generated content at query #38
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
    array_field = Array(items=String(), min_items=1, max_items=10)
    expected = {
        "type": "array",
        "minItems": 1,
        "maxItems": 10,
        "items": {"type": "string"}
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
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
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(all_of_field) == expected

    # Test Reference field
    definitions = Definitions()
    ref_field = Reference(to="Test", definitions=definitions)
    definitions["Test"] = String()
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

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=Const(const=True),
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
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected


# LLM-generated content at query #39
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
    const_field = from_json_schema({"const": "fixed_value"})
    assert isinstance(const_field, Const)
    assert const_field.value == "fixed_value"

    # Test allOf schema
    all_of_field = from_json_schema({
        "allOf": [
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 10}
        ]
    })
    assert isinstance(all_of_field, AllOf)
    assert len(all_of_field.schemas) == 2

    # Test anyOf schema
    any_of_field = from_json_schema({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(any_of_field, OneOf)
    assert len(any_of_field.schemas) == 2

    # Test oneOf schema
    one_of_field = from_json_schema({
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })
    assert isinstance(one_of_field, OneOf)
    assert len(one_of_field.schemas) == 2

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
        "pattern": "^[a-zA-Z]+$"
    })
    assert isinstance(combined_field, AllOf)
    assert len(combined_field.schemas) == 2  # type and pattern constraints

    # Test empty schema
    assert isinstance(from_json_schema({}), Any)


# LLM-generated content at query #40
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
    object_field = Object(
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
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {"allOf": [{"type": "string"}, {"const": "test"}]}
    assert to_json_schema(all_of_field) == expected

    # Test Reference field
    definitions = Definitions({"Test": String()})
    ref_field = Reference(to="Test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/Test",
        "components": {"schemas": {"Test": {"type": "string"}}}
    }
    assert to_json_schema(ref_field) == expected

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=Const(const=True),
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
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected


# LLM-generated content at query #41
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
        "exclusiveMinimum": True,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5, default=1.5)
    expected = {
        "type": "number",
        "multipleOf": 0.5,
        "default": 1.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean(default=False)
    expected = {
        "type": "boolean",
        "default": False
    }
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(
        items=String(),
        min_items=1,
        max_items=5,
        unique_items=True,
        additional_items=False
    )
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True,
        "additionalItems": False
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name"],
        min_properties=1,
        max_properties=3
    )
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 3
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
    const_field = Const(const="fixed", default="fixed")
    expected = {
        "const": "fixed",
        "default": "fixed"
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
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    expected = {
        "allOf": [
            {"type": "string", "minLength": 1},
            {"type": "string", "maxLength": 10}
        ]
    }
    assert to_json_schema(all_of_field) == expected

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


# LLM-generated content at query #42
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
    field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email")
    schema = to_json_schema(field)
    assert schema == {
        "type": "string",
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }

    # Test Integer field
    field = Integer(minimum=0, maximum=100, exclusive_minimum=True, exclusive_maximum=True)
    schema = to_json_schema(field)
    assert schema == {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "exclusiveMinimum": True,
        "exclusiveMaximum": True
    }

    # Test Float field
    field = Float(minimum=0.0, maximum=1.0, multiple_of=0.1)
    schema = to_json_schema(field)
    assert schema == {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "multipleOf": 0.1
    }

    # Test Boolean field
    field = Boolean()
    schema = to_json_schema(field)
    assert schema == {"type": "boolean"}

    # Test Array field
    field = Array(items=String(), min_items=1, max_items=10, unique_items=True)
    schema = to_json_schema(field)
    assert schema == {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": True
    }

    # Test Object field
    field = Object(
        properties={"name": String()},
        required=["name"],
        min_properties=1,
        max_properties=10
    )
    schema = to_json_schema(field)
    assert schema == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 10
    }

    # Test Choice field
    field = Choice(choices=[("a", "a"), ("b", "b")])
    schema = to_json_schema(field)
    assert schema == {"enum": ["a", "b"]}

    # Test Const field
    field = Const(const="fixed_value")
    schema = to_json_schema(field)
    assert schema == {"const": "fixed_value"}

    # Test Union field
    field = Union(any_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert schema == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

    # Test OneOf field
    field = OneOf(one_of=[String(), Integer()])
    schema = to_json_schema(field)
    assert schema == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

    # Test AllOf field
    field = AllOf(all_of=[String(), Const(const="test")])
    schema = to_json_schema(field)
    assert schema == {"allOf": [{"type": "string"}, {"const": "test"}]}

    # Test Not field
    field = Not(negated=String())
    schema = to_json_schema(field)
    assert schema == {"not": {"type": "string"}}

    # Test IfThenElse field
    field = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    schema = to_json_schema(field)
    assert schema == {
        "if": {"type": "string"},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }

    # Test Reference field
    definitions = Definitions()
    field = Reference(to="Test", definitions=definitions)
    schema = to_json_schema(field)
    assert schema == {"$ref": "#/components/schemas/Test"}

    # Test Schema field
    field = Schema(fields={"name": String()}, required=["name"])
    schema = to_json_schema(field)
    assert schema == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }

    # Test with definitions
    definitions = Definitions()
    definitions["Test"] = String()
    schema = to_json_schema(definitions)
    assert schema == {
        "components": {
            "schemas": {
                "Test": {"type": "string"}
            }
        }
    }


# LLM-generated content at query #43
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
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=2)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 2
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(allow_null=True, exclusive_minimum=0, exclusive_maximum=1)
    expected = {
        "type": ["number", "null"],
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 1
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean(allow_null=False)
    expected = {"type": "boolean"}
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(allow_null=True, min_items=1, max_items=5, items=String(), additional_items=False)
    expected = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"},
        "additionalItems": False,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(
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
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected


# LLM-generated content at query #44
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
    all_of_field = AllOf(all_of=[String(), Const(const="test")])
    expected = {
        "allOf": [{"type": "string"}, {"const": "test"}]
    }
    assert to_json_schema(all_of_field) == expected

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


# LLM-generated content at query #45
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
    array_field = Array(items=String(), min_items=1, max_items=5)
    expected = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 5
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(properties={"name": String(), "age": Integer()})
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
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

    # Test IfThenElse field
    if_field = IfThenElse(if_clause=String(), then_clause=Integer())
    expected = {
        "if": {"type": "string"},
        "then": {"type": "integer"}
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


# LLM-generated content at query #46
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(min_length=1, max_length=10, pattern="[a-z]+", format="email", allow_null=True)
    expected = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected

    # Test Integer field
    int_field = Integer(minimum=0, maximum=100, allow_null=True)
    expected = {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": 100
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(multiple_of=0.5, allow_null=True)
    expected = {
        "type": ["number", "null"],
        "multipleOf": 0.5
    }
    assert to_json_schema(float_field) == expected

    # Test Boolean field
    bool_field = Boolean(allow_null=True)
    expected = {
        "type": ["boolean", "null"]
    }
    assert to_json_schema(bool_field) == expected

    # Test Array field
    array_field = Array(items=String(), min_items=1, max_items=10, allow_null=True)
    expected = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 10,
        "items": {"type": "string"}
    }
    assert to_json_schema(array_field) == expected

    # Test Object field
    object_field = Object(properties={"name": String()}, required=["name"], allow_null=True)
    expected = {
        "type": ["object", "null"],
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
    if_field = String()
    then_field = Integer()
    else_field = Boolean()
    if_then_else_field = IfThenElse(if_clause=if_field, then_clause=then_field, else_clause=else_field)
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
    ref_field = Reference(to="test", definitions=definitions)
    expected = {
        "$ref": "#/components/schemas/test",
        "components": {"schemas": {}}
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


# LLM-generated content at query #47
#--------------------------

```python
def test_to_json_schema():
    # Test Any field
    assert to_json_schema(Any()) == True

    # Test NeverMatch field
    assert to_json_schema(NeverMatch()) == False

    # Test String field
    string_field = String(allow_null=True, min_length=1, max_length=10, pattern="[a-z]+", format="email")
    expected_string_schema = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 10,
        "pattern": "[a-z]+",
        "format": "email"
    }
    assert to_json_schema(string_field) == expected_string_schema

    # Test Integer field
    integer_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=2)
    expected_integer_schema = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 2
    }
    assert to_json_schema(integer_field) == expected_integer_schema

    # Test Float field
    float_field = Float(allow_null=True, exclusive_minimum=0, exclusive_maximum=1)
    expected_float_schema = {
        "type": ["number", "null"],
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 1
    }
    assert to_json_schema(float_field) == expected_float_schema

    # Test Boolean field
    boolean_field = Boolean(allow_null=False)
    expected_boolean_schema = {
        "type": "boolean"
    }
    assert to_json_schema(boolean_field) == expected_boolean_schema

    # Test Array field
    array_field = Array(allow_null=True, min_items=1, max_items=5, items=String(), additional_items=False)
    expected_array_schema = {
        "type": ["array", "null"],
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string"},
        "additionalItems": False,
        "uniqueItems": True
    }
    assert to_json_schema(array_field) == expected_array_schema

    # Test Object field
    object_field = Object(
        allow_null=False,
        properties={"name": String()},
        additional_properties=False,
        required=["name"]
    )
    expected_object_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "additionalProperties": False,
        "required": ["name"]
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
    all_of_field = AllOf(all_of=[String(min_length=1), String(max_length=10)])
    expected_all_of_schema = {
        "allOf": [{"type": "string", "minLength": 1}, {"type": "string", "maxLength": 10}]
    }
    assert to_json_schema(all_of_field) == expected_all_of_schema

    # Test OneOf field
    one_of_field = OneOf(one_of=[String(), Integer()])
    expected_one_of_schema = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert to_json_schema(one_of_field) == expected_one_of_schema

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

    # Test Not field
    not_field = Not(negated=String())
    expected_not_schema = {
        "not": {"type": "string"}
    }
    assert to_json_schema(not_field) == expected_not_schema

    # Test Reference field
    definitions = Definitions({"Person": Object(properties={"name": String()})})
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

    # Test Schema field
    schema_field = Schema(fields={"name": String()}, required=["name"])
    expected_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    assert to_json_schema(schema_field) == expected_schema


# LLM-generated content at query #48
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
        "uniqueItems": True,
        "default": ["a", "b"]
    }
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items == True
    assert field.default == ["a", "b"]

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

    # Test allow_null parameter
    data = {"type": "string"}
    field = from_json_schema_type(data, "string", True, Definitions())
    assert field.allow_null == True


# LLM-generated content at query #49
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
    int_field = Integer(allow_null=False, minimum=0, maximum=100, multiple_of=2)
    expected = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "multipleOf": 2
    }
    assert to_json_schema(int_field) == expected

    # Test Float field
    float_field = Float(allow_null=True, exclusive_minimum=0, exclusive_maximum=1)
    expected = {
        "type": ["number", "null"],
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 1
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
        required=["name"]
    )
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "additionalProperties": False,
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

    # Test IfThenElse field
    if_field = IfThenElse(
        if_clause=String(min_length=1),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    expected = {
        "if": {"type": "string", "minLength": 1},
        "then": {"type": "integer"},
        "else": {"type": "boolean"}
    }
    assert to_json_schema(if_field) == expected

    # Test Not field
    not_field = Not(negated=String())
    expected = {"not": {"type": "string"}}
    assert to_json_schema(not_field) == expected


# LLM-generated content at query #50
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
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1

    # Test object type
    data = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
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


# LLM-generated content at query #51
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
    data = {"type": "invalid"}
    try:
        field = from_json_schema_type(data, "invalid", False, Definitions())
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid'"


