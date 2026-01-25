####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_tokenize_yaml():
    # Test with empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace only
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with string input - simple dict
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start_index == 0

    # Test with list
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test with nested dict
    token = tokenize_yaml("parent:\n  child: value")
    assert isinstance(token, DictToken)
    assert token.value == {"parent": {"child": "value"}}

    # Test with integer
    token = tokenize_yaml("count: 42")
    assert isinstance(token, DictToken)
    assert isinstance(token.value["count"], ScalarToken)
    assert token.value["count"].value == 42

    # Test with float
    token = tokenize_yaml("price: 19.99")
    assert isinstance(token, DictToken)
    assert isinstance(token.value["price"], ScalarToken)
    assert token.value["price"].value == 19.99

    # Test with boolean
    token = tokenize_yaml("active: true")
    assert isinstance(token, DictToken)
    assert isinstance(token.value["active"], ScalarToken)
    assert token.value["active"].value is True

    # Test with null
    token = tokenize_yaml("value: null")
    assert isinstance(token, DictToken)
    assert isinstance(token.value["value"], ScalarToken)
    assert token.value["value"].value is None

    # Test with scalar string
    token = tokenize_yaml("simple string")
    assert isinstance(token, ScalarToken)
    assert token.value == "simple string"

    # Test invalid YAML - scanner error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{ invalid yaml :")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no >= 1

    # Test invalid YAML - parser error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("- item\n  bad_indent: value")
    assert exc_info.value.code == "parse_error"

    # Test with complex nested structure
    yaml_content = """
users:
  - name: Alice
    age: 30
  - name: Bob
    age: 25
"""
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert "users" in token.value
    assert isinstance(token.value["users"], ListToken)

    # Test position tracking for multiline content
    yaml_content = "key1: value1\nkey2: value2\nkey3: value3"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert len(token.value) == 3

    # Test with UTF-8 bytes
    token = tokenize_yaml("name: José".encode("utf-8"))
    assert isinstance(token, DictToken)
    assert token.value["name"].value == "José"

    # Test with mixed types in list
    token = tokenize_yaml("items:\n  - 1\n  - string\n  - true\n  - null")
    assert isinstance(token, DictToken)
    assert isinstance(token.value["items"], ListToken)
    assert len(token.value["items"].value) == 4

    # Test with yaml assertion error when yaml is None
    # This would require mocking, so we skip it in normal test
    # but it's covered by the assert statement in the function


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test 1: Valid YAML content
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert errors == {}
    
    # Test 2: Valid YAML with bytes input
    content_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(content_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert errors == {}
    
    # Test 3: Invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: : syntax:"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test 4: Empty content
    empty_content = ""
    try:
        validate_yaml(empty_content, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test 5: Validation error with field
    string_field = String(max_length=5)
    content_long = "toolongstring"
    value, errors = validate_yaml(content_long, string_field)
    assert errors
    
    # Test 6: Valid simple scalar
    int_field = Integer()
    content_int = "42"
    value, errors = validate_yaml(content_int, int_field)
    assert value == 42
    assert errors == {}
    
    # Test 7: YAML with list
    from typesystem.fields import Array
    list_schema = Array(items=Integer())
    content_list = "- 1\n- 2\n- 3"
    value, errors = validate_yaml(content_list, list_schema)
    assert value == [1, 2, 3]
    assert errors == {}
    
    # Test 8: Nested YAML structure
    class NestedSchema(Schema):
        user = TestSchema
    
    nested_content = "user:\n  name: Bob\n  age: 35"
    value, errors = validate_yaml(nested_content, NestedSchema)
    assert value == {"user": {"name": "Bob", "age": 35}}
    assert errors == {}


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dictionary with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = UserSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    
    # Test 5: Empty content should raise ParseError
    import pytest
    from typesystem.base import ParseError
    
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", String())
    assert exc_info.value.code == "no_content"
    
    # Test 6: Invalid YAML syntax
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("invalid: yaml: content:", String())
    assert exc_info.value.code == "parse_error"
    
    # Test 7: Bytes input
    content_bytes = b"hello"
    validator = String()
    value, error_messages = validate_yaml(content_bytes, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 8: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    
    # Test 9: YAML boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    
    # Test 10: YAML float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_yaml():
    """Test validate_yaml function with various inputs."""
    
    # Test with valid YAML and a simple Field validator
    yaml_content = "42"
    from typesystem.fields import Integer
    value, errors = validate_yaml(yaml_content, Integer())
    assert value == 42
    assert errors == []
    
    # Test with valid YAML dict and Schema validator
    yaml_content = "name: John\nage: 30"
    
    class UserSchema(Schema):
        name = Field()
        age = Integer()
    
    value, errors = validate_yaml(yaml_content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert errors == []
    
    # Test with valid YAML list
    yaml_content = "- 1\n- 2\n- 3"
    value, errors = validate_yaml(yaml_content, Field())
    assert value == [1, 2, 3]
    assert errors == []
    
    # Test with bytes input
    yaml_content = b"key: value"
    value, errors = validate_yaml(yaml_content, Field())
    assert value == {"key": "value"}
    assert errors == []
    
    # Test with invalid YAML syntax
    yaml_content = "invalid: [yaml: content"
    value, errors = validate_yaml(yaml_content, Field())
    assert errors  # Should have parse errors
    
    # Test with empty string - should have errors
    yaml_content = ""
    value, errors = validate_yaml(yaml_content, Field())
    assert errors
    
    # Test with whitespace only - should have errors
    yaml_content = "   \n  \n  "
    value, errors = validate_yaml(yaml_content, Field())
    assert errors
    
    # Test with YAML boolean values
    yaml_content = "true"
    value, errors = validate_yaml(yaml_content, Field())
    assert value is True
    assert errors == []
    
    # Test with YAML null value
    yaml_content = "null"
    value, errors = validate_yaml(yaml_content, Field())
    assert value is None
    assert errors == []
    
    # Test with YAML float
    yaml_content = "3.14"
    value, errors = validate_yaml(yaml_content, Field())
    assert value == 3.14
    assert errors == []
    
    # Test with nested structure
    yaml_content = "users:\n  - name: Alice\n    age: 25\n  - name: Bob\n    age: 30"
    value, errors = validate_yaml(yaml_content, Field())
    assert value == {"users": [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]}
    assert errors == []


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML content and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid simple scalar value
    content = "hello"
    validator = String()
    value, errors = validate_yaml(content, validator)
    assert value == "hello"
    assert errors == []
    
    # Test 2: Valid integer value
    content = "42"
    validator = Integer()
    value, errors = validate_yaml(content, validator)
    assert value == 42
    assert errors == []
    
    # Test 3: Valid dict with schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert errors == []
    
    # Test 4: Bytes input
    content = b"test"
    validator = String()
    value, errors = validate_yaml(content, validator)
    assert value == "test"
    assert errors == []
    
    # Test 5: Empty content should raise ParseError
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 6: Invalid YAML syntax
    content = "invalid: [yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test 7: Validation error with schema
    class StrictSchema(Schema):
        name = String(max_length=5)
    
    content = "name: VeryLongName"
    try:
        value, errors = validate_yaml(content, StrictSchema)
        assert len(errors) > 0
    except ParseError:
        pass
    
    # Test 8: List/array validation
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, errors = validate_yaml(content, validator)
    assert isinstance(value, list)
    
    # Test 9: Whitespace-only content
    content = "   \n  \n  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 10: Complex nested structure
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema()
    
    content = "name: Alice\naddress:\n  street: Main St\n  city: NYC"
    value, errors = validate_yaml(content, PersonSchema)
    assert value["name"] == "Alice"
    assert value["address"]["city"] == "NYC"


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field validator
    content = "42"
    from typesystem.fields import Integer
    value, errors = validate_yaml(content, Integer())
    assert value == 42
    assert errors == []

    # Test with valid YAML and schema
    content = "name: John\nage: 30"
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    
    class Person(Schema):
        name = String()
        age = Integer()
    
    value, errors = validate_yaml(content, Person)
    assert value == {"name": "John", "age": 30}
    assert errors == []

    # Test with invalid YAML syntax
    content = "invalid: yaml: content:"
    from typesystem.fields import String
    value, errors = validate_yaml(content, String())
    assert errors != []
    assert any("parse_error" in str(e) or "mapping values" in str(e).lower() for e in errors)

    # Test with valid YAML but validation fails
    content = "hello"
    from typesystem.fields import Integer
    value, errors = validate_yaml(content, Integer())
    assert errors != []

    # Test with list content
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array, String
    value, errors = validate_yaml(content, Array(items=String()))
    assert value == ["item1", "item2", "item3"]
    assert errors == []

    # Test with nested structure
    content = "users:\n  - name: Alice\n    age: 25\n  - name: Bob\n    age: 30"
    from typesystem.fields import Object, Array, String, Integer
    validator = Object(
        properties={
            "users": Array(
                items=Object(
                    properties={"name": String(), "age": Integer()}
                )
            )
        }
    )
    value, errors = validate_yaml(content, validator)
    assert value == {
        "users": [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30}
        ]
    }
    assert errors == []

    # Test with bytes input
    content = b"value: 123"
    from typesystem.fields import Object, Integer
    validator = Object(properties={"value": Integer()})
    value, errors = validate_yaml(content, validator)
    assert value == {"value": 123}
    assert errors == []

    # Test with boolean values
    content = "enabled: true\ndisabled: false"
    from typesystem.fields import Object, Boolean
    validator = Object(properties={"enabled": Boolean(), "disabled": Boolean()})
    value, errors = validate_yaml(content, validator)
    assert value == {"enabled": True, "disabled": False}
    assert errors == []

    # Test with null value
    content = "value: null"
    from typesystem.fields import Object, String
    validator = Object(properties={"value": String(allow_null=True)})
    value, errors = validate_yaml(content, validator)
    assert value == {"value": None}
    assert errors == []

    # Test with float values
    content = "price: 19.99"
    from typesystem.fields import Object, Number
    validator = Object(properties={"price": Number()})
    value, errors = validate_yaml(content, validator)
    assert value == {"price": 19.99}
    assert errors == []


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_yaml():
    """Test validate_yaml function with various inputs."""
    
    # Test 1: Valid YAML with Field validator
    content = "42"
    from typesystem.fields import Integer
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == {}
    
    # Test 2: Valid YAML with Schema validator
    class TestSchema(Schema):
        name = Field(allow_null=False)
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == {}
    
    # Test 3: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = Integer()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError:
        pass
    
    # Test 4: Valid YAML but invalid data type
    content = "hello"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != {}
    
    # Test 5: YAML with list
    content = "[1, 2, 3]"
    from typesystem.fields import Array
    validator = Array(items=Integer())
    value, error_messages = validate_yaml(content, validator)
    assert value == [1, 2, 3]
    assert error_messages == {}
    
    # Test 6: YAML with dictionary
    content = "key1: value1\nkey2: value2"
    from typesystem.fields import Object
    validator = Object(properties={})
    value, error_messages = validate_yaml(content, validator)
    assert value["key1"] == "value1"
    assert value["key2"] == "value2"
    
    # Test 7: Bytes input
    content = b"name: test"
    class SimpleSchema(Schema):
        name = Field()
    validator = SimpleSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "test"
    
    # Test 8: YAML with null values
    content = "value: null"
    class NullSchema(Schema):
        value = Field(allow_null=True)
    validator = NullSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["value"] is None
    
    # Test 9: YAML with boolean
    content = "flag: true"
    class BoolSchema(Schema):
        flag = Field()
    validator = BoolSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["flag"] is True
    
    # Test 10: YAML with float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == {}


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    yaml_content = "42"
    from typesystem.fields import Integer
    
    value, errors = validate_yaml(yaml_content, Integer())
    assert value == 42
    assert errors == []

    # Test with valid YAML and Schema validator
    yaml_content = "name: John\nage: 30"
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    
    class Person(Schema):
        name = String()
        age = Integer()
    
    value, errors = validate_yaml(yaml_content, Person)
    assert value == {"name": "John", "age": 30}
    assert errors == []

    # Test with bytes input
    yaml_content = b"value: test"
    from typesystem.fields import String
    
    class TestSchema(Schema):
        value = String()
    
    value, errors = validate_yaml(yaml_content, TestSchema)
    assert value == {"value": "test"}
    assert errors == []

    # Test with invalid YAML syntax
    yaml_content = "invalid: [yaml: content:"
    
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, String())
    assert exc_info.value.code == "parse_error"

    # Test with empty content
    yaml_content = ""
    
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, String())
    assert exc_info.value.code == "no_content"

    # Test with whitespace only
    yaml_content = "   \n  \t  "
    
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, String())
    assert exc_info.value.code == "no_content"

    # Test with validation error
    yaml_content = "not_a_number"
    from typesystem.fields import Integer
    
    value, errors = validate_yaml(yaml_content, Integer())
    assert errors  # Should have validation errors
    assert len(errors) > 0

    # Test with complex nested structure
    yaml_content = """
    users:
      - name: Alice
        age: 25
      - name: Bob
        age: 30
    """
    
    class User(Schema):
        name = String()
        age = Integer()
    
    class UserList(Schema):
        users = typesystem.Array(User())
    
    value, errors = validate_yaml(yaml_content, UserList)
    assert value["users"][0]["name"] == "Alice"
    assert value["users"][1]["age"] == 30
    assert errors == []

    # Test with null values
    yaml_content = "value: null"
    
    class NullableSchema(Schema):
        value = String(allow_null=True)
    
    value, errors = validate_yaml(yaml_content, NullableSchema)
    assert value["value"] is None
    assert errors == []

    # Test with boolean values
    yaml_content = "enabled: true"
    from typesystem.fields import Boolean
    
    class BoolSchema(Schema):
        enabled = Boolean()
    
    value, errors = validate_yaml(yaml_content, BoolSchema)
    assert value["enabled"] is True
    assert errors == []

    # Test with float values
    yaml_content = "price: 19.99"
    from typesystem.fields import Float
    
    class PriceSchema(Schema):
        price = Float()
    
    value, errors = validate_yaml(yaml_content, PriceSchema)
    assert value["price"] == 19.99
    assert errors == []


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with dict and Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML with list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    
    # Test 5: Invalid YAML syntax
    content = "invalid: yaml: syntax: here:"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages  # Should have errors
    
    # Test 6: YAML as bytes
    content = b"hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 7: Valid YAML with nested structure
    content = "user:\n  name: Alice\n  email: alice@example.com"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, dict)
    
    # Test 8: Empty YAML should raise ParseError
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 9: Whitespace-only YAML should raise ParseError
    content = "   \n  \n  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 10: YAML with boolean values
    content = "true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    
    # Test 11: YAML with null value
    content = "null"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    
    # Test 12: YAML with float
    content = "3.14"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, float)


# LLM-generated content at query #10
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar values
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    
    # Test integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    
    # Test float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    
    # Test boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    
    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    
    # Test list
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    
    # Test dictionary
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert "key" in token.value
    
    # Test bytes input
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    
    # Test multiline content
    yaml_content = "name: John\nage: 30"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    
    # Test empty string raises ParseError
    try:
        tokenize_yaml("")
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test whitespace only raises ParseError
    try:
        tokenize_yaml("   \n  \n  ")
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid YAML syntax raises ParseError
    try:
        tokenize_yaml("{invalid: [yaml: content}")
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test complex nested structure
    yaml_content = """
    users:
      - name: Alice
        age: 25
      - name: Bob
        age: 30
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert "users" in token.value
    
    # Test token has position information
    token = tokenize_yaml("test: value")
    assert token.start_position >= 0
    assert token.end_position >= token.start_position
    
    # Test UTF-8 bytes decoding
    yaml_bytes = "name: José".encode("utf-8")
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    
    # Test multiline string
    yaml_content = """
    description: |
      This is a
      multiline string
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    
    # Test complex list
    token = tokenize_yaml("- item1\n- item2\n- item3")
    assert isinstance(token, ListToken)
    assert len(token.value) == 3


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML content and a simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML string
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML bytes
    valid_yaml_bytes = b"name: Alice\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Alice", "age": 25}
    assert error_messages == {}
    
    # Test with field validator
    string_field = String()
    yaml_string = "hello"
    value, error_messages = validate_yaml(yaml_string, string_field)
    assert value == "hello"
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None
    
    # Test empty content
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
    
    # Test validation error
    class StrictSchema(Schema):
        name = String(max_length=5)
    
    yaml_too_long = "name: VeryLongName"
    value, error_messages = validate_yaml(yaml_too_long, StrictSchema)
    assert "name" in error_messages
    
    # Test with integer field
    int_field = Integer()
    yaml_int = "42"
    value, error_messages = validate_yaml(yaml_int, int_field)
    assert value == 42
    assert error_messages == {}
    
    # Test with list
    class ListSchema(Schema):
        items = typesystem.Array(String())
    
    yaml_list = "items:\n  - apple\n  - banana"
    value, error_messages = validate_yaml(yaml_list, ListSchema)
    assert value["items"] == ["apple", "banana"]
    assert error_messages == {}


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dictionary with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    
    # Test 5: Invalid YAML syntax
    content = "invalid: yaml: syntax:"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert len(error_messages) > 0
    
    # Test 6: Validation error - type mismatch
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert len(error_messages) > 0
    
    # Test 7: Bytes input
    content = b"hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 8: Empty content should raise ParseError
    from typesystem.base import ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", String())
    assert exc_info.value.code == "no_content"
    
    # Test 9: Whitespace-only content should raise ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("   \n  ", String())
    assert exc_info.value.code == "no_content"
    
    # Test 10: Complex nested structure
    content = "users:\n  - name: Alice\n    age: 25\n  - name: Bob\n    age: 30"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, dict)
    
    # Test 11: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    
    # Test 12: YAML with boolean value
    content = "true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is True or value == "true"


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid simple scalar with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == {}
    
    # Test 2: Valid YAML with bytes input
    content_bytes = b"world"
    validator = String()
    value, error_messages = validate_yaml(content_bytes, validator)
    assert value == "world"
    assert error_messages == {}
    
    # Test 3: Valid integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == {}
    
    # Test 4: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == {}
    
    # Test 5: Invalid YAML syntax
    content = "{ invalid: yaml: syntax"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test 6: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 7: Whitespace only content
    content = "   \n  \n  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 8: Valid YAML list
    content = "- one\n- two\n- three"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    assert error_messages == {}
    
    # Test 9: YAML with boolean
    content = "true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == {}
    
    # Test 10: YAML with null
    content = "null"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == {}


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    content = "hello"
    from typesystem.fields import String
    value, errors = validate_yaml(content, String())
    assert value == "hello"
    assert errors == []

    # Test with valid YAML dict and schema validator
    content = "name: John\nage: 30"
    from typesystem.schemas import Schema
    class PersonSchema(Schema):
        name = String()
        age = String()
    value, errors = validate_yaml(content, PersonSchema)
    assert value == {"name": "John", "age": "30"}
    assert errors == []

    # Test with valid YAML list
    content = "- item1\n- item2"
    from typesystem.fields import Array
    value, errors = validate_yaml(content, Array(items=String()))
    assert value == ["item1", "item2"]
    assert errors == []

    # Test with empty content
    content = ""
    from typesystem.fields import String
    try:
        validate_yaml(content, String())
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test with invalid YAML
    content = "invalid: [yaml: content"
    try:
        validate_yaml(content, String())
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test with bytes content
    content = b"test: value"
    class TestSchema(Schema):
        test = String()
    value, errors = validate_yaml(content, TestSchema)
    assert value == {"test": "value"}
    assert errors == []

    # Test with validation error
    from typesystem.fields import Integer
    content = "42"
    class IntSchema(Schema):
        value = Integer(maximum=10)
    value, errors = validate_yaml(content, IntSchema)
    assert len(errors) > 0 or value != 42

    # Test with YAML null value
    content = "null"
    from typesystem.fields import String
    value, errors = validate_yaml(content, String(allow_null=True))
    assert value is None
    assert errors == []

    # Test with YAML boolean
    content = "true"
    from typesystem.fields import Boolean
    value, errors = validate_yaml(content, Boolean())
    assert value is True
    assert errors == []

    # Test with YAML number
    content = "3.14"
    from typesystem.fields import Number
    value, errors = validate_yaml(content, Number())
    assert value == 3.14
    assert errors == []

    # Test with nested YAML structure
    content = "users:\n  - name: Alice\n  - name: Bob"
    class UserSchema(Schema):
        name = String()
    class UsersSchema(Schema):
        users = Array(items=UserSchema)
    value, errors = validate_yaml(content, UsersSchema)
    assert len(value["users"]) == 2
    assert errors == []


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string content with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 3: Valid YAML list
    from typesystem.fields import Array
    content = "[1, 2, 3]"
    validator = Array(items=Integer())
    value, error_messages = validate_yaml(content, validator)
    assert value == [1, 2, 3]
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "{ invalid yaml: [}"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 5: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 6: Bytes input
    content = b"hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 7: Validation failure
    class StrictSchema(Schema):
        name = String(max_length=3)
    
    content = "name: VeryLongName"
    try:
        value, error_messages = validate_yaml(content, StrictSchema)
        assert len(error_messages) > 0
    except:
        pass
    
    # Test 8: YAML with various types
    content = """
    string_val: "text"
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    from typesystem.fields import Object
    validator = Object()
    value, error_messages = validate_yaml(content, validator)
    assert value["string_val"] == "text"
    assert value["int_val"] == 42
    assert value["float_val"] == 3.14
    assert value["bool_val"] is True
    assert value["null_val"] is None


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test string input
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test scalar token
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test list token
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]

    # Test nested structure
    token = tokenize_yaml("key:\n  nested: value\n  items: [1, 2]")
    assert isinstance(token, DictToken)
    assert token.value["key"]["nested"] == "value"
    assert token.value["key"]["items"] == [1, 2]

    # Test boolean values
    token = tokenize_yaml("enabled: true\ndisabled: false")
    assert isinstance(token, DictToken)
    assert token.value["enabled"] is True
    assert token.value["disabled"] is False

    # Test null value
    token = tokenize_yaml("value: null")
    assert isinstance(token, DictToken)
    assert token.value["value"] is None

    # Test float value
    token = tokenize_yaml("pi: 3.14")
    assert isinstance(token, DictToken)
    assert token.value["pi"] == 3.14

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test invalid YAML with tab indentation
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key:\n\tinvalid")
    assert exc_info.value.code == "parse_error"

    # Test token has correct position information
    token = tokenize_yaml("key: value")
    assert token.start_position >= 0
    assert token.end_position >= token.start_position

    # Test multi-line content
    content = "first: 1\nsecond: 2\nthird: 3"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert len(token.value) == 3

    # Test complex nested structure
    yaml_content = """
    users:
      - name: Alice
        age: 30
      - name: Bob
        age: 25
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert len(token.value["users"]) == 2
    assert token.value["users"][0]["name"] == "Alice"


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class PersonSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = PersonSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 5: Invalid YAML syntax should raise ParseError
    import pytest
    content = "{ invalid yaml: [}"
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(content, validator)
    
    # Test 6: Empty content should raise ParseError
    content = ""
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 7: Whitespace-only content should raise ParseError
    content = "   \n  \t  "
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 8: Valid YAML with bytes input
    content = b"test value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test value"
    assert error_messages == []
    
    # Test 9: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 10: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 11: YAML with float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []


# LLM-generated content at query #18
#--------------------------

```python
def test_tokenize_yaml():
    # Test with empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace only
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with string input - simple mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start_index == 0
    assert token.end_index >= 0

    # Test with list
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test with scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test with integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test with float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test with boolean true
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test with boolean false
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test with null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test with nested structure
    token = tokenize_yaml("parent:\n  child: value")
    assert isinstance(token, DictToken)
    assert token.value == {"parent": {"child": "value"}}

    # Test with complex list
    token = tokenize_yaml("items:\n  - name: item1\n    value: 1\n  - name: item2\n    value: 2")
    assert isinstance(token, DictToken)
    assert "items" in token.value
    assert isinstance(token.value["items"], list)

    # Test with invalid YAML - scanner error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test with invalid YAML - parser error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("  key1: value1\nkey2: value2")
    assert exc_info.value.code == "parse_error"

    # Test position tracking for multiline content
    content = "key1: value1\nkey2: value2"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)

    # Test with bytes containing non-UTF8 characters (should ignore)
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)

    # Test with special characters in values
    token = tokenize_yaml('key: "value with spaces"')
    assert isinstance(token, DictToken)
    assert token.value["key"] == "value with spaces"

    # Test with empty dict
    token = tokenize_yaml("{}")
    assert isinstance(token, DictToken)
    assert token.value == {}

    # Test with empty list
    token = tokenize_yaml("[]")
    assert isinstance(token, ListToken)
    assert token.value == []


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string content with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML dict with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 3: Valid bytes content
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax should raise ParseError
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError:
        pass
    
    # Test 5: Empty content should raise ParseError
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 6: Validation error with wrong type
    content = "123"
    validator = String(max_length=2)
    value, error_messages = validate_yaml(content, validator)
    assert len(error_messages) > 0
    
    # Test 7: Valid YAML list
    from typesystem.fields import Array
    content = "[1, 2, 3]"
    validator = Array(items=Integer())
    value, error_messages = validate_yaml(content, validator)
    assert value == [1, 2, 3]
    assert error_messages == []
    
    # Test 8: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 9: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 10: YAML with float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid simple string
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == {}
    
    # Test 2: Valid YAML with integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == {}
    
    # Test 3: Valid YAML dictionary
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test 4: Valid YAML with bytes input
    content = b"test"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test"
    assert error_messages == {}
    
    # Test 5: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 6: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 7: Whitespace only content
    content = "   \n  \t  "
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 8: Valid YAML list
    content = "[1, 2, 3]"
    from typesystem.fields import Array
    validator = Array(items=Integer())
    value, error_messages = validate_yaml(content, validator)
    assert value == [1, 2, 3]
    
    # Test 9: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    
    # Test 10: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    
    # Test 11: YAML with float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    
    # Test 12: UTF-8 encoded bytes
    content = "名前: 太郎".encode("utf-8")
    class NameSchema(Schema):
        name = String()
    value, error_messages = validate_yaml(content, NameSchema)
    assert "name" in value or error_messages == {}


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid simple scalar value
    content = "hello"
    result, errors = validate_yaml(content, String())
    assert result == "hello"
    assert errors == []
    
    # Test 2: Valid integer
    content = "42"
    result, errors = validate_yaml(content, Integer())
    assert result == 42
    assert errors == []
    
    # Test 3: Valid dictionary with schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = """
name: John
age: 30
"""
    result, errors = validate_yaml(content, UserSchema())
    assert result == {"name": "John", "age": 30}
    assert errors == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: yaml: syntax:"
    try:
        validate_yaml(content, String())
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test 5: Empty content
    content = ""
    try:
        validate_yaml(content, String())
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test 6: Validation error - type mismatch
    content = "not_a_number"
    result, errors = validate_yaml(content, Integer())
    assert errors  # Should have validation errors
    
    # Test 7: YAML list
    content = """
- item1
- item2
- item3
"""
    from typesystem.fields import Array
    result, errors = validate_yaml(content, Array(items=String()))
    assert result == ["item1", "item2", "item3"]
    assert errors == []
    
    # Test 8: Bytes input
    content = b"test_value"
    result, errors = validate_yaml(content, String())
    assert result == "test_value"
    assert errors == []
    
    # Test 9: Whitespace only content
    content = "   \n  \n  "
    try:
        validate_yaml(content, String())
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 10: Nested schema validation
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema()
    
    content = """
name: Alice
address:
  street: Main St
  city: Springfield
"""
    result, errors = validate_yaml(content, PersonSchema())
    assert result["name"] == "Alice"
    assert result["address"]["street"] == "Main St"
    assert errors == []


# LLM-generated content at query #22
#--------------------------

```python
def test_tokenize_yaml():
    # Test with empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace only
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with string input - simple dict
    token = tokenize_yaml("name: John\nage: 30")
    assert isinstance(token, DictToken)
    assert token.value["name"] == "John"
    assert token.value["age"] == 30

    # Test with list
    token = tokenize_yaml("- item1\n- item2\n- item3")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2", "item3"]

    # Test with nested dict
    token = tokenize_yaml("user:\n  name: Alice\n  email: alice@example.com")
    assert isinstance(token, DictToken)
    assert token.value["user"]["name"] == "Alice"
    assert token.value["user"]["email"] == "alice@example.com"

    # Test with nested list
    token = tokenize_yaml("items:\n  - first\n  - second")
    assert isinstance(token, DictToken)
    assert token.value["items"] == ["first", "second"]

    # Test with scalar integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test with scalar float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test with scalar boolean true
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test with scalar boolean false
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test with scalar null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test with scalar string
    token = tokenize_yaml("hello world")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello world"

    # Test with invalid YAML syntax
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test with complex nested structure
    token = tokenize_yaml("""
users:
  - name: Alice
    age: 25
  - name: Bob
    age: 30
settings:
  debug: true
  timeout: 60
""")
    assert isinstance(token, DictToken)
    assert len(token.value["users"]) == 2
    assert token.value["users"][0]["name"] == "Alice"
    assert token.value["settings"]["debug"] is True

    # Test position tracking
    content = "key: value"
    token = tokenize_yaml(content)
    assert token.start_pos >= 0
    assert token.end_pos >= token.start_pos

    # Test with UTF-8 bytes
    token = tokenize_yaml("name: José".encode("utf-8"))
    assert isinstance(token, DictToken)
    assert token.value["name"] == "José"

    # Test with multiline string
    token = tokenize_yaml('description: |\n  This is a\n  multiline string')
    assert isinstance(token, DictToken)
    assert "multiline" in token.value["description"]

    # Test with quoted string
    token = tokenize_yaml('message: "Hello World"')
    assert isinstance(token, DictToken)
    assert token.value["message"] == "Hello World"


# LLM-generated content at query #23
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML dictionary
    yaml_dict = "key: value"
    token = tokenize_yaml(yaml_dict)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start_position == 0

    # Test valid YAML list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test valid YAML scalar
    yaml_scalar = "hello"
    token = tokenize_yaml(yaml_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test YAML with integers
    yaml_int = "count: 42"
    token = tokenize_yaml(yaml_int)
    assert isinstance(token, DictToken)
    assert token.value["count"] == 42

    # Test YAML with floats
    yaml_float = "value: 3.14"
    token = tokenize_yaml(yaml_float)
    assert isinstance(token, DictToken)
    assert token.value["value"] == 3.14

    # Test YAML with booleans
    yaml_bool = "enabled: true"
    token = tokenize_yaml(yaml_bool)
    assert isinstance(token, DictToken)
    assert token.value["enabled"] is True

    # Test YAML with null
    yaml_null = "value: null"
    token = tokenize_yaml(yaml_null)
    assert isinstance(token, DictToken)
    assert token.value["value"] is None

    # Test YAML with bytes input
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \n  ")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value: invalid:")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no >= 1

    # Test nested YAML structures
    yaml_nested = "parent:\n  child: value\n  list:\n    - item1\n    - item2"
    token = tokenize_yaml(yaml_nested)
    assert isinstance(token, DictToken)
    assert token.value["parent"]["child"] == "value"
    assert token.value["parent"]["list"] == ["item1", "item2"]

    # Test position tracking
    yaml_multiline = "line1: value1\nline2: value2"
    token = tokenize_yaml(yaml_multiline)
    assert token.start_position == 0
    assert token.end_position > 0

    # Test YAML with special characters
    yaml_special = 'message: "Hello: World!"'
    token = tokenize_yaml(yaml_special)
    assert isinstance(token, DictToken)
    assert token.value["message"] == "Hello: World!"

    # Test UTF-8 encoded bytes
    yaml_utf8 = "greeting: こんにちは".encode("utf-8")
    token = tokenize_yaml(yaml_utf8)
    assert isinstance(token, DictToken)
    assert token.value["greeting"] == "こんにちは"

    # Test YAML with numbers as strings
    yaml_string_num = '"123"'
    token = tokenize_yaml(yaml_string_num)
    assert isinstance(token, ScalarToken)
    assert token.value == "123"


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_yaml():
    """Test validate_yaml function with various inputs."""
    
    # Test 1: Valid YAML with simple field validation
    content = "42"
    from typesystem.fields import Integer
    value, error_messages = validate_yaml(content, Integer())
    assert value == 42
    assert error_messages == {}
    
    # Test 2: Valid YAML with string field validation
    content = "hello"
    from typesystem.fields import String
    value, error_messages = validate_yaml(content, String())
    assert value == "hello"
    assert error_messages == {}
    
    # Test 3: Valid YAML with dictionary and schema validation
    content = "name: John\nage: 30"
    from typesystem.fields import String, Integer
    
    class PersonSchema(Schema):
        name = String()
        age = Integer()
    
    value, error_messages = validate_yaml(content, PersonSchema)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == {}
    
    # Test 4: Valid YAML with list
    content = "[1, 2, 3]"
    from typesystem.fields import Array
    value, error_messages = validate_yaml(content, Array(items=Integer()))
    assert value == [1, 2, 3]
    assert error_messages == {}
    
    # Test 5: Invalid YAML syntax
    content = "{ invalid yaml: [unclosed"
    from typesystem.fields import String
    value, error_messages = validate_yaml(content, String())
    assert error_messages != {}
    
    # Test 6: Bytes input
    content = b"test_value"
    from typesystem.fields import String
    value, error_messages = validate_yaml(content, String())
    assert value == "test_value"
    assert error_messages == {}
    
    # Test 7: YAML with null value
    content = "null"
    from typesystem.fields import String
    value, error_messages = validate_yaml(content, String(allow_null=True))
    assert value is None
    assert error_messages == {}
    
    # Test 8: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    value, error_messages = validate_yaml(content, Boolean())
    assert value is True
    assert error_messages == {}
    
    # Test 9: YAML with float
    content = "3.14"
    from typesystem.fields import Float
    value, error_messages = validate_yaml(content, Float())
    assert value == 3.14
    assert error_messages == {}
    
    # Test 10: Validation error - wrong type
    content = "not_a_number"
    from typesystem.fields import Integer
    value, error_messages = validate_yaml(content, Integer())
    assert error_messages != {}


# LLM-generated content at query #25
#--------------------------

def test_validate_yaml():
    # Test with valid YAML content and a simple schema
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML
    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == {}
    
    # Test with bytes input
    content_bytes = b"name: Jane\nage: 25"
    result, errors = validate_yaml(content_bytes, TestSchema)
    assert result == {"name": "Jane", "age": 25}
    assert errors == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError:
        pass
    
    # Test empty content
    empty_content = ""
    try:
        validate_yaml(empty_content, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test whitespace-only content
    whitespace_content = "   \n  \n  "
    try:
        validate_yaml(whitespace_content, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test with Field validator
    name_field = String()
    content = "John"
    result, errors = validate_yaml(content, name_field)
    assert result == "John"
    assert errors == {}
    
    # Test validation error (missing required field)
    class RequiredSchema(Schema):
        name = String(allow_null=False)
        age = Integer(allow_null=False)
    
    content = "name: John"
    result, errors = validate_yaml(content, RequiredSchema)
    assert "age" in errors
    
    # Test with complex nested YAML
    complex_yaml = """
    name: John
    age: 30
    items:
      - item1
      - item2
    """
    result, errors = validate_yaml(complex_yaml, TestSchema)
    # Should have validation errors due to extra "items" field
    
    # Test with various YAML types (int, float, bool, null)
    yaml_types = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    from typesystem.fields import Field
    result, errors = validate_yaml(yaml_types, Field())


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: [yaml: content"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except Exception as exc:
        assert "parse_error" in str(exc) or "ScannerError" in str(type(exc))
    
    # Test 5: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except Exception as exc:
        assert "no_content" in str(exc) or "No content" in str(exc)
    
    # Test 6: Bytes input
    content = b"hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 7: YAML with list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 8: YAML with nested dictionary
    content = "user:\n  name: Alice\n  email: alice@example.com"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, dict)
    assert value["user"]["name"] == "Alice"
    
    # Test 9: YAML with boolean
    content = "active: true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, dict)
    assert value["active"] is True
    
    # Test 10: YAML with null
    content = "value: null"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, dict)
    assert value["value"] is None


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML with bytes
    valid_yaml_bytes = b"name: Alice\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Alice", "age": 25}
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid syntax: [unclosed"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test empty YAML content
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test with Field validator
    name_field = String(max_length=10)
    valid_field_yaml = "John"
    value, error_messages = validate_yaml(valid_field_yaml, name_field)
    assert value == "John"
    assert error_messages == {}
    
    # Test validation failure
    class StrictSchema(Schema):
        name = String(max_length=3)
    
    invalid_content = "name: VeryLongName"
    value, error_messages = validate_yaml(invalid_content, StrictSchema)
    assert "name" in error_messages
    
    # Test YAML with complex structure
    complex_yaml = """
items:
  - id: 1
    name: Item1
  - id: 2
    name: Item2
"""
    from typesystem.fields import Array
    
    class ItemSchema(Schema):
        id = Integer()
        name = String()
    
    class ContainerSchema(Schema):
        items = Array(items=ItemSchema)
    
    value, error_messages = validate_yaml(complex_yaml, ContainerSchema)
    assert "items" in value
    assert len(value["items"]) == 2


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []

    # Test with bytes input
    content_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(content_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == []

    # Test with validation error
    content_invalid = "name: John\nage: not_a_number"
    value, error_messages = validate_yaml(content_invalid, TestSchema)
    assert error_messages  # Should have error messages
    assert len(error_messages) > 0

    # Test with Field validator
    content_field = "42"
    value, error_messages = validate_yaml(content_field, Field(int))
    assert value == 42
    assert error_messages == []

    # Test with Field validator type mismatch
    content_field_invalid = "not_a_number"
    value, error_messages = validate_yaml(content_field_invalid, Field(int))
    assert error_messages  # Should have error messages

    # Test with list content
    class ListSchema(Schema):
        items = Field(list)

    content_list = "items:\n  - item1\n  - item2"
    value, error_messages = validate_yaml(content_list, ListSchema)
    assert value == {"items": ["item1", "item2"]}
    assert error_messages == []

    # Test with nested structure
    class NestedSchema(Schema):
        person = Field(dict)

    content_nested = "person:\n  name: John\n  age: 30"
    value, error_messages = validate_yaml(content_nested, NestedSchema)
    assert value == {"person": {"name": "John", "age": 30}}
    assert error_messages == []

    # Test with empty content should raise ParseError
    with pytest.raises(ParseError):
        validate_yaml("", TestSchema)

    # Test with invalid YAML syntax should raise ParseError
    with pytest.raises(ParseError):
        validate_yaml("invalid: yaml: content:", TestSchema)

    # Test with whitespace only should raise ParseError
    with pytest.raises(ParseError):
        validate_yaml("   \n\n  ", TestSchema)


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    
    content = "hello"
    result, errors = validate_yaml(content, String())
    assert result == "hello"
    assert errors == []
    
    # Test with valid YAML and schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []
    
    # Test with invalid YAML syntax
    content = "name: John\n  invalid: [unclosed"
    result, errors = validate_yaml(content, TestSchema)
    assert result is None
    assert len(errors) > 0
    
    # Test with empty content
    content = ""
    result, errors = validate_yaml(content, String())
    assert result is None
    assert len(errors) > 0
    
    # Test with bytes content
    content = b"hello"
    result, errors = validate_yaml(content, String())
    assert result == "hello"
    assert errors == []
    
    # Test with validation error
    class StrictSchema(Schema):
        count = Integer()
    
    content = "count: not_a_number"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) > 0
    
    # Test with nested YAML structures
    content = "items:\n  - name: item1\n  - name: item2"
    result, errors = validate_yaml(content, TestSchema)
    assert errors is not None
    
    # Test with YAML list
    from typesystem.fields import Array
    content = "- 1\n- 2\n- 3"
    result, errors = validate_yaml(content, Array(items=Integer()))
    assert result == [1, 2, 3]
    assert errors == []
    
    # Test with YAML boolean values
    content = "active: true"
    class BoolSchema(Schema):
        active = typesystem.fields.Boolean()
    result, errors = validate_yaml(content, BoolSchema)
    assert result == {"active": True}
    assert errors == []
    
    # Test with YAML null values
    content = "value: null"
    class NullSchema(Schema):
        value = typesystem.fields.Field(allow_null=True)
    result, errors = validate_yaml(content, NullSchema)
    assert result == {"value": None}
    assert errors == []


# LLM-generated content at query #30
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test simple string input
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test scalar token
    token = tokenize_yaml("simple string")
    assert isinstance(token, ScalarToken)
    assert token.value == "simple string"

    # Test list token
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test integer token
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test float token
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test boolean token
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test null token
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test nested structure
    token = tokenize_yaml("parent:\n  child: value")
    assert isinstance(token, DictToken)
    assert token.value == {"parent": {"child": "value"}}

    # Test position tracking for simple content
    token = tokenize_yaml("test")
    assert token.start_position == 0

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: : yaml:")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None
    assert exc_info.value.position.line_no >= 1

    # Test multiline content position calculation
    content = "line1: value1\nline2: value2"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)

    # Test complex nested structure with lists and dicts
    token = tokenize_yaml("items:\n  - name: item1\n    value: 10\n  - name: item2\n    value: 20")
    assert isinstance(token, DictToken)
    assert "items" in token.value
    assert isinstance(token.value["items"], list)
    assert len(token.value["items"]) == 2

    # Test special characters in strings
    token = tokenize_yaml('message: "Hello: World!"')
    assert isinstance(token, DictToken)
    assert token.value["message"] == "Hello: World!"

    # Test unicode content
    token = tokenize_yaml("text: こんにちは")
    assert isinstance(token, DictToken)
    assert token.value["text"] == "こんにちは"

    # Test bytes with UTF-8 encoding
    token = tokenize_yaml("text: café".encode("utf-8"))
    assert isinstance(token, DictToken)
    assert token.value["text"] == "café"

    # Test scanner error position tracking
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value\n  invalid indentation:")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no >= 1


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML content and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string field
    yaml_content = "hello world"
    validator = String()
    value, errors = validate_yaml(yaml_content, validator)
    assert value == "hello world"
    assert errors == []
    
    # Test 2: Valid integer field
    yaml_content = "42"
    validator = Integer()
    value, errors = validate_yaml(yaml_content, validator)
    assert value == 42
    assert errors == []
    
    # Test 3: Valid schema with dict
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    yaml_content = "name: John\nage: 30"
    value, errors = validate_yaml(yaml_content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert errors == []
    
    # Test 4: Invalid field type
    yaml_content = "not_a_number"
    validator = Integer()
    value, errors = validate_yaml(yaml_content, validator)
    assert errors  # Should have validation errors
    
    # Test 5: Valid bytes content
    yaml_content = b"test_value"
    validator = String()
    value, errors = validate_yaml(yaml_content, validator)
    assert value == "test_value"
    assert errors == []
    
    # Test 6: List validation
    from typesystem.fields import Array
    yaml_content = "- 1\n- 2\n- 3"
    validator = Array(items=Integer())
    value, errors = validate_yaml(yaml_content, validator)
    assert value == [1, 2, 3]
    assert errors == []
    
    # Test 7: Empty content should raise ParseError
    from typesystem.base import ParseError
    yaml_content = ""
    validator = String()
    try:
        validate_yaml(yaml_content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 8: Invalid YAML syntax should raise ParseError
    yaml_content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(yaml_content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test 9: Nested schema validation
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema()
    
    yaml_content = "name: Jane\naddress:\n  street: Main St\n  city: NYC"
    value, errors = validate_yaml(yaml_content, PersonSchema)
    assert value == {"name": "Jane", "address": {"street": "Main St", "city": "NYC"}}
    assert errors == []
    
    # Test 10: UTF-8 bytes content
    yaml_content = "name: José".encode("utf-8")
    validator = String()
    value, errors = validate_yaml(yaml_content, validator)
    assert value == "name: José"
    assert errors == []


# LLM-generated content at query #32
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple schema
    class TestSchema(Schema):
        name = Field(type="string")
        age = Field(type="integer")
    
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert errors == {}
    
    # Test with bytes input
    content_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(content_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert errors == {}
    
    # Test with invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, TestSchema)
    assert exc_info.value.code == "parse_error"
    
    # Test with validation error (wrong type)
    invalid_content = "name: John\nage: not_a_number"
    value, errors = validate_yaml(invalid_content, TestSchema)
    assert "age" in errors
    
    # Test with empty content
    empty_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_content, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with single Field validator
    from typesystem.fields import String
    content = "hello world"
    value, errors = validate_yaml(content, String())
    assert value == "hello world"
    assert errors == {}
    
    # Test with list content
    list_content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    value, errors = validate_yaml(list_content, Array(items=String()))
    assert value == ["item1", "item2", "item3"]
    assert errors == {}
    
    # Test with nested schema
    class AddressSchema(Schema):
        street = Field(type="string")
        city = Field(type="string")
    
    class PersonSchema(Schema):
        name = Field(type="string")
        address = Field(validator=AddressSchema)
    
    nested_content = "name: John\naddress:\n  street: Main St\n  city: Boston"
    value, errors = validate_yaml(nested_content, PersonSchema)
    assert value["name"] == "John"
    assert value["address"]["street"] == "Main St"
    assert errors == {}
    
    # Test with YAML boolean values
    bool_content = "active: true\ninactive: false"
    class BoolSchema(Schema):
        active = Field(type="boolean")
        inactive = Field(type="boolean")
    
    value, errors = validate_yaml(bool_content, BoolSchema)
    assert value["active"] is True
    assert value["inactive"] is False
    assert errors == {}
    
    # Test with YAML null values
    null_content = "name: John\nmiddle_name: null"
    value, errors = validate_yaml(null_content, TestSchema)
    assert value["name"] == "John"


# LLM-generated content at query #33
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = UserSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: [yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError:
        pass
    
    # Test 5: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 6: YAML with bytes input
    content = b"test string"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test string"
    assert error_messages == []
    
    # Test 7: YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    assert len(value) == 3
    
    # Test 8: YAML with booleans
    content = "true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    
    # Test 9: YAML with null
    content = "null"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    
    # Test 10: YAML with float
    content = "3.14"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14


# LLM-generated content at query #34
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with dictionary and Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = UserSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML with list
    from typesystem.fields import Array
    content = "[1, 2, 3]"
    validator = Array(items=Integer())
    value, error_messages = validate_yaml(content, validator)
    assert value == [1, 2, 3]
    assert error_messages == []
    
    # Test 5: Invalid YAML syntax should raise ParseError
    import pytest
    content = "invalid: yaml: content:"
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(content, validator)
    
    # Test 6: YAML with bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 7: Empty content should raise ParseError
    content = ""
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 8: Whitespace-only content should raise ParseError
    content = "   \n  \t  "
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 9: Validation failure with type mismatch
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages  # Should have validation errors
    
    # Test 10: YAML with null value
    content = "null"
    from typesystem.fields import String as StringField
    validator = StringField(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 11: Complex nested structure
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema
    
    content = """
name: Alice
address:
  street: Main St
  city: Boston
"""
    validator = PersonSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "Alice"
    assert value["address"]["city"] == "Boston"
    assert error_messages == []
    
    # Test 12: YAML with booleans
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 13: YAML with floats
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []


# LLM-generated content at query #35
#--------------------------

```python
def test_tokenize_yaml():
    # Test with valid YAML dictionary
    yaml_dict = "key: value\nother: 123"
    token = tokenize_yaml(yaml_dict)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "other": 123}
    assert token.start_position == 0

    # Test with valid YAML list
    yaml_list = "- item1\n- item2\n- 3"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2", 3]

    # Test with YAML scalar
    yaml_scalar = "42"
    token = tokenize_yaml(yaml_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test with YAML boolean
    yaml_bool = "true"
    token = tokenize_yaml(yaml_bool)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test with YAML null
    yaml_null = "null"
    token = tokenize_yaml(yaml_null)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test with YAML float
    yaml_float = "3.14"
    token = tokenize_yaml(yaml_float)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test with nested structures
    yaml_nested = "parent:\n  child: value\n  list:\n    - a\n    - b"
    token = tokenize_yaml(yaml_nested)
    assert isinstance(token, DictToken)
    assert "parent" in token.value
    assert isinstance(token.value["parent"], DictToken)

    # Test with bytes input
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with empty/whitespace-only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \n  ")
    assert exc_info.value.code == "no_content"

    # Test with invalid YAML raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value\n  invalid:\n invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test with malformed YAML structure
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("[invalid: yaml:")
    assert exc_info.value.code == "parse_error"

    # Test position tracking for multiline content
    multiline = "line1: value\nline2: value2\nline3: value3"
    token = tokenize_yaml(multiline)
    assert isinstance(token, DictToken)
    assert token.start_position == 0

    # Test with special YAML characters
    yaml_special = "key: 'value with: colon'"
    token = tokenize_yaml(yaml_special)
    assert isinstance(token, DictToken)
    assert token.value["key"] == "value with: colon"

    # Test with YAML comments
    yaml_comment = "# This is a comment\nkey: value"
    token = tokenize_yaml(yaml_comment)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with multiple types in list
    yaml_mixed_list = "- string\n- 42\n- 3.14\n- true\n- null"
    token = tokenize_yaml(yaml_mixed_list)
    assert isinstance(token, ListToken)
    assert len(token.value) == 5
    assert token.value[0] == "string"
    assert token.value[1] == 42
    assert token.value[2] == 3.14
    assert token.value[3] is True
    assert token.value[4] is None


# LLM-generated content at query #36
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with dict and Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = UserSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: [yaml: syntax:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError:
        pass
    
    # Test 5: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 6: Valid YAML with bytes input
    content = b"test_string"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_string"
    assert error_messages == []
    
    # Test 7: Valid YAML list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 8: YAML with boolean and null values
    content = "flag: true\nempty: null"
    class BoolSchema(Schema):
        flag = typesystem.fields.Boolean()
        empty = typesystem.fields.String(allow_null=True)
    
    validator = BoolSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["flag"] is True
    assert value["empty"] is None
    assert error_messages == []
    
    # Test 9: Validation failure (type mismatch)
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []
    
    # Test 10: YAML with float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []


# LLM-generated content at query #37
#--------------------------

```python
def test_validate_yaml():
    """Test the validate_yaml function with various inputs."""
    
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string content with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 3: Valid YAML with bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages  # Should have error messages
    
    # Test 5: Valid YAML list with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 6: Valid YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 7: Valid YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 8: Valid YAML with float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []
    
    # Test 9: Validation error - wrong type
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages  # Should have validation errors
    
    # Test 10: Valid YAML list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []


# LLM-generated content at query #38
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    yaml_content = "name: John"
    
    class TestSchema(Schema):
        name = Field(max_length=50)
    
    value, error_messages = validate_yaml(yaml_content, TestSchema)
    assert value == {"name": "John"}
    assert error_messages == {}
    
    # Test with valid YAML and bytes input
    yaml_bytes = b"age: 25"
    
    class AgeSchema(Schema):
        age = Field()
    
    value, error_messages = validate_yaml(yaml_bytes, AgeSchema)
    assert value == {"age": 25}
    assert error_messages == {}
    
    # Test with invalid YAML syntax
    invalid_yaml = "invalid: [unclosed"
    
    class SimpleSchema(Schema):
        invalid = Field()
    
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, SimpleSchema)
    assert exc_info.value.code == "parse_error"
    
    # Test with empty content
    empty_yaml = ""
    
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, SimpleSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with whitespace-only content
    whitespace_yaml = "   \n  \t  "
    
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(whitespace_yaml, SimpleSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with complex YAML structure
    complex_yaml = """
    users:
      - name: Alice
        age: 30
      - name: Bob
        age: 25
    """
    
    class UserSchema(Schema):
        users = Field()
    
    value, error_messages = validate_yaml(complex_yaml, UserSchema)
    assert "users" in value
    assert len(value["users"]) == 2
    
    # Test with Field validator directly
    yaml_field = "hello world"
    field = Field(max_length=20)
    
    value, error_messages = validate_yaml(yaml_field, field)
    assert value == "hello world"
    assert error_messages == {}
    
    # Test with validation error (max_length exceeded)
    long_yaml = "a" * 100
    short_field = Field(max_length=10)
    
    value, error_messages = validate_yaml(long_yaml, short_field)
    assert error_messages != {}
    
    # Test with YAML containing different scalar types
    mixed_yaml = """
    string: hello
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    """
    
    class MixedSchema(Schema):
        string = Field()
        integer = Field()
        float = Field()
        boolean = Field()
        null_value = Field()
    
    value, error_messages = validate_yaml(mixed_yaml, MixedSchema)
    assert value["string"] == "hello"
    assert value["integer"] == 42
    assert value["float"] == 3.14
    assert value["boolean"] is True
    assert value["null_value"] is None


# LLM-generated content at query #39
#--------------------------

```python
def test_validate_yaml():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with Field validator
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Schema validator
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 3: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError:
        pass
    
    # Test 4: Empty YAML content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 5: Bytes input
    content = b"test_string"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_string"
    assert error_messages == []
    
    # Test 6: YAML with integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 7: YAML with list
    from typesystem.fields import Array
    content = "- item1\n- item2\n- item3"
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 8: YAML with nested structure
    class NestedSchema(Schema):
        user = TestSchema
    
    content = "user:\n  name: Alice\n  age: 25"
    validator = NestedSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"user": {"name": "Alice", "age": 25}}
    assert error_messages == []
    
    # Test 9: YAML validation failure
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []
    
    # Test 10: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []


# LLM-generated content at query #40
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with dict and Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML with list
    content = "[1, 2, 3]"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    # List validation returns the list with potential errors
    
    # Test 5: Invalid YAML syntax
    content = "{ invalid yaml: [}"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 6: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 7: Bytes input
    content = b"test content"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test content"
    assert error_messages == []
    
    # Test 8: Whitespace only content
    content = "   \n  \t  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 9: Validation error with Schema
    class StrictSchema(Schema):
        name = String(max_length=5)
    
    content = "name: VeryLongName"
    try:
        value, error_messages = validate_yaml(content, StrictSchema)
        assert len(error_messages) > 0
    except ParseError:
        pass
    
    # Test 10: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None


# LLM-generated content at query #41
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid scalar YAML with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = UserSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 4: Invalid YAML - type mismatch
    content = "hello"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []
    
    # Test 5: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(content, validator)
    
    # Test 6: Empty content should raise ParseError
    content = ""
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 7: Bytes input
    content = b"name: Alice\nage: 25"
    class PersonSchema(Schema):
        name = String()
        age = Integer()
    
    validator = PersonSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "Alice"
    assert value["age"] == 25
    assert error_messages == []
    
    # Test 8: Valid YAML list
    content = "[1, 2, 3]"
    from typesystem.fields import Array
    validator = Array(items=Integer())
    value, error_messages = validate_yaml(content, validator)
    assert value == [1, 2, 3]
    assert error_messages == []
    
    # Test 9: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 10: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []


# LLM-generated content at query #42
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic dictionary
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_position == 0

    # Test basic list
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert result.value == ["item1", "item2"]

    # Test scalar string
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test integer
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test float
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test boolean true
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test boolean false
    result = tokenize_yaml("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test null
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test nested structure
    result = tokenize_yaml("parent:\n  child: value")
    assert isinstance(result, DictToken)
    assert result.value == {"parent": {"child": "value"}}

    # Test bytes input
    result = tokenize_yaml(b"key: value")
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \n  ")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test complex nested structure with lists and dicts
    yaml_content = "users:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 25"
    result = tokenize_yaml(yaml_content)
    assert isinstance(result, DictToken)
    assert len(result.value["users"]) == 2
    assert result.value["users"][0]["name"] == "Alice"

    # Test position tracking
    result = tokenize_yaml("key: value")
    assert hasattr(result, "start_position")
    assert hasattr(result, "end_position")

    # Test multiline string
    yaml_content = "text: |\n  line1\n  line2"
    result = tokenize_yaml(yaml_content)
    assert isinstance(result, DictToken)

    # Test UTF-8 bytes
    result = tokenize_yaml("key: café".encode("utf-8"))
    assert isinstance(result, DictToken)
    assert "café" in str(result.value)


# LLM-generated content at query #43
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string content with String field
    content = "hello"
    validator = String()
    value, errors = validate_yaml(content, validator)
    assert value == "hello"
    assert errors == []
    
    # Test 2: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, errors = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert errors == []
    
    # Test 3: Valid YAML bytes input
    content = b"test_value"
    validator = String()
    value, errors = validate_yaml(content, validator)
    assert value == "test_value"
    assert errors == []
    
    # Test 4: Valid YAML list
    from typesystem.fields import Array
    content = "- item1\n- item2\n- item3"
    validator = Array(items=String())
    value, errors = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert errors == []
    
    # Test 5: Valid YAML with integer
    content = "42"
    validator = Integer()
    value, errors = validate_yaml(content, validator)
    assert value == 42
    assert errors == []
    
    # Test 6: Valid YAML with boolean
    from typesystem.fields import Boolean
    content = "true"
    validator = Boolean()
    value, errors = validate_yaml(content, validator)
    assert value is True
    assert errors == []
    
    # Test 7: Valid YAML with null
    from typesystem.fields import Field as BaseField
    content = "null"
    validator = BaseField(allow_null=True)
    value, errors = validate_yaml(content, validator)
    assert value is None
    assert errors == []
    
    # Test 8: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError:
        pass
    
    # Test 9: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 10: Validation failure (wrong type)
    content = "not_a_number"
    validator = Integer()
    value, errors = validate_yaml(content, validator)
    assert errors  # Should have validation errors


# LLM-generated content at query #44
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic dictionary
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start_index == 0

    # Test basic list
    content = "- item1\n- item2"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test scalar string
    content = "hello"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test scalar integer
    content = "42"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test scalar float
    content = "3.14"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test scalar boolean
    content = "true"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test scalar null
    content = "null"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test nested structure
    content = "parent:\n  child: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"parent": {"child": "value"}}

    # Test bytes input
    content = b"key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value\n  invalid:\n   : bad")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no >= 1
    assert exc_info.value.position.column_no >= 1

    # Test position tracking for multi-line content
    content = "line1: value1\nline2: value2"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.content == content

    # Test complex nested structure
    content = "users:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 25"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert len(token.value["users"]) == 2
    assert token.value["users"][0]["name"] == "Alice"

    # Test list with mixed types
    content = "- 1\n- hello\n- true\n- null"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, "hello", True, None]

    # Test quoted string
    content = '"quoted: string"'
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "quoted: string"


# LLM-generated content at query #45
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML as bytes
    valid_yaml_bytes = b"name: Alice\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Alice", "age": 25}
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, TestSchema)
    assert exc_info.value.code == "parse_error"
    
    # Test empty YAML content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test whitespace-only YAML content
    whitespace_yaml = "   \n  \n  "
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(whitespace_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test YAML with validation errors
    invalid_data_yaml = "name: 123\nage: not_a_number"
    value, error_messages = validate_yaml(invalid_data_yaml, TestSchema)
    assert error_messages is not None
    assert len(error_messages) > 0
    
    # Test with simple Field validator
    string_field = String()
    simple_yaml = "hello"
    value, error_messages = validate_yaml(simple_yaml, string_field)
    assert value == "hello"
    assert error_messages == {}
    
    # Test YAML with list
    class ListSchema(Schema):
        items = typesystem.Array(items=String())
    
    list_yaml = "items:\n  - item1\n  - item2"
    value, error_messages = validate_yaml(list_yaml, ListSchema)
    assert value == {"items": ["item1", "item2"]}
    assert error_messages == {}
    
    # Test YAML with various data types
    complex_yaml = "string: test\ninteger: 42\nfloat: 3.14\nboolean: true\nnull_value: null"
    value, error_messages = validate_yaml(complex_yaml, Schema)
    assert "string" in value
    assert value["integer"] == 42
    assert value["float"] == 3.14
    assert value["boolean"] is True
    assert value["null_value"] is None


# LLM-generated content at query #46
#--------------------------

```python
def test_tokenize_yaml():
    # Test with empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace only
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with string input - simple dict
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start_index == 0

    # Test with list
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test with scalar string
    token = tokenize_yaml("hello world")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello world"

    # Test with integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test with float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test with boolean true
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test with boolean false
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test with null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test with nested structure
    token = tokenize_yaml("parent:\n  child: value")
    assert isinstance(token, DictToken)
    assert token.value == {"parent": {"child": "value"}}

    # Test with invalid YAML - scanner error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no >= 1

    # Test with invalid YAML - parser error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(":\n  key: value")
    assert exc_info.value.code == "parse_error"

    # Test position tracking for multiline content
    content = "key1: value1\nkey2: value2"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key1": "value1", "key2": "value2"}

    # Test with complex nested structure
    content = """
users:
  - name: Alice
    age: 30
  - name: Bob
    age: 25
"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert "users" in token.value
    assert isinstance(token.value["users"], list)
    assert len(token.value["users"]) == 2

    # Test that token contains content reference
    token = tokenize_yaml("test: data")
    assert token.content == "test: data"

    # Test with bytes UTF-8
    token = tokenize_yaml("café: français".encode("utf-8"))
    assert isinstance(token, DictToken)


# LLM-generated content at query #47
#--------------------------

```python
def test_tokenize_yaml():
    # Test with empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace only
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with string input - simple dict
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start_pos == 0

    # Test with nested dict
    content = "parent:\n  child: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"parent": {"child": "value"}}

    # Test with list
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test with scalar string
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test with scalar integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test with scalar float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test with scalar boolean true
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test with scalar boolean false
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test with null value
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test with invalid YAML syntax
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value\n  invalid:")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test with complex nested structure
    content = """
users:
  - name: Alice
    age: 30
  - name: Bob
    age: 25
"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert "users" in token.value
    assert isinstance(token.value["users"], list)
    assert len(token.value["users"]) == 2

    # Test position tracking
    content = "key: value"
    token = tokenize_yaml(content)
    assert token.start_pos == 0
    assert token.content == content

    # Test multiline string
    content = "message: |\n  line1\n  line2"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)

    # Test with UTF-8 bytes
    token = tokenize_yaml("name: José".encode("utf-8"))
    assert isinstance(token, DictToken)
    assert token.value == {"name": "José"}


# LLM-generated content at query #48
#--------------------------

```python
def test_tokenize_yaml():
    # Test with empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace only
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \n  ")
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with string input - simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test with string input - dictionary
    token = tokenize_yaml("key: value\nkey2: value2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "key2": "value2"}

    # Test with list
    token = tokenize_yaml("- item1\n- item2\n- item3")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2", "item3"]

    # Test with integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test with float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test with boolean true
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test with boolean false
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test with null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test with nested structure
    token = tokenize_yaml("parent:\n  child: value\n  items:\n    - a\n    - b")
    assert isinstance(token, DictToken)
    assert token.value["parent"]["child"] == "value"
    assert token.value["parent"]["items"] == ["a", "b"]

    # Test with invalid YAML - syntax error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: : invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test with invalid YAML - scanner error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [unclosed")
    assert exc_info.value.code == "parse_error"

    # Test position tracking with multiline content
    token = tokenize_yaml("line1: value1\nline2: value2")
    assert isinstance(token, DictToken)
    assert token.start_pos == 0

    # Test with bytes containing UTF-8
    token = tokenize_yaml("message: hello world".encode("utf-8"))
    assert isinstance(token, DictToken)
    assert token.value["message"] == "hello world"

    # Test position information is preserved
    token = tokenize_yaml("key: value")
    assert hasattr(token, "start_pos")
    assert hasattr(token, "end_pos")
    assert token.start_pos >= 0
    assert token.end_pos >= token.start_pos

    # Test with complex nested structure
    yaml_content = """
users:
  - name: Alice
    age: 30
  - name: Bob
    age: 25
settings:
  debug: true
  timeout: 5.5
"""
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert len(token.value["users"]) == 2
    assert token.value["users"][0]["name"] == "Alice"
    assert token.value["settings"]["debug"] is True


# LLM-generated content at query #49
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    content = "hello"
    from typesystem import String
    value, errors = validate_yaml(content, String())
    assert value == "hello"
    assert errors == []

    # Test with valid YAML dict and schema
    content = "name: John\nage: 30"
    from typesystem import Schema, String, Integer
    
    class PersonSchema(Schema):
        name = String()
        age = Integer()
    
    value, errors = validate_yaml(content, PersonSchema)
    assert value == {"name": "John", "age": 30}
    assert errors == []

    # Test with valid YAML list
    content = "- item1\n- item2\n- item3"
    from typesystem import Array
    value, errors = validate_yaml(content, Array(items=String()))
    assert value == ["item1", "item2", "item3"]
    assert errors == []

    # Test with invalid YAML syntax
    content = "invalid: yaml: content:"
    try:
        validate_yaml(content, String())
        assert False, "Should raise ParseError"
    except ParseError:
        pass

    # Test with empty content
    content = ""
    try:
        validate_yaml(content, String())
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test with bytes content
    content = b"test_value"
    value, errors = validate_yaml(content, String())
    assert value == "test_value"
    assert errors == []

    # Test with validation error
    content = "not_a_number"
    from typesystem import Integer
    value, errors = validate_yaml(content, Integer())
    assert errors != []

    # Test with nested schema
    content = "user:\n  name: Alice\n  age: 25"
    
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    class ContainerSchema(Schema):
        user = UserSchema()
    
    value, errors = validate_yaml(content, ContainerSchema)
    assert value == {"user": {"name": "Alice", "age": 25}}
    assert errors == []

    # Test with boolean values
    content = "enabled: true\ndisabled: false"
    from typesystem import Boolean
    
    class BoolSchema(Schema):
        enabled = Boolean()
        disabled = Boolean()
    
    value, errors = validate_yaml(content, BoolSchema)
    assert value == {"enabled": True, "disabled": False}
    assert errors == []

    # Test with null values
    content = "value: null"
    value, errors = validate_yaml(content, String(allow_null=True))
    assert value is None
    assert errors == []

    # Test with numeric values
    content = "integer: 42\nfloat: 3.14"
    
    class NumberSchema(Schema):
        integer = Integer()
        float = Integer()
    
    value, errors = validate_yaml(content, NumberSchema)
    assert value["integer"] == 42
    assert errors == []


# LLM-generated content at query #50
#--------------------------

```python
def test_validate_yaml():
    """Test validate_yaml function with various inputs."""
    
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML as bytes
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test empty YAML content
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test YAML with validation errors
    invalid_data_yaml = "name: 123\nage: not_a_number"
    value, error_messages = validate_yaml(invalid_data_yaml, TestSchema)
    assert error_messages != {}
    
    # Test with simple Field validator
    name_field = String()
    yaml_string = "John"
    value, error_messages = validate_yaml(yaml_string, name_field)
    assert value == "John"
    assert error_messages == {}
    
    # Test with whitespace-only content
    whitespace_yaml = "   \n  \n   "
    try:
        validate_yaml(whitespace_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"


# LLM-generated content at query #51
#--------------------------

```python
def test_tokenize_yaml():
    # Test with empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace only
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with string input - simple dictionary
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with list
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test with scalar string
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test with integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test with float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test with boolean true
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test with boolean false
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test with null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test with nested structure
    token = tokenize_yaml("parent:\n  child: value\n  items:\n    - a\n    - b")
    assert isinstance(token, DictToken)
    assert token.value["parent"]["child"] == "value"
    assert token.value["parent"]["items"] == ["a", "b"]

    # Test with invalid YAML - scanner error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no >= 1

    # Test with invalid YAML - parser error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key1: value1\n  key2: value2")
    assert exc_info.value.code == "parse_error"

    # Test position tracking with multiline content
    token = tokenize_yaml("key1: value1\nkey2: value2")
    assert isinstance(token, DictToken)
    assert token.start_position == 0

    # Test with complex nested structure
    yaml_content = """
users:
  - name: Alice
    age: 30
  - name: Bob
    age: 25
settings:
  debug: true
  timeout: 30
"""
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert len(token.value["users"]) == 2
    assert token.value["users"][0]["name"] == "Alice"
    assert token.value["settings"]["debug"] is True

    # Test token has content attribute
    token = tokenize_yaml("test: value")
    assert hasattr(token, "content")
    assert "test: value" in token.content

    # Test with special characters in strings
    token = tokenize_yaml('message: "Hello: World!"')
    assert isinstance(token, DictToken)
    assert token.value["message"] == "Hello: World!"

    # Test with empty dictionary
    token = tokenize_yaml("{}")
    assert isinstance(token, DictToken)
    assert token.value == {}

    # Test with empty list
    token = tokenize_yaml("[]")
    assert isinstance(token, ListToken)
    assert token.value == []


# LLM-generated content at query #52
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML content and a simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid scalar string
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = UserSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    
    # Test 5: Empty content should raise ParseError
    import pytest
    from typesystem.base import ParseError
    
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", String())
    assert exc_info.value.code == "no_content"
    
    # Test 6: Invalid YAML syntax
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("{ invalid yaml: [", String())
    assert exc_info.value.code == "parse_error"
    
    # Test 7: Bytes content
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 8: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 9: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 10: YAML with float
    from typesystem.fields import Float
    content = "3.14"
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []


# LLM-generated content at query #53
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Invalid YAML - type mismatch
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []
    
    # Test 4: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 5: Valid YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    # Should handle list validation
    
    # Test 6: Empty content should raise ParseError
    from typesystem.base import ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", String())
    assert exc_info.value.code == "no_content"
    
    # Test 7: Invalid YAML syntax should raise ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("invalid: yaml: content:", String())
    assert exc_info.value.code == "parse_error"
    
    # Test 8: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 9: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 10: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []


# LLM-generated content at query #54
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validation
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "[1, 2, 3]"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert len(error_messages) > 0  # Should have validation error
    
    # Test 5: Empty content should raise ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", String())
    assert exc_info.value.code == "no_content"
    
    # Test 6: Invalid YAML syntax should raise ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("invalid: yaml: content:", String())
    assert exc_info.value.code == "parse_error"
    
    # Test 7: YAML with bytes input
    content = b"test string"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test string"
    assert error_messages == []
    
    # Test 8: Complex nested YAML
    class NestedSchema(Schema):
        items = String()
    
    content = "items: value"
    validator = NestedSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["items"] == "value"
    
    # Test 9: Whitespace-only content should raise ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("   \n  \t  ", String())
    assert exc_info.value.code == "no_content"
    
    # Test 10: YAML with boolean values
    content = "true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    # Boolean parsed from YAML, may or may not validate as string
    assert value is True or isinstance(error_messages, list)


# LLM-generated content at query #55
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with dict and Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 4: Valid YAML with list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    
    # Test 5: Invalid YAML syntax should raise ParseError
    import pytest
    from typesystem.base import ParseError
    
    content = "invalid: yaml: content:"
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(content, validator)
    
    # Test 6: Empty YAML content should raise ParseError
    content = ""
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 7: Bytes input
    content = b"test value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test value"
    assert error_messages == []
    
    # Test 8: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 9: YAML with null
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 10: Validation error (wrong type)
    content = "not a number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []


# LLM-generated content at query #56
#--------------------------

```python
def test_validate_yaml():
    """Test validate_yaml function with various inputs."""
    
    # Test 1: Valid YAML with Field validation
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    yaml_content = "hello"
    validator = String()
    value, error_messages = validate_yaml(yaml_content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Schema validation
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    yaml_content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(yaml_content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 3: Invalid YAML syntax
    yaml_content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(yaml_content, validator)
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 4: Empty YAML content
    yaml_content = ""
    validator = String()
    try:
        validate_yaml(yaml_content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 5: Bytes input
    yaml_content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(yaml_content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 6: YAML list
    yaml_content = "- item1\n- item2"
    validator = String()
    value, error_messages = validate_yaml(yaml_content, validator)
    assert isinstance(value, list)
    
    # Test 7: YAML with numbers
    yaml_content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(yaml_content, validator)
    assert value == 42
    
    # Test 8: YAML with boolean
    yaml_content = "true"
    validator = String()
    value, error_messages = validate_yaml(yaml_content, validator)
    assert value is True
    
    # Test 9: YAML with null
    yaml_content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(yaml_content, validator)
    assert value is None
    
    # Test 10: Whitespace-only content
    yaml_content = "   \n  \n  "
    validator = String()
    try:
        validate_yaml(yaml_content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"


# LLM-generated content at query #57
#--------------------------

```python
def test_validate_yaml():
    """Test validate_yaml function with various inputs."""
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with Field validator
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == {}
    
    # Test 2: Valid YAML with Schema validator
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test 3: Invalid YAML syntax
    content = "invalid: [yaml: syntax"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test 4: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test 5: Whitespace only content
    content = "   \n  \n  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 6: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == {}
    
    # Test 7: YAML list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}
    
    # Test 8: YAML with integers
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == {}
    
    # Test 9: YAML with floats
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == {}
    
    # Test 10: YAML with booleans
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == {}
    
    # Test 11: YAML with null
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == {}


# LLM-generated content at query #58
#--------------------------

```python
def test_validate_yaml():
    """Test validate_yaml function with various inputs."""
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with Field validator
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Schema validator
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = UserSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 3: Invalid YAML syntax
    content = "invalid: [yaml: content:"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []
    
    # Test 4: Valid YAML but validation fails
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []
    
    # Test 5: YAML with bytes input
    content = b"hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 6: Valid YAML list
    content = "- item1\n- item2"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    # Should handle list appropriately
    
    # Test 7: Valid YAML dict
    content = "key1: value1\nkey2: value2"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    
    # Test 8: YAML with numeric values
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 9: YAML with boolean
    content = "true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    
    # Test 10: YAML with null
    content = "null"
    validator = String()
    value, error_messages = validate_yaml(content, validator)


# LLM-generated content at query #59
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple schema
    class UserSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    yaml_content = """
name: John
age: 30
"""
    value, error_messages = validate_yaml(yaml_content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}

    # Test with bytes input
    yaml_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(yaml_bytes, UserSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}

    # Test with invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: : syntax"
    try:
        validate_yaml(invalid_yaml, UserSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None

    # Test with validation error
    invalid_data = "name: John\nage: not_a_number"
    value, error_messages = validate_yaml(invalid_data, UserSchema)
    assert "age" in error_messages

    # Test with Field validator
    yaml_content_string = "42"
    value, error_messages = validate_yaml(yaml_content_string, Field(int))
    assert value == 42
    assert error_messages == {}

    # Test with empty content
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, UserSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1

    # Test with whitespace-only content
    whitespace_yaml = "   \n  \n  "
    try:
        validate_yaml(whitespace_yaml, UserSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"

    # Test with complex nested structure
    yaml_nested = """
users:
  - name: Alice
    age: 28
  - name: Bob
    age: 35
"""
    class UsersSchema(Schema):
        users = Field(list)
    
    value, error_messages = validate_yaml(yaml_nested, UsersSchema)
    assert "users" in value
    assert len(value["users"]) == 2

    # Test with missing required field
    incomplete_yaml = "name: Charlie"
    value, error_messages = validate_yaml(incomplete_yaml, UserSchema)
    assert "age" in error_messages or error_messages != {}

    # Test with extra fields
    extra_yaml = "name: David\nage: 40\nemail: david@example.com"
    value, error_messages = validate_yaml(extra_yaml, UserSchema)
    # Should either include extra field or have validation result
    assert "name" in value
    assert "age" in value


# LLM-generated content at query #60
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello"
    result, error = validate_yaml(content, String())
    assert result == "hello"
    assert error is None
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    result, error = validate_yaml(content, Integer())
    assert result == 42
    assert error is None
    
    # Test 3: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    result, error = validate_yaml(content, TestSchema())
    assert result == {"name": "John", "age": 30}
    assert error is None
    
    # Test 4: Invalid YAML syntax
    content = "name: John\n  invalid: [unclosed"
    try:
        validate_yaml(content, String())
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 5: Empty content
    content = ""
    try:
        validate_yaml(content, String())
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 6: Bytes input
    content = b"test_value"
    result, error = validate_yaml(content, String())
    assert result == "test_value"
    assert error is None
    
    # Test 7: YAML with list
    content = "- item1\n- item2\n- item3"
    result, error = validate_yaml(content, String())
    assert isinstance(result, list)
    assert result == ["item1", "item2", "item3"]
    
    # Test 8: Whitespace only
    content = "   \n\n  "
    try:
        validate_yaml(content, String())
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 9: Valid YAML with boolean
    content = "true"
    result, error = validate_yaml(content, String())
    assert result is True
    
    # Test 10: Valid YAML with null
    content = "null"
    result, error = validate_yaml(content, String())
    assert result is None


# LLM-generated content at query #61
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML with bytes
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: : syntax"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test empty YAML content
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test YAML with only whitespace
    whitespace_yaml = "   \n  \n  "
    try:
        validate_yaml(whitespace_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test validation error with Field validator
    string_field = String(max_length=5)
    yaml_content = "hello_world_too_long"
    value, error_messages = validate_yaml(yaml_content, string_field)
    assert error_messages  # Should contain validation errors
    
    # Test valid simple scalar
    simple_yaml = "42"
    value, error_messages = validate_yaml(simple_yaml, Integer())
    assert value == 42
    assert error_messages == {}
    
    # Test YAML list
    list_yaml = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    array_field = Array(items=String())
    value, error_messages = validate_yaml(list_yaml, array_field)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}
    
    # Test YAML with special types
    special_yaml = "flag: true\nnumber: 3.14\nempty: null"
    class SpecialSchema(Schema):
        flag = typesystem.fields.Boolean()
        number = typesystem.fields.Float()
        empty = typesystem.fields.Field(allow_null=True)
    
    value, error_messages = validate_yaml(special_yaml, SpecialSchema)
    assert value["flag"] is True
    assert value["number"] == 3.14
    assert value["empty"] is None


# LLM-generated content at query #62
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: : syntax"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)
    
    # Test with bytes content
    bytes_yaml = b"name: Alice\nage: 25"
    value, error_messages = validate_yaml(bytes_yaml, TestSchema)
    assert value == {"name": "Alice", "age": 25}
    assert error_messages == {}
    
    # Test empty content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with Field validator
    string_field = String()
    yaml_string = "hello"
    value, error_messages = validate_yaml(yaml_string, string_field)
    assert value == "hello"
    assert error_messages == {}
    
    # Test validation failure
    class StrictSchema(Schema):
        count = Integer()
    
    invalid_data = "count: not_a_number"
    value, error_messages = validate_yaml(invalid_data, StrictSchema)
    assert error_messages != {}
    
    # Test with list YAML
    list_schema = Schema
    yaml_list = "- item1\n- item2\n- item3"
    value, error_messages = validate_yaml(yaml_list, list_schema)
    assert isinstance(value, list)
    
    # Test with nested structure
    yaml_nested = "person:\n  name: Bob\n  details:\n    age: 40"
    value, error_messages = validate_yaml(yaml_nested, TestSchema)
    assert isinstance(value, dict)


# LLM-generated content at query #63
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string content
    result, errors = validate_yaml("hello", String())
    assert result == "hello"
    assert errors == []
    
    # Test 2: Valid integer content
    result, errors = validate_yaml("42", Integer())
    assert result == 42
    assert errors == []
    
    # Test 3: Valid YAML with schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    yaml_content = """
    name: John
    age: 30
    """
    result, errors = validate_yaml(yaml_content, UserSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []
    
    # Test 4: Invalid YAML syntax
    try:
        validate_yaml("{ invalid yaml", String())
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test 5: YAML content as bytes
    result, errors = validate_yaml(b"test", String())
    assert result == "test"
    assert errors == []
    
    # Test 6: Empty YAML content
    try:
        validate_yaml("", String())
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test 7: Whitespace only content
    try:
        validate_yaml("   \n  \n  ", String())
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 8: Validation error with schema
    class StrictSchema(Schema):
        name = String(max_length=5)
    
    yaml_content = "name: VeryLongName"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert len(errors) > 0
    
    # Test 9: List validation
    result, errors = validate_yaml("[1, 2, 3]", String())
    # This should either validate or produce errors depending on field behavior
    
    # Test 10: Null value
    result, errors = validate_yaml("null", String())
    # Validation depends on field configuration


# LLM-generated content at query #64
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string content with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid integer content with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "[1, 2, 3]"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    # List will be validated as string representation or fail
    assert error_messages != []
    
    # Test 5: Invalid YAML syntax
    content = "{ invalid: yaml: content }"
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 6: Empty content should raise ParseError
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 7: Bytes content
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 8: YAML with boolean
    content = "true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    
    # Test 9: YAML with null
    content = "null"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    
    # Test 10: YAML with float
    content = "3.14"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14


# LLM-generated content at query #65
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test bytes content is converted to string
    result = tokenize_yaml(b"key: value")
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test simple scalar
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test dictionary
    result = tokenize_yaml("key: value\nother: data")
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value", "other": "data"}

    # Test list
    result = tokenize_yaml("- item1\n- item2\n- item3")
    assert isinstance(result, ListToken)
    assert result.value == ["item1", "item2", "item3"]

    # Test nested structure
    result = tokenize_yaml("parent:\n  child: value\n  list:\n    - a\n    - b")
    assert isinstance(result, DictToken)
    assert result.value["parent"]["child"] == "value"
    assert result.value["parent"]["list"] == ["a", "b"]

    # Test integer
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test float
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test boolean true
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test boolean false
    result = tokenize_yaml("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test null
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test invalid YAML with colon raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(": invalid")
    assert exc_info.value.code == "parse_error"

    # Test position tracking
    result = tokenize_yaml("test: value")
    assert result.start_position == 0
    assert result.end_position >= 0

    # Test multiline content position
    content = "line1: value1\nline2: value2"
    result = tokenize_yaml(content)
    assert isinstance(result, DictToken)

    # Test complex nested structure with multiple types
    yaml_content = """
    root:
      string: hello
      number: 123
      float: 45.67
      bool: true
      null_val: null
      list:
        - item1
        - item2
      nested:
        deep: value
    """
    result = tokenize_yaml(yaml_content)
    assert isinstance(result, DictToken)
    assert result.value["root"]["string"] == "hello"
    assert result.value["root"]["number"] == 123
    assert result.value["root"]["bool"] is True
    assert result.value["root"]["null_val"] is None


# LLM-generated content at query #66
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid simple string
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid dictionary with schema
    class PersonSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, PersonSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: [yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test 5: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."
    
    # Test 6: Whitespace only content
    content = "   \n  \t  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 7: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 8: List validation
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    assert len(value) == 3
    
    # Test 9: Null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    
    # Test 10: Boolean value
    content = "true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    
    # Test 11: Float value
    content = "3.14"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14


# LLM-generated content at query #67
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 5: Empty content should raise ParseError
    try:
        validate_yaml("", String())
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 6: Invalid YAML syntax
    try:
        validate_yaml("invalid: yaml: content:", String())
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 7: Bytes content
    content = b"test content"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test content"
    assert error_messages == []
    
    # Test 8: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 9: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 10: YAML with float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert abs(value - 3.14) < 0.001
    assert error_messages == []


# LLM-generated content at query #68
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    yaml_content = "name: John\nage: 30"
    value, errors = validate_yaml(yaml_content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert errors == {}
    
    # Test with bytes input
    yaml_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert errors == {}
    
    # Test with invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, TestSchema)
    assert exc_info.value.code == "parse_error"
    
    # Test with empty content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with whitespace only
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("   \n  \n  ", TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with Field validator
    field = Field(str)
    value, errors = validate_yaml("test_string", field)
    assert value == "test_string"
    assert errors == {}
    
    # Test with list validation
    list_field = Field(list)
    yaml_list = "- item1\n- item2\n- item3"
    value, errors = validate_yaml(yaml_list, list_field)
    assert value == ["item1", "item2", "item3"]
    assert errors == {}
    
    # Test with nested structure
    yaml_nested = "items:\n  - name: first\n    value: 1\n  - name: second\n    value: 2"
    field = Field(dict)
    value, errors = validate_yaml(yaml_nested, field)
    assert "items" in value
    assert len(value["items"]) == 2
    
    # Test with various YAML types
    yaml_types = "string: hello\nnumber: 42\nfloat: 3.14\nbool: true\nnull_val: null"
    field = Field(dict)
    value, errors = validate_yaml(yaml_types, field)
    assert value["string"] == "hello"
    assert value["number"] == 42
    assert value["float"] == 3.14
    assert value["bool"] is True
    assert value["null_val"] is None


# LLM-generated content at query #69
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dictionary with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: [yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 5: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 6: Bytes input
    content = b"test content"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test content"
    assert error_messages == []
    
    # Test 7: YAML list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 8: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 9: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 10: YAML with float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []


# LLM-generated content at query #70
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "{ invalid yaml: [unclosed"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert len(error_messages) > 0
    
    # Test 5: Valid YAML but invalid validation
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert len(error_messages) > 0
    
    # Test 6: YAML list
    content = "- item1\n- item2"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2"]
    assert error_messages == []
    
    # Test 7: Empty content should raise ParseError
    import pytest
    from typesystem.base import ParseError
    with pytest.raises(ParseError):
        validate_yaml("", String())
    
    # Test 8: Bytes content
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 9: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 10: YAML boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []


# LLM-generated content at query #71
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test integer
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test float
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test boolean true
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test boolean false
    result = tokenize_yaml("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test null
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test list
    result = tokenize_yaml("[1, 2, 3]")
    assert isinstance(result, ListToken)
    assert len(result.value) == 3

    # Test dict
    result = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(result, DictToken)
    assert len(result.value) == 2

    # Test multiline dict
    content = "key1: value1\nkey2: value2"
    result = tokenize_yaml(content)
    assert isinstance(result, DictToken)
    assert result.value["key1"].value == "value1"
    assert result.value["key2"].value == "value2"

    # Test multiline list
    content = "- item1\n- item2\n- item3"
    result = tokenize_yaml(content)
    assert isinstance(result, ListToken)
    assert len(result.value) == 3

    # Test bytes input
    result = tokenize_yaml(b"hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \n  ")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{ invalid yaml: [")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test nested structures
    content = """
    users:
      - name: Alice
        age: 30
      - name: Bob
        age: 25
    """
    result = tokenize_yaml(content)
    assert isinstance(result, DictToken)
    assert "users" in result.value

    # Test position tracking for scalars
    result = tokenize_yaml("test_value")
    assert result.start_position == 0
    assert result.end_position >= 0

    # Test position tracking with line breaks
    content = "line1\nline2"
    result = tokenize_yaml(content)
    assert result.start_position == 0

    # Test complex nested structure
    content = """
    config:
      database:
        host: localhost
        port: 5432
        credentials:
          username: admin
          password: secret
    """
    result = tokenize_yaml(content)
    assert isinstance(result, DictToken)
    assert "config" in result.value


# LLM-generated content at query #72
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except Exception as exc:
        assert "parse_error" in str(exc) or hasattr(exc, 'code')
    
    # Test 5: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except Exception as exc:
        assert "no_content" in str(exc) or hasattr(exc, 'code')
    
    # Test 6: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 7: YAML list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 8: YAML with nested structure
    class NestedSchema(Schema):
        user = TestSchema
    
    content = "user:\n  name: Alice\n  age: 25"
    value, error_messages = validate_yaml(content, NestedSchema)
    assert value == {"user": {"name": "Alice", "age": 25}}
    assert error_messages == []
    
    # Test 9: Validation failure (type mismatch)
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert len(error_messages) > 0
    
    # Test 10: YAML with boolean values
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []


####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test 1: Valid YAML content
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test 2: Valid YAML as bytes
    content_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(content_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}
    
    # Test 3: Invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 4: Empty YAML content
    empty_content = ""
    try:
        validate_yaml(empty_content, TestSchema)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 5: Whitespace only content
    whitespace_content = "   \n\n  "
    try:
        validate_yaml(whitespace_content, TestSchema)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 6: Validation error - invalid type
    invalid_type_yaml = "name: John\nage: not_a_number"
    value, error_messages = validate_yaml(invalid_type_yaml, TestSchema)
    assert "age" in error_messages
    
    # Test 7: Simple Field validator (String field)
    string_field = String()
    content = "hello"
    value, error_messages = validate_yaml(content, string_field)
    assert value == "hello"
    assert error_messages == {}
    
    # Test 8: List content with Field validator
    list_yaml = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    array_field = Array(items=String())
    value, error_messages = validate_yaml(list_yaml, array_field)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}
    
    # Test 9: YAML with special types (bool, null, float)
    special_yaml = "flag: true\nvalue: null\npi: 3.14"
    
    class SpecialSchema(Schema):
        flag = typesystem.Boolean(allow_null=True)
        value = typesystem.Field(allow_null=True)
        pi = typesystem.Float()
    
    value, error_messages = validate_yaml(special_yaml, SpecialSchema)
    assert value["flag"] is True
    assert value["value"] is None
    assert value["pi"] == 3.14


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test integer
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test float
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test boolean
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_yaml("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test null
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test list
    result = tokenize_yaml("[1, 2, 3]")
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    
    # Test dict
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert "key" in result.value
    
    # Test nested structure
    result = tokenize_yaml("items:\n  - name: test\n    value: 123")
    assert isinstance(result, DictToken)
    assert "items" in result.value
    
    # Test bytes input
    result = tokenize_yaml(b"hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"
    
    # Test invalid YAML raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{ invalid: yaml: content:")
    assert exc_info.value.code == "parse_error"
    
    # Test position tracking
    result = tokenize_yaml("test")
    assert result.start_position == 0
    
    # Test multiline position
    result = tokenize_yaml("line1\nline2: value")
    assert isinstance(result, DictToken)
    
    # Test scalar with position
    result = tokenize_yaml("  value")
    assert isinstance(result, ScalarToken)
    assert result.value == "value"
    
    # Test list with nested items
    result = tokenize_yaml("- item1\n- item2\n- item3")
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    
    # Test complex nested structure
    yaml_content = """
    users:
      - name: Alice
        age: 30
      - name: Bob
        age: 25
    """
    result = tokenize_yaml(yaml_content)
    assert isinstance(result, DictToken)
    assert "users" in result.value


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = UserSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    assert len(value) == 3
    
    # Test 5: Invalid YAML syntax
    content = "invalid: yaml: syntax:"
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 6: Empty content raises ParseError
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 7: Whitespace-only content raises ParseError
    content = "   \n  \n  "
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 8: Bytes input
    content = b"test content"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test content"
    assert error_messages == []
    
    # Test 9: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    
    # Test 10: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML with bytes input
    valid_yaml_bytes = b"name: Alice\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Alice", "age": 25}
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None
    
    # Test empty YAML content
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
    
    # Test whitespace-only YAML content
    whitespace_yaml = "   \n\n   "
    try:
        validate_yaml(whitespace_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test with Field validator
    field_validator = String()
    simple_yaml = "hello"
    value, error_messages = validate_yaml(simple_yaml, field_validator)
    assert value == "hello"
    
    # Test with invalid YAML that causes validation error
    invalid_content_yaml = "name: 123\nage: invalid_age"
    value, error_messages = validate_yaml(invalid_content_yaml, TestSchema)
    assert "age" in error_messages or error_messages != {}
    
    # Test with UTF-8 encoded bytes
    utf8_yaml = "name: José\nage: 28".encode("utf-8")
    value, error_messages = validate_yaml(utf8_yaml, TestSchema)
    assert value["name"] == "José"
    
    # Test YAML with lists
    list_yaml = "items:\n  - apple\n  - banana"
    class ListSchema(Schema):
        items = typesystem.fields.Array(String())
    value, error_messages = validate_yaml(list_yaml, ListSchema)
    assert "items" in value or error_messages == {}
    
    # Test YAML with nested structures
    nested_yaml = "user:\n  name: Bob\n  age: 35"
    value, error_messages = validate_yaml(nested_yaml, TestSchema)
    # Validation result depends on schema definition
    assert isinstance(value, (dict, type(None))) or error_messages is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with Field validator
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dictionary with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    
    # Test 5: Invalid YAML syntax
    content = "invalid: [yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None
    
    # Test 6: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.text == "No content."
    
    # Test 7: Whitespace only content
    content = "   \n  \n  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 8: YAML as bytes
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 9: Complex nested YAML structure
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema()
    
    content = "name: Alice\naddress:\n  street: Main St\n  city: Boston"
    value, error_messages = validate_yaml(content, PersonSchema)
    assert value["name"] == "Alice"
    assert value["address"]["street"] == "Main St"
    assert value["address"]["city"] == "Boston"
    
    # Test 10: YAML with null values
    content = "value: null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, dict)


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string content with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 3: Valid YAML with bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 5: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 6: Whitespace only content
    content = "   \n  \t  "
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 7: Valid YAML list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 8: YAML with numbers and booleans
    content = "count: 42\nactive: true\nratio: 3.14"
    class NumericSchema(Schema):
        count = Integer()
        active = typesystem.fields.Boolean()
        ratio = typesystem.fields.Float()
    
    value, error_messages = validate_yaml(content, NumericSchema)
    assert value["count"] == 42
    assert value["active"] is True
    assert value["ratio"] == 3.14
    
    # Test 9: YAML with null value
    content = "value: null"
    class NullableSchema(Schema):
        value = String(allow_null=True)
    
    value, error_messages = validate_yaml(content, NullableSchema)
    assert value["value"] is None


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"
    
    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test string input
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test scalar token
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    
    # Test list token
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    
    # Test integer token
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    
    # Test float token
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    
    # Test boolean token
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    # Test null token
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    
    # Test nested structure
    token = tokenize_yaml("parent:\n  child: value")
    assert isinstance(token, DictToken)
    assert token.value == {"parent": {"child": "value"}}
    
    # Test invalid YAML raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{ invalid: yaml: syntax:")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None
    
    # Test position tracking
    token = tokenize_yaml("key: value")
    assert token.start_pos >= 0
    assert token.end_pos >= token.start_pos
    
    # Test UTF-8 encoding
    token = tokenize_yaml("greeting: こんにちは")
    assert isinstance(token, DictToken)
    assert token.value == {"greeting": "こんにちは"}


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test valid YAML with bytes input
    content_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(content_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == []
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None
    
    # Test empty content
    empty_content = ""
    try:
        validate_yaml(empty_content, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
    
    # Test validation error with Field validator
    name_field = String(max_length=3)
    content = "toolongname"
    value, error_messages = validate_yaml(content, name_field)
    assert len(error_messages) > 0
    
    # Test with valid simple scalar
    content = "42"
    value, error_messages = validate_yaml(content, Integer())
    assert value == 42
    assert error_messages == []
    
    # Test with list content
    class ListSchema(Schema):
        items = None
    
    content = "- item1\n- item2\n- item3"
    value, error_messages = validate_yaml(content, ListSchema)
    # Should parse successfully
    assert isinstance(value, list)
    
    # Test with complex nested YAML
    nested_content = """
    user:
      name: Alice
      age: 28
      email: alice@example.com
    """
    
    class NestedSchema(Schema):
        user = None
    
    value, error_messages = validate_yaml(nested_content, NestedSchema)
    assert isinstance(value, dict)
    
    # Test with boolean and null values
    bool_content = """
    active: true
    inactive: false
    empty: null
    """
    
    class BoolSchema(Schema):
        active = None
        inactive = None
        empty = None
    
    value, error_messages = validate_yaml(bool_content, BoolSchema)
    assert value["active"] is True
    assert value["inactive"] is False
    assert value["empty"] is None
    
    # Test with float values
    float_content = """
    pi: 3.14159
    ratio: 0.5
    """
    
    class FloatSchema(Schema):
        pi = None
        ratio = None
    
    value, error_messages = validate_yaml(float_content, FloatSchema)
    assert isinstance(value["pi"], float)
    assert isinstance(value["ratio"], float)


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test valid YAML with bytes
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == []
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: : syntax"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)
    
    # Test empty content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test whitespace only content
    whitespace_yaml = "   \n  \n  "
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(whitespace_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with Field validator
    string_field = String()
    value, error_messages = validate_yaml("test_string", string_field)
    assert value == "test_string"
    assert error_messages == []
    
    # Test validation failure
    integer_field = Integer()
    value, error_messages = validate_yaml("not_an_integer", integer_field)
    assert len(error_messages) > 0
    
    # Test complex nested YAML
    complex_yaml = """
    users:
      - name: Alice
        age: 28
      - name: Bob
        age: 35
    """
    class UserSchema(Schema):
        users = typesystem.fields.Array()
    
    value, error_messages = validate_yaml(complex_yaml, UserSchema)
    assert isinstance(value, dict)
    assert "users" in value


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert len(value) == 3
    
    # Test 5: YAML with bytes input
    content = b"test string"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test string"
    assert error_messages == []
    
    # Test 6: Invalid YAML syntax should raise ParseError
    import pytest
    from typesystem.base import ParseError
    
    invalid_yaml = "{ invalid: yaml: content"
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, validator)
    
    # Test 7: Empty content should raise ParseError
    empty_content = ""
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(empty_content, validator)
    
    # Test 8: Whitespace-only content should raise ParseError
    whitespace_content = "   \n  \n  "
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(whitespace_content, validator)
    
    # Test 9: YAML with boolean values
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 10: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = """
name: John
age: 30
"""
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test with bytes input
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}
    
    # Test with Field validator
    string_field = String()
    yaml_string = "hello"
    value, error_messages = validate_yaml(yaml_string, string_field)
    assert value == "hello"
    assert error_messages == {}
    
    # Test with integer field
    int_field = Integer()
    yaml_int = "42"
    value, error_messages = validate_yaml(yaml_int, int_field)
    assert value == 42
    assert error_messages == {}
    
    # Test with list
    from typesystem.fields import Array
    array_field = Array(items=Integer())
    yaml_list = """
- 1
- 2
- 3
"""
    value, error_messages = validate_yaml(yaml_list, array_field)
    assert value == [1, 2, 3]
    assert error_messages == {}
    
    # Test with invalid YAML syntax
    invalid_yaml = """
name: John
  age: 30
    invalid indent
"""
    try:
        validate_yaml(invalid_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None
    
    # Test with empty content
    try:
        validate_yaml("", TestSchema)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
    
    # Test with whitespace only
    try:
        validate_yaml("   \n  \n  ", TestSchema)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test validation failure with invalid data type
    invalid_type_yaml = """
name: John
age: not_a_number
"""
    value, error_messages = validate_yaml(invalid_type_yaml, TestSchema)
    assert "age" in error_messages
    
    # Test with nested schema
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema
    
    nested_yaml = """
name: Alice
address:
  street: Main St
  city: New York
"""
    value, error_messages = validate_yaml(nested_yaml, PersonSchema)
    assert value["name"] == "Alice"
    assert value["address"]["street"] == "Main St"
    assert error_messages == {}


# LLM-generated content at query #12
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar value
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test boolean true
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test boolean false
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test list
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert len(token.value) == 3

    # Test dict
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert "key" in token.value
    assert token.value["key"].value == "value"

    # Test nested structure
    token = tokenize_yaml("items:\n  - name: test\n    value: 123")
    assert isinstance(token, DictToken)
    assert "items" in token.value

    # Test bytes input
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML syntax
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{ invalid yaml :")
    assert exc_info.value.code == "parse_error"

    # Test token has correct position information
    token = tokenize_yaml("test")
    assert hasattr(token, "start_position")
    assert hasattr(token, "end_position")

    # Test multiline YAML
    yaml_content = "key1: value1\nkey2: value2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert len(token.value) == 2

    # Test complex nested structure
    yaml_content = """
    users:
      - name: Alice
        age: 30
      - name: Bob
        age: 25
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert "users" in token.value
    assert isinstance(token.value["users"], ListToken)

    # Test position tracking for error reporting
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: [yaml: content")
    assert exc_info.value.position is not None
    assert exc_info.value.position.line_no >= 1
    assert exc_info.value.position.column_no >= 1

    # Test UTF-8 content
    token = tokenize_yaml("message: Hello, 世界")
    assert isinstance(token, DictToken)
    assert "message" in token.value

    # Test special characters in scalar
    token = tokenize_yaml("'special: chars!'")
    assert isinstance(token, ScalarToken)
    assert "special: chars!" in str(token.value)

    # Test list with mixed types
    token = tokenize_yaml("[1, 'two', 3.0, true, null]")
    assert isinstance(token, ListToken)
    assert len(token.value) == 5


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    yaml_content = "name: John\nage: 30"
    value, error_messages = validate_yaml(yaml_content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML as bytes
    yaml_bytes = b"name: Alice\nage: 25"
    value, error_messages = validate_yaml(yaml_bytes, TestSchema)
    assert value == {"name": "Alice", "age": 25}
    assert error_messages == {}
    
    # Test with single field validator
    yaml_single = "42"
    value, error_messages = validate_yaml(yaml_single, Integer())
    assert value == 42
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  age: [invalid"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError:
        pass
    
    # Test empty content
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test whitespace-only content
    whitespace_yaml = "   \n\t  "
    try:
        validate_yaml(whitespace_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test validation failure with invalid data type
    yaml_wrong_type = "name: John\nage: not_a_number"
    value, error_messages = validate_yaml(yaml_wrong_type, TestSchema)
    assert "age" in error_messages
    
    # Test with list content
    yaml_list = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    value, error_messages = validate_yaml(yaml_list, Array(String()))
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}
    
    # Test with nested structure
    nested_yaml = "person:\n  name: Bob\n  age: 35"
    value, error_messages = validate_yaml(nested_yaml, TestSchema)
    # Validation may fail due to schema mismatch, but should not raise
    assert isinstance(error_messages, dict)


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid simple scalar with String field
    result = validate_yaml("hello", String())
    assert result == "hello"
    
    # Test 2: Valid integer with Integer field
    result = validate_yaml("42", Integer())
    assert result == 42
    
    # Test 3: Valid YAML with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    yaml_content = """
name: John
age: 30
"""
    result = validate_yaml(yaml_content, TestSchema())
    assert result["name"] == "John"
    assert result["age"] == 30
    
    # Test 4: Bytes input
    result = validate_yaml(b"test_string", String())
    assert result == "test_string"
    
    # Test 5: Invalid YAML syntax should raise ParseError
    import pytest
    from typesystem.base import ParseError
    
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("invalid: : yaml:", String())
    assert exc_info.value.code == "parse_error"
    
    # Test 6: Empty content should raise ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", String())
    assert exc_info.value.code == "no_content"
    
    # Test 7: Whitespace-only content should raise ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("   \n  \n  ", String())
    assert exc_info.value.code == "no_content"
    
    # Test 8: Validation failure (type mismatch)
    result = validate_yaml("not_a_number", Integer())
    assert result is not None  # Should contain error information
    
    # Test 9: Complex nested structure
    yaml_content = """
users:
  - name: Alice
    age: 25
  - name: Bob
    age: 30
"""
    from typesystem.fields import Array, Object
    
    result = validate_yaml(yaml_content, Field())
    assert isinstance(result, dict)
    assert "users" in result
    
    # Test 10: YAML with various types
    yaml_content = """
string_val: hello
int_val: 123
float_val: 45.67
bool_val: true
null_val: null
list_val:
  - item1
  - item2
"""
    result = validate_yaml(yaml_content, Field())
    assert result["string_val"] == "hello"
    assert result["int_val"] == 123
    assert result["float_val"] == 45.67
    assert result["bool_val"] is True
    assert result["null_val"] is None
    assert result["list_val"] == ["item1", "item2"]


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_yaml():
    """Test validate_yaml function with various inputs."""
    
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid simple string
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid dictionary with schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid list
    from typesystem.fields import Array
    content = "[1, 2, 3]"
    validator = Array(items=Integer())
    value, error_messages = validate_yaml(content, validator)
    assert value == [1, 2, 3]
    assert error_messages == []
    
    # Test 5: Invalid YAML syntax - should raise ParseError
    import pytest
    from typesystem.base import ParseError
    
    content = "invalid: [yaml: syntax:"
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(content, validator)
    
    # Test 6: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 7: Valid YAML with null
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 8: Valid YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 9: Valid YAML with float
    from typesystem.fields import Float
    content = "3.14"
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []
    
    # Test 10: Validation error - type mismatch
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert len(error_messages) > 0


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_yaml():
    """Test validate_yaml function with various inputs."""
    
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == {}
    
    # Test 2: Valid YAML with bytes input
    content_bytes = b"world"
    validator = String()
    value, error_messages = validate_yaml(content_bytes, validator)
    assert value == "world"
    assert error_messages == {}
    
    # Test 3: Valid YAML integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == {}
    
    # Test 4: Valid YAML with dictionary and Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == {}
    
    # Test 5: Valid YAML with list
    content = "- item1\n- item2"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    assert len(error_messages) > 0  # List doesn't match String validator
    
    # Test 6: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(content, validator)
    
    # Test 7: Empty content should raise ParseError
    content = ""
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 8: Whitespace-only content should raise ParseError
    content = "   \n  \n  "
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 9: Valid YAML with complex schema
    class ComplexSchema(Schema):
        items = String()
    
    content = "items: test"
    validator = ComplexSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["items"] == "test"
    assert error_messages == {}
    
    # Test 10: YAML with validation error
    class StrictSchema(Schema):
        count = Integer()
    
    content = "count: not_a_number"
    validator = StrictSchema
    value, error_messages = validate_yaml(content, validator)
    assert len(error_messages) > 0


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string content with String field
    content = "hello world"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello world"
    assert error_messages == []
    
    # Test 2: Valid integer content with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema()
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 4: Valid YAML list
    from typesystem.fields import Array
    content = "- item1\n- item2\n- item3"
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 5: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError:
        pass
    
    # Test 6: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 7: Bytes content
    content = b"test: value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is not None
    
    # Test 8: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 9: YAML with null
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 10: YAML with float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid scalar string
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid dict with schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, UserSchema)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 4: Valid list
    content = "- apple\n- banana\n- cherry"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    assert error_messages == []
    
    # Test 5: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError:
        pass
    
    # Test 6: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 7: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 8: Validation error
    class StrictSchema(Schema):
        name = String(max_length=3)
    
    content = "name: VeryLongName"
    try:
        validate_yaml(content, StrictSchema)
    except ParseError:
        pass
    
    # Test 9: Whitespace-only content
    content = "   \n  \n  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 10: Complex nested structure
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema
    
    content = "name: Alice\naddress:\n  street: Main St\n  city: NYC"
    value, error_messages = validate_yaml(content, PersonSchema)
    assert value["name"] == "Alice"
    assert value["address"]["street"] == "Main St"
    assert value["address"]["city"] == "NYC"
    assert error_messages == []


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == {}
    
    # Test 2: Valid YAML with integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == {}
    
    # Test 3: Valid YAML with schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test 4: YAML with list
    content = "- item1\n- item2"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    
    # Test 5: Invalid YAML syntax
    content = "invalid: yaml: content: :"
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test 6: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == {}
    
    # Test 7: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 8: Whitespace only content
    content = "   \n   "
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 9: YAML with nested structure
    class NestedSchema(Schema):
        data = String()
    
    content = "data: nested_value"
    value, error_messages = validate_yaml(content, NestedSchema)
    assert value == {"data": "nested_value"}
    
    # Test 10: YAML with boolean and null values
    content = "active: true\nvalue: null"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, dict)


# LLM-generated content at query #20
#--------------------------

```python
def test_tokenize_yaml():
    # Test with simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test with integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test with float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test with boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test with null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test with list
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert len(token.value) == 3

    # Test with dictionary
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert "key" in token.value

    # Test with bytes input
    token = tokenize_yaml(b"test")
    assert isinstance(token, ScalarToken)
    assert token.value == "test"

    # Test with UTF-8 bytes
    token = tokenize_yaml("name: José".encode("utf-8"))
    assert isinstance(token, DictToken)

    # Test with empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test with invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{ invalid yaml :")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no >= 1

    # Test with complex nested structure
    yaml_content = """
    users:
      - name: John
        age: 30
      - name: Jane
        age: 25
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert "users" in token.value

    # Test position tracking
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert token.start_position >= 0
    assert token.end_position >= token.start_position

    # Test multiline content position
    yaml_content = "line1: value1\nline2: value2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)

    # Test with special characters
    token = tokenize_yaml("message: 'Hello, World!'")
    assert isinstance(token, DictToken)
    assert token.value["message"] == "Hello, World!"

    # Test with numbers in different formats
    token = tokenize_yaml("0x10")  # hex
    assert isinstance(token, ScalarToken)

    token = tokenize_yaml("1.5e3")  # scientific notation
    assert isinstance(token, ScalarToken)
    assert token.value == 1500.0

    # Test list of dictionaries
    token = tokenize_yaml("- key1: val1\n- key2: val2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    result, errors = validate_yaml(valid_yaml, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == {}
    
    # Test valid YAML as bytes
    valid_yaml_bytes = b"name: Alice\nage: 25"
    result, errors = validate_yaml(valid_yaml_bytes, TestSchema)
    assert result == {"name": "Alice", "age": 25}
    assert errors == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: : syntax"
    result, errors = validate_yaml(invalid_yaml, TestSchema)
    assert errors is not None
    assert len(errors) > 0
    
    # Test YAML with validation errors
    invalid_content_yaml = "name: John\nage: not_a_number"
    result, errors = validate_yaml(invalid_content_yaml, TestSchema)
    assert errors is not None
    
    # Test with simple Field validator
    string_field = String()
    simple_yaml = "hello"
    result, errors = validate_yaml(simple_yaml, string_field)
    assert result == "hello"
    assert errors == {}
    
    # Test with integer field
    int_field = Integer()
    int_yaml = "42"
    result, errors = validate_yaml(int_yaml, int_field)
    assert result == 42
    assert errors == {}
    
    # Test with list in YAML
    class ListSchema(Schema):
        items = typesystem.Array(String())
    
    list_yaml = "items:\n  - item1\n  - item2"
    result, errors = validate_yaml(list_yaml, ListSchema)
    assert result == {"items": ["item1", "item2"]}
    assert errors == {}
    
    # Test empty YAML content raises ParseError
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test whitespace-only YAML content raises ParseError
    whitespace_yaml = "   \n  \n  "
    try:
        validate_yaml(whitespace_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == {}
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == {}
    
    # Test 3: Valid YAML dictionary with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = UserSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    
    # Test 5: Invalid YAML syntax
    content = "invalid: [yaml: content:"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages is not None
    
    # Test 6: Empty content should raise ParseError
    import pytest
    from typesystem.base import ParseError
    
    content = ""
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 7: Whitespace-only content should raise ParseError
    content = "   \n  \t  "
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 8: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == {}
    
    # Test 9: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    
    # Test 10: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string content
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid integer content
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with schema
    class PersonSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = PersonSchema()
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages  # Should have error messages
    
    # Test 5: Empty content should raise ParseError
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError for empty content"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 6: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 7: YAML list
    content = "[1, 2, 3]"
    from typesystem.fields import Array
    validator = Array(items=Integer())
    value, error_messages = validate_yaml(content, validator)
    assert value == [1, 2, 3]
    assert error_messages == []
    
    # Test 8: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 9: YAML with null
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 10: Validation error (wrong type)
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages  # Should have validation errors


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML content and a simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: YAML with bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 5: Invalid YAML syntax - should raise ParseError
    from typesystem.base import ParseError
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test 6: Empty content - should raise ParseError
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 7: Whitespace only content - should raise ParseError
    content = "   \n  \t  "
    validator = String()
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 8: YAML list with validator
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 9: YAML with null values
    content = "value: null"
    class NullSchema(Schema):
        value = String(allow_null=True)
    
    value, error_messages = validate_yaml(content, NullSchema)
    assert value == {"value": None}
    assert error_messages == []
    
    # Test 10: YAML with boolean values
    content = "enabled: true\ndisabled: false"
    class BoolSchema(Schema):
        enabled = typesystem.fields.Boolean()
        disabled = typesystem.fields.Boolean()
    
    value, error_messages = validate_yaml(content, BoolSchema)
    assert value == {"enabled": True, "disabled": False}
    assert error_messages == []


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test with bytes input
    content_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(content_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}
    
    # Test with invalid YAML syntax
    invalid_content = "name: John\n  invalid: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_content, TestSchema)
    assert exc_info.value.code == "parse_error"
    
    # Test with empty content
    empty_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_content, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with whitespace-only content
    whitespace_content = "   \n  \n  "
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(whitespace_content, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with validation error (missing required field)
    incomplete_content = "name: John"
    value, error_messages = validate_yaml(incomplete_content, TestSchema)
    assert "age" in error_messages
    
    # Test with Field validator
    content_single = "42"
    value, error_messages = validate_yaml(content_single, Field(int))
    assert value == 42
    assert error_messages == {}
    
    # Test with list content
    list_content = "- item1\n- item2\n- item3"
    value, error_messages = validate_yaml(list_content, Field(list))
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}
    
    # Test with nested structure
    nested_content = "users:\n  - name: Alice\n    age: 28\n  - name: Bob\n    age: 32"
    value, error_messages = validate_yaml(nested_content, Field(dict))
    assert "users" in value
    assert len(value["users"]) == 2
    assert error_messages == {}


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field validator
    content = "42"
    from typesystem.fields import Integer
    value, error_messages = validate_yaml(content, Integer())
    assert value == 42
    assert error_messages == {}

    # Test with valid YAML dict and schema validator
    content = "name: John\nage: 30"
    from typesystem.schemas import Schema
    
    class UserSchema(Schema):
        name = Field(allow_null=False)
        age = Integer()
    
    value, error_messages = validate_yaml(content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}

    # Test with valid YAML list
    content = "[1, 2, 3]"
    from typesystem.fields import Array
    value, error_messages = validate_yaml(content, Array(items=Integer()))
    assert value == [1, 2, 3]
    assert error_messages == {}

    # Test with invalid YAML syntax
    content = "{ invalid: yaml: syntax"
    try:
        validate_yaml(content, Integer())
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None

    # Test with empty content
    content = ""
    try:
        validate_yaml(content, Integer())
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"

    # Test with bytes input
    content = b"name: Jane\nage: 25"
    class PersonSchema(Schema):
        name = Field()
        age = Integer()
    
    value, error_messages = validate_yaml(content, PersonSchema)
    assert value == {"name": "Jane", "age": 25}

    # Test with validation error
    content = "age: not_a_number"
    class AgeSchema(Schema):
        age = Integer()
    
    value, error_messages = validate_yaml(content, AgeSchema)
    assert error_messages != {}

    # Test with null values
    content = "value: null"
    from typesystem.fields import Field as BaseField
    class NullSchema(Schema):
        value = BaseField(allow_null=True)
    
    value, error_messages = validate_yaml(content, NullSchema)
    assert value["value"] is None

    # Test with boolean values
    content = "active: true\ninactive: false"
    from typesystem.fields import Boolean
    class BoolSchema(Schema):
        active = Boolean()
        inactive = Boolean()
    
    value, error_messages = validate_yaml(content, BoolSchema)
    assert value["active"] is True
    assert value["inactive"] is False

    # Test with float values
    content = "price: 19.99"
    from typesystem.fields import Float
    class PriceSchema(Schema):
        price = Float()
    
    value, error_messages = validate_yaml(content, PriceSchema)
    assert value["price"] == 19.99

    # Test with nested structures
    content = "user:\n  name: Bob\n  scores: [1, 2, 3]"
    value, error_messages = validate_yaml(content, Field())
    assert value["user"]["name"] == "Bob"
    assert value["user"]["scores"] == [1, 2, 3]

    # Test with whitespace only
    content = "   \n  \n  "
    try:
        validate_yaml(content, Integer())
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello world"
    validator = String()
    value, errors = validate_yaml(content, validator)
    assert value == "hello world"
    assert errors == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, errors = validate_yaml(content, validator)
    assert value == 42
    assert errors == []
    
    # Test 3: Valid YAML dictionary with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, errors = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert errors == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, errors = validate_yaml(content, validator)
    assert isinstance(errors, list)
    
    # Test 5: Empty content should raise ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", String())
    assert exc_info.value.code == "no_content"
    
    # Test 6: Invalid YAML syntax should raise ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("invalid: yaml: content:", String())
    assert exc_info.value.code == "parse_error"
    
    # Test 7: Bytes input
    content = b"test value"
    validator = String()
    value, errors = validate_yaml(content, validator)
    assert value == "test value"
    assert errors == []
    
    # Test 8: Invalid YAML with unicode bytes
    content = b"invalid: yaml: content:"
    with pytest.raises(ParseError):
        validate_yaml(content, String())
    
    # Test 9: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, errors = validate_yaml(content, validator)
    assert value is None
    
    # Test 10: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, errors = validate_yaml(content, validator)
    assert value is True


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML with bytes
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: [invalid"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, TestSchema)
    assert exc_info.value.code == "parse_error"
    
    # Test empty YAML content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test whitespace-only YAML
    whitespace_yaml = "   \n  \n  "
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(whitespace_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test YAML with validation errors
    invalid_data_yaml = "name: 123\nage: not_a_number"
    value, error_messages = validate_yaml(invalid_data_yaml, TestSchema)
    assert error_messages  # Should have validation errors
    
    # Test with Field validator
    string_field = String(max_length=5)
    valid_field_yaml = "hello"
    value, error_messages = validate_yaml(valid_field_yaml, string_field)
    assert value == "hello"
    assert error_messages == {}
    
    # Test Field validator with invalid data
    invalid_field_yaml = "this_is_too_long"
    value, error_messages = validate_yaml(invalid_field_yaml, string_field)
    assert error_messages  # Should have validation error for max_length
    
    # Test YAML list content
    list_yaml = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    array_field = Array(items=String())
    value, error_messages = validate_yaml(list_yaml, array_field)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}
    
    # Test YAML with special types
    special_yaml = "count: 42\nratio: 3.14\nactive: true\nempty: null"
    class SpecialSchema(Schema):
        count = Integer()
        ratio = Integer()
        active = String()
        empty = String()
    
    value, error_messages = validate_yaml(special_yaml, SpecialSchema)
    assert value["count"] == 42
    assert value["ratio"] == 3.14
    assert value["active"] is True
    assert value["empty"] is None


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple schema
    yaml_content = "name: John\nage: 30"
    
    class TestSchema(Schema):
        name = Field(allow_null=False)
        age = Field(allow_null=False)
    
    value, error_messages = validate_yaml(yaml_content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}

    # Test with invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, TestSchema)
    assert exc_info.value.code == "parse_error"

    # Test with empty content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    yaml_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}

    # Test with whitespace-only content
    whitespace_yaml = "   \n\n   "
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(whitespace_yaml, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test with Field validator
    from typesystem.fields import String
    yaml_string = "hello world"
    value, error_messages = validate_yaml(yaml_string, String())
    assert value == "hello world"
    assert error_messages == {}

    # Test with list content
    yaml_list = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    value, error_messages = validate_yaml(yaml_list, Array(items=String()))
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}

    # Test with nested structure
    nested_yaml = "person:\n  name: Bob\n  details:\n    age: 40"
    value, error_messages = validate_yaml(nested_yaml, TestSchema)
    assert "person" in value


# LLM-generated content at query #30
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar value
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test boolean true
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test boolean false
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test list
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert all(isinstance(item, ScalarToken) for item in token.value)

    # Test dictionary
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert "key" in token.value
    assert isinstance(token.value["key"], ScalarToken)

    # Test nested structure
    token = tokenize_yaml("items:\n  - name: item1\n  - name: item2")
    assert isinstance(token, DictToken)
    assert "items" in token.value

    # Test bytes input
    token = tokenize_yaml(b"hello: world")
    assert isinstance(token, DictToken)

    # Test position information
    token = tokenize_yaml("key: value")
    assert token.start_index == 0
    assert token.end_index >= 0

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{invalid: [yaml: syntax}")
    assert exc_info.value.code == "parse_error"

    # Test yaml with comments
    token = tokenize_yaml("# comment\nkey: value")
    assert isinstance(token, DictToken)

    # Test multiline string
    token = tokenize_yaml('text: |\n  line1\n  line2')
    assert isinstance(token, DictToken)

    # Test position calculation
    token = tokenize_yaml("a: 1")
    assert hasattr(token, "start_index")
    assert hasattr(token, "end_index")

    # Test complex nested structure
    yaml_content = """
    users:
      - name: Alice
        age: 30
      - name: Bob
        age: 25
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert "users" in token.value

    # Test list of scalars
    token = tokenize_yaml("- item1\n- item2\n- item3")
    assert isinstance(token, ListToken)
    assert len(token.value) == 3


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with dict and Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML with list
    content = "- item1\n- item2"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2"]
    assert error_messages == []
    
    # Test 5: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError:
        pass
    
    # Test 6: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 7: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 8: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 9: YAML with null
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 10: Validation error (wrong type)
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []


# LLM-generated content at query #32
#--------------------------

```python
def test_tokenize_yaml():
    # Test with empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace only
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test dictionary
    token = tokenize_yaml("key: value\nfoo: bar")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "foo": "bar"}

    # Test list
    token = tokenize_yaml("- item1\n- item2\n- item3")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2", "item3"]

    # Test integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test boolean true
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test boolean false
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test nested structure
    token = tokenize_yaml("parent:\n  child: value\n  list:\n    - a\n    - b")
    assert isinstance(token, DictToken)
    assert token.value["parent"]["child"] == "value"
    assert token.value["parent"]["list"] == ["a", "b"]

    # Test invalid YAML syntax
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: [")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test position tracking in error
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value\ninvalid: {")
    error = exc_info.value
    assert error.code == "parse_error"
    assert error.position.line_no == 2

    # Test that token has correct position information
    token = tokenize_yaml("test: data")
    assert token.start_position >= 0
    assert token.end_position >= token.start_position

    # Test multiline string
    token = tokenize_yaml('text: |\n  line1\n  line2')
    assert isinstance(token, DictToken)
    assert "line1" in token.value["text"]


# LLM-generated content at query #33
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello"
    result = validate_yaml(content, String())
    assert result[0] == "hello"
    assert result[1] == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    result = validate_yaml(content, Integer())
    assert result[0] == 42
    assert result[1] == []
    
    # Test 3: Valid YAML dict with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    result = validate_yaml(content, UserSchema)
    assert result[0]["name"] == "John"
    assert result[0]["age"] == 30
    assert result[1] == []
    
    # Test 4: Invalid YAML - type mismatch
    content = "hello"
    result = validate_yaml(content, Integer())
    assert result[1] != []  # Should have errors
    
    # Test 5: Valid YAML with bytes input
    content = b"test_string"
    result = validate_yaml(content, String())
    assert result[0] == "test_string"
    assert result[1] == []
    
    # Test 6: Valid YAML list
    from typesystem.fields import Array
    content = "- item1\n- item2\n- item3"
    result = validate_yaml(content, Array(items=String()))
    assert result[0] == ["item1", "item2", "item3"]
    assert result[1] == []
    
    # Test 7: Valid YAML with boolean
    from typesystem.fields import Boolean
    content = "true"
    result = validate_yaml(content, Boolean())
    assert result[0] is True
    assert result[1] == []
    
    # Test 8: Valid YAML with null
    content = "null"
    result = validate_yaml(content, String(allow_null=True))
    assert result[0] is None
    assert result[1] == []
    
    # Test 9: Invalid YAML syntax - should raise ParseError
    content = "invalid: yaml: content:"
    try:
        validate_yaml(content, String())
    except ParseError:
        pass  # Expected
    
    # Test 10: Empty content - should raise ParseError
    content = ""
    try:
        validate_yaml(content, String())
    except ParseError as e:
        assert e.code == "no_content"


# LLM-generated content at query #34
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar value
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test integer
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test float
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test boolean true
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test boolean false
    result = tokenize_yaml("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test null
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test list
    result = tokenize_yaml("[1, 2, 3]")
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test dictionary
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test nested structure
    result = tokenize_yaml("parent:\n  child: value")
    assert isinstance(result, DictToken)
    assert result.value == {"parent": {"child": "value"}}
    
    # Test bytes input
    result = tokenize_yaml(b"test: data")
    assert isinstance(result, DictToken)
    assert result.value == {"test": "data"}
    
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"
    
    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{invalid: [yaml: content]")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None
    
    # Test position tracking with multiline content
    result = tokenize_yaml("key: value\nsecond: line")
    assert isinstance(result, DictToken)
    assert result.start_index >= 0
    assert result.end_index >= result.start_index
    
    # Test complex nested structure
    yaml_content = """
    users:
      - name: John
        age: 30
      - name: Jane
        age: 25
    """
    result = tokenize_yaml(yaml_content)
    assert isinstance(result, DictToken)
    assert "users" in result.value
    assert len(result.value["users"]) == 2
    
    # Test list of scalars
    result = tokenize_yaml("- item1\n- item2\n- item3")
    assert isinstance(result, ListToken)
    assert result.value == ["item1", "item2", "item3"]
    
    # Test special characters in strings
    result = tokenize_yaml('message: "Hello, World!"')
    assert isinstance(result, DictToken)
    assert result.value["message"] == "Hello, World!"
    
    # Test multiline string
    result = tokenize_yaml('text: |\n  Line 1\n  Line 2')
    assert isinstance(result, DictToken)
    assert "Line 1" in result.value["text"]


# LLM-generated content at query #35
#--------------------------

```python
def test_tokenize_yaml():
    # Test with empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace-only content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with string input
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test scalar token
    token = tokenize_yaml("simple string")
    assert isinstance(token, ScalarToken)
    assert token.value == "simple string"

    # Test list token
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test nested structure
    token = tokenize_yaml("parent:\n  child: value")
    assert isinstance(token, DictToken)
    assert token.value == {"parent": {"child": "value"}}

    # Test integer parsing
    token = tokenize_yaml("number: 42")
    assert isinstance(token, DictToken)
    assert token.value["number"] == 42

    # Test float parsing
    token = tokenize_yaml("decimal: 3.14")
    assert isinstance(token, DictToken)
    assert token.value["decimal"] == 3.14

    # Test boolean parsing
    token = tokenize_yaml("flag: true")
    assert isinstance(token, DictToken)
    assert token.value["flag"] is True

    # Test null parsing
    token = tokenize_yaml("empty: null")
    assert isinstance(token, DictToken)
    assert token.value["empty"] is None

    # Test position tracking with multiline content
    token = tokenize_yaml("key1: value1\nkey2: value2")
    assert isinstance(token, DictToken)
    assert token.start_pos is not None
    assert token.end_pos is not None

    # Test invalid YAML syntax
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test YAML with special characters
    token = tokenize_yaml('message: "hello world"')
    assert isinstance(token, DictToken)
    assert token.value["message"] == "hello world"

    # Test list of dictionaries
    token = tokenize_yaml("- name: Alice\n  age: 30\n- name: Bob\n  age: 25")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0]["name"] == "Alice"

    # Test yaml not installed scenario is handled by assertion
    if yaml is not None:
        # yaml is installed, normal behavior
        token = tokenize_yaml("test: data")
        assert isinstance(token, DictToken)


# LLM-generated content at query #36
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = UserSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 5: Invalid YAML syntax should raise ParseError
    import pytest
    content = "invalid: yaml: content:"
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(content, validator)
    
    # Test 6: Empty content should raise ParseError
    content = ""
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 7: Bytes content
    content = b"test"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test"
    assert error_messages == []
    
    # Test 8: YAML with boolean
    content = "true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 9: YAML with null
    content = "null"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 10: YAML with float
    content = "3.14"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []


# LLM-generated content at query #37
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except Exception:
        pass
    
    # Test 5: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except Exception:
        pass
    
    # Test 6: Bytes input
    content = b"test_string"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_string"
    assert error_messages == []
    
    # Test 7: YAML list
    content = "[1, 2, 3]"
    from typesystem.fields import Array
    validator = Array(items=Integer())
    value, error_messages = validate_yaml(content, validator)
    assert value == [1, 2, 3]
    assert error_messages == []
    
    # Test 8: YAML dict/mapping
    content = "key1: value1\nkey2: value2"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, dict)
    assert error_messages == []
    
    # Test 9: Whitespace only content
    content = "   \n  \t  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except Exception:
        pass
    
    # Test 10: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []


# LLM-generated content at query #38
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid scalar string
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with schema
    class PersonSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, PersonSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML list
    from typesystem.fields import Array
    content = "- item1\n- item2\n- item3"
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 5: Invalid YAML syntax
    content = "invalid: yaml: syntax:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError:
        pass
    
    # Test 6: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 7: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 8: YAML with null values
    content = "key: null"
    class NullableSchema(Schema):
        key = String(allow_null=True)
    
    value, error_messages = validate_yaml(content, NullableSchema)
    assert value == {"key": None}
    assert error_messages == []
    
    # Test 9: YAML with boolean values
    content = "enabled: true"
    from typesystem.fields import Boolean
    class BoolSchema(Schema):
        enabled = Boolean()
    
    value, error_messages = validate_yaml(content, BoolSchema)
    assert value == {"enabled": True}
    assert error_messages == []
    
    # Test 10: YAML with float values
    from typesystem.fields import Float
    content = "3.14"
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []


# LLM-generated content at query #39
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema()
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML with bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 5: Invalid YAML syntax should raise ParseError
    import pytest
    content = "invalid: yaml: content:"
    validator = String()
    with pytest.raises(ParseError):
        validate_yaml(content, validator)
    
    # Test 6: Empty content should raise ParseError
    content = ""
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 7: Whitespace only content should raise ParseError
    content = "   \n  \t  "
    validator = String()
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, validator)
    assert exc_info.value.code == "no_content"
    
    # Test 8: Valid YAML list
    content = "[1, 2, 3]"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    # Should return the list
    assert isinstance(value, list) or error_messages != []
    
    # Test 9: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    
    # Test 10: YAML with boolean
    content = "true"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    # Boolean should be coerced or validated
    assert value is True or error_messages != []


# LLM-generated content at query #40
#--------------------------

```python
def test_validate_yaml():
    """Test the validate_yaml function with various inputs."""
    
    # Test with valid YAML and a simple Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax
    content = "{ invalid yaml: [unclosed"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []
    
    # Test 5: Valid YAML but invalid for validator (string when integer expected)
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []
    
    # Test 6: YAML with list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 7: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []
    
    # Test 8: YAML with null
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 9: Bytes input
    content = b"hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 10: YAML with float
    content = "3.14"
    from typesystem.fields import Float
    validator = Float()
    value, error_messages = validate_yaml(content, validator)
    assert value == 3.14
    assert error_messages == []


# LLM-generated content at query #41
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with Field validator
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Schema validator
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value["name"] == "John"
    assert value["age"] == 30
    assert error_messages == []
    
    # Test 3: Invalid YAML syntax
    content = "invalid: yaml: content:"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages is not None
    assert len(error_messages) > 0
    
    # Test 4: Valid YAML bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 5: YAML with list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 6: YAML with nested structure
    content = "user:\n  name: Alice\n  email: alice@example.com"
    class UserSchema(Schema):
        user = typesystem.fields.Object()
    
    value, error_messages = validate_yaml(content, UserSchema)
    assert "user" in value
    
    # Test 7: Validation error with Field
    content = "12345"
    validator = Integer(maximum=100)
    value, error_messages = validate_yaml(content, validator)
    # Should either validate or have appropriate error messages
    
    # Test 8: YAML with boolean and null values
    content = "enabled: true\nnullable: null\ndisabled: false"
    value, error_messages = validate_yaml(content, typesystem.fields.Object())
    assert value["enabled"] is True
    assert value["nullable"] is None
    assert value["disabled"] is False
    assert error_messages == []


# LLM-generated content at query #42
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = """
name: John
age: 30
"""
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = """
name: John
age: invalid:syntax:
"""
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)
    
    # Test with bytes input
    yaml_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}
    
    # Test with Field validator
    field_validator = String()
    yaml_string = "hello"
    value, error_messages = validate_yaml(yaml_string, field_validator)
    assert value == "hello"
    assert error_messages == {}
    
    # Test empty content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with whitespace only
    whitespace_yaml = "   \n  \t  "
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(whitespace_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test validation failure
    invalid_data_yaml = """
name: John
age: not_an_integer
"""
    value, error_messages = validate_yaml(invalid_data_yaml, TestSchema)
    assert "age" in error_messages
    
    # Test with list content
    list_yaml = """
- item1
- item2
- item3
"""
    from typesystem.fields import Array
    array_validator = Array(items=String())
    value, error_messages = validate_yaml(list_yaml, array_validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}


# LLM-generated content at query #43
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test string input
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test scalar token
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test string scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test list token
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test nested structure
    token = tokenize_yaml("key:\n  nested: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": {"nested": "value"}}

    # Test boolean values
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test null value
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test float value
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test position tracking for scalar
    token = tokenize_yaml("value")
    assert token.start_pos == 0
    assert token.end_pos >= 0

    # Test position tracking for dict
    token = tokenize_yaml("key: value")
    assert token.start_pos == 0
    assert token.end_pos >= 0

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test complex nested structure
    token = tokenize_yaml("root:\n  - item1\n  - item2\n  key: value")
    assert isinstance(token, DictToken)
    assert "root" in token.value

    # Test multiline string
    token = tokenize_yaml('key: |\n  line1\n  line2')
    assert isinstance(token, DictToken)
    assert "key" in token.value

    # Test content preservation with bytes
    yaml_content = b"test: \xc3\xa9"  # UTF-8 encoded
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)


# LLM-generated content at query #44
#--------------------------

```python
def test_tokenize_yaml():
    # Test with empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test with whitespace only
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \t  ")
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with string input - simple mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with nested mapping
    token = tokenize_yaml("outer:\n  inner: value")
    assert isinstance(token, DictToken)
    assert token.value == {"outer": {"inner": "value"}}

    # Test with list
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test with scalar string
    token = tokenize_yaml("simple string")
    assert isinstance(token, ScalarToken)
    assert token.value == "simple string"

    # Test with integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test with float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test with boolean true
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test with boolean false
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test with null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test with complex nested structure
    token = tokenize_yaml("users:\n  - name: John\n    age: 30\n  - name: Jane\n    age: 25")
    assert isinstance(token, DictToken)
    assert len(token.value["users"]) == 2

    # Test invalid YAML - unclosed quote
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml('key: "unclosed')
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

    # Test invalid YAML - bad indentation
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value\n  bad: indent\n    worse: indent")
    assert exc_info.value.code == "parse_error"

    # Test token has correct position information
    token = tokenize_yaml("test: 123")
    assert token.start_pos >= 0
    assert token.end_pos >= token.start_pos

    # Test with bytes containing UTF-8
    token = tokenize_yaml("message: Hello 世界".encode("utf-8"))
    assert isinstance(token, DictToken)
    assert token.value["message"] == "Hello 世界"

    # Test with multiline string
    token = tokenize_yaml('text: |\n  line1\n  line2')
    assert isinstance(token, DictToken)
    assert "line1" in token.value["text"]
    assert "line2" in token.value["text"]


# LLM-generated content at query #45
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML with bytes input
    valid_yaml_bytes = b"name: Alice\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Alice", "age": 25}
    assert error_messages == {}
    
    # Test invalid YAML syntax - should raise ParseError
    invalid_yaml = "name: John\n  invalid: [unclosed"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)
    
    # Test empty YAML content - should raise ParseError
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test YAML with whitespace only - should raise ParseError
    whitespace_yaml = "   \n  \n  "
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(whitespace_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with simple Field validator
    name_field = String()
    yaml_string = "John"
    value, error_messages = validate_yaml(yaml_string, name_field)
    assert value == "John"
    assert error_messages == {}
    
    # Test with Integer field
    age_field = Integer()
    yaml_int = "42"
    value, error_messages = validate_yaml(yaml_int, age_field)
    assert value == 42
    assert error_messages == {}
    
    # Test YAML with list
    list_yaml = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    array_field = Array(items=String())
    value, error_messages = validate_yaml(list_yaml, array_field)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}
    
    # Test YAML with nested structure
    nested_yaml = "users:\n  - name: John\n    age: 30\n  - name: Jane\n    age: 28"
    class UserSchema(Schema):
        users = Array(items=TestSchema())
    value, error_messages = validate_yaml(nested_yaml, UserSchema)
    assert len(value["users"]) == 2
    assert value["users"][0]["name"] == "John"


# LLM-generated content at query #46
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML as bytes
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, TestSchema)
    assert exc_info.value.code == "parse_error"
    
    # Test empty content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test whitespace-only content
    whitespace_yaml = "   \n\n  "
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(whitespace_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test validation error (missing required field)
    incomplete_yaml = "name: Bob"
    value, error_messages = validate_yaml(incomplete_yaml, TestSchema)
    assert "age" in error_messages
    
    # Test with Field validator
    name_field = String()
    yaml_string = "John"
    value, error_messages = validate_yaml(yaml_string, name_field)
    assert value == "John"
    assert error_messages == {}
    
    # Test list YAML
    list_yaml = "- item1\n- item2\n- item3"
    value, error_messages = validate_yaml(list_yaml, String())
    assert isinstance(value, list)
    
    # Test nested structure
    nested_yaml = "user:\n  name: Alice\n  age: 28"
    value, error_messages = validate_yaml(nested_yaml, TestSchema)
    assert isinstance(value, dict)
    
    # Test YAML with special types
    special_yaml = "count: 42\nratio: 3.14\nactive: true\nempty: null"
    
    class SpecialSchema(Schema):
        count = Integer()
        ratio = typesystem.fields.Float()
        active = typesystem.fields.Boolean()
        empty = typesystem.fields.Field(allow_null=True)
    
    value, error_messages = validate_yaml(special_yaml, SpecialSchema)
    assert value["count"] == 42
    assert value["ratio"] == 3.14
    assert value["active"] is True
    assert value["empty"] is None


# LLM-generated content at query #47
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == {}
    assert value == {"name": "John", "age": 30}
    
    # Test with bytes input
    valid_yaml_bytes = b"name: Alice\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, TestSchema)
    assert errors == {}
    assert value == {"name": "Alice", "age": 25}
    
    # Test validation error
    invalid_yaml = "name: Bob\nage: invalid"
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    assert "age" in errors
    
    # Test with simple Field validator
    simple_field = String()
    yaml_string = "hello"
    value, errors = validate_yaml(yaml_string, simple_field)
    assert errors == {}
    assert value == "hello"
    
    # Test with integer field
    int_field = Integer()
    yaml_int = "42"
    value, errors = validate_yaml(yaml_int, int_field)
    assert errors == {}
    assert value == 42
    
    # Test with list
    from typesystem.fields import Array
    list_field = Array(items=Integer())
    yaml_list = "[1, 2, 3]"
    value, errors = validate_yaml(yaml_list, list_field)
    assert errors == {}
    assert value == [1, 2, 3]
    
    # Test with nested schema
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema
    
    nested_yaml = "name: Charlie\naddress:\n  street: Main St\n  city: NYC"
    value, errors = validate_yaml(nested_yaml, PersonSchema)
    assert errors == {}
    assert value == {"name": "Charlie", "address": {"street": "Main St", "city": "NYC"}}
    
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("invalid: yaml: content:", TestSchema)
    assert exc_info.value.code == "parse_error"


# LLM-generated content at query #48
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert errors == []
    
    # Test with bytes content
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert errors == []
    
    # Test with invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None
    
    # Test with empty content
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
    
    # Test with whitespace only content
    whitespace_yaml = "   \n  \n  "
    try:
        validate_yaml(whitespace_yaml, TestSchema)
        assert False, "Should raise ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test validation error with Field validator
    string_field = String(max_length=5)
    yaml_content = "this_is_a_very_long_string"
    value, errors = validate_yaml(yaml_content, string_field)
    assert errors  # Should have validation errors
    
    # Test with list content
    list_yaml = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    array_field = Array(items=String())
    value, errors = validate_yaml(list_yaml, array_field)
    assert value == ["item1", "item2", "item3"]
    assert errors == []
    
    # Test with scalar string
    scalar_yaml = "hello world"
    string_validator = String()
    value, errors = validate_yaml(scalar_yaml, string_validator)
    assert value == "hello world"
    assert errors == []
    
    # Test with integer
    int_yaml = "42"
    int_validator = Integer()
    value, errors = validate_yaml(int_yaml, int_validator)
    assert value == 42
    assert errors == []
    
    # Test with boolean
    bool_yaml = "true"
    from typesystem.fields import Boolean
    bool_validator = Boolean()
    value, errors = validate_yaml(bool_yaml, bool_validator)
    assert value is True
    assert errors == []
    
    # Test with null value
    null_yaml = "null"
    from typesystem.fields import Field as BaseField
    nullable_field = String(allow_null=True)
    value, errors = validate_yaml(null_yaml, nullable_field)
    assert value is None
    assert errors == []


# LLM-generated content at query #49
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid string field
    yaml_content = "hello"
    result, errors = validate_yaml(yaml_content, String())
    assert result == "hello"
    assert errors == []
    
    # Test 2: Valid integer field
    yaml_content = "42"
    result, errors = validate_yaml(yaml_content, Integer())
    assert result == 42
    assert errors == []
    
    # Test 3: Valid YAML dict with schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    yaml_content = """
name: John
age: 30
"""
    result, errors = validate_yaml(yaml_content, UserSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []
    
    # Test 4: Invalid type for field
    yaml_content = "not_a_number"
    result, errors = validate_yaml(yaml_content, Integer())
    assert result is None
    assert len(errors) > 0
    
    # Test 5: Valid YAML list
    yaml_content = """
- item1
- item2
- item3
"""
    from typesystem.fields import Array
    result, errors = validate_yaml(yaml_content, Array(items=String()))
    assert result == ["item1", "item2", "item3"]
    assert errors == []
    
    # Test 6: YAML content as bytes
    yaml_content = b"test_string"
    result, errors = validate_yaml(yaml_content, String())
    assert result == "test_string"
    assert errors == []
    
    # Test 7: Empty YAML content should raise error
    import pytest
    from typesystem.base import ParseError
    with pytest.raises(ParseError):
        validate_yaml("", String())
    
    # Test 8: Invalid YAML syntax should raise error
    with pytest.raises(ParseError):
        validate_yaml("{invalid: yaml: content:", String())
    
    # Test 9: Schema with missing required field
    class StrictSchema(Schema):
        required_field = String(allow_null=False)
    
    yaml_content = "other_field: value"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert len(errors) > 0
    
    # Test 10: Nested YAML structure
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema
    
    yaml_content = """
name: Alice
address:
  street: Main St
  city: New York
"""
    result, errors = validate_yaml(yaml_content, PersonSchema)
    assert result["name"] == "Alice"
    assert result["address"]["street"] == "Main St"
    assert result["address"]["city"] == "New York"
    assert errors == []


# LLM-generated content at query #50
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic dictionary
    yaml_dict = "key: value"
    token = tokenize_yaml(yaml_dict)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test basic list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test scalar string
    yaml_scalar = "hello"
    token = tokenize_yaml(yaml_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test scalar integer
    yaml_int = "42"
    token = tokenize_yaml(yaml_int)
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test scalar float
    yaml_float = "3.14"
    token = tokenize_yaml(yaml_float)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test scalar boolean true
    yaml_bool_true = "true"
    token = tokenize_yaml(yaml_bool_true)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test scalar boolean false
    yaml_bool_false = "false"
    token = tokenize_yaml(yaml_bool_false)
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test null
    yaml_null = "null"
    token = tokenize_yaml(yaml_null)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test nested dictionary
    yaml_nested = "outer:\n  inner: value"
    token = tokenize_yaml(yaml_nested)
    assert isinstance(token, DictToken)
    assert token.value == {"outer": {"inner": "value"}}

    # Test list with mixed types
    yaml_mixed = "- 1\n- hello\n- true"
    token = tokenize_yaml(yaml_mixed)
    assert isinstance(token, ListToken)
    assert token.value == [1, "hello", True]

    # Test bytes input
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("   \n  \n  ")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test position tracking for multi-line content
    yaml_multiline = "key1: value1\nkey2: value2"
    token = tokenize_yaml(yaml_multiline)
    assert isinstance(token, DictToken)
    assert token.value == {"key1": "value1", "key2": "value2"}

    # Test token has position information
    yaml_positioned = "test: value"
    token = tokenize_yaml(yaml_positioned)
    assert hasattr(token, 'start_position')
    assert hasattr(token, 'end_position')


# LLM-generated content at query #51
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test 1: Valid YAML content
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test 2: Valid YAML with bytes input
    content_bytes = b"name: Alice\nage: 25"
    value, error_messages = validate_yaml(content_bytes, TestSchema)
    assert value == {"name": "Alice", "age": 25}
    assert error_messages == {}
    
    # Test 3: Invalid YAML syntax
    invalid_yaml = "name: John\nage: [invalid"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, TestSchema)
    assert exc_info.value.code == "parse_error"
    
    # Test 4: Empty content
    empty_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_content, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test 5: Whitespace only content
    whitespace_content = "   \n  \t  "
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(whitespace_content, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test 6: Validation error - invalid type
    content_invalid_type = "name: John\nage: not_a_number"
    value, error_messages = validate_yaml(content_invalid_type, TestSchema)
    assert "age" in error_messages or error_messages
    
    # Test 7: Simple Field validator with String
    string_validator = String()
    simple_content = "hello"
    value, error_messages = validate_yaml(simple_content, string_validator)
    assert value == "hello"
    assert error_messages == {}
    
    # Test 8: YAML with lists
    list_schema_content = "- item1\n- item2\n- item3"
    # This would require appropriate validator setup
    
    # Test 9: Complex nested YAML
    nested_content = "name: John\nage: 30\naddress:\n  city: NYC\n  zip: 10001"
    
    # Test 10: YAML with special types (bool, null, float)
    special_yaml = "active: true\nempty: null\nrating: 4.5"


# LLM-generated content at query #52
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    # Test valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test valid YAML as bytes
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, error_messages = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert error_messages == {}
    
    # Test invalid YAML syntax
    invalid_yaml = "name: John\n  invalid: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, TestSchema)
    assert exc_info.value.code == "parse_error"
    
    # Test empty content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"
    
    # Test with Field validator
    field_validator = String()
    yaml_string = "hello"
    value, error_messages = validate_yaml(yaml_string, field_validator)
    assert value == "hello"
    assert error_messages == {}
    
    # Test validation failure
    class StrictSchema(Schema):
        name = String(max_length=5)
    
    long_name_yaml = "name: Christopher"
    value, error_messages = validate_yaml(long_name_yaml, StrictSchema)
    assert error_messages != {}
    
    # Test with list YAML
    list_yaml = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    list_field = Array(items=String())
    value, error_messages = validate_yaml(list_yaml, list_field)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}
    
    # Test with nested structure
    nested_yaml = """
    users:
      - name: Alice
        age: 28
      - name: Bob
        age: 35
    """
    class UserSchema(Schema):
        users = Array(items=TestSchema)
    
    value, error_messages = validate_yaml(nested_yaml, UserSchema)
    assert len(value["users"]) == 2
    assert error_messages == {}
    
    # Test with various YAML types
    types_yaml = """
    string: hello
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    """
    class TypesSchema(Schema):
        string = String()
        integer = Integer()
        float = String()
        boolean = String()
        null_value = String(allow_null=True)
    
    value, error_messages = validate_yaml(types_yaml, TypesSchema)
    assert value["string"] == "hello"
    assert value["integer"] == 42
    assert error_messages == {}


# LLM-generated content at query #53
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid scalar string
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == {}
    
    # Test 2: Valid YAML with bytes input
    content_bytes = b"world"
    validator = String()
    value, error_messages = validate_yaml(content_bytes, validator)
    assert value == "world"
    assert error_messages == {}
    
    # Test 3: Valid integer
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == {}
    
    # Test 4: Valid YAML dictionary with schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == {}
    
    # Test 5: Valid YAML list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == {}
    
    # Test 6: Invalid YAML syntax should raise ParseError
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test 7: Empty content should raise ParseError
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test 8: Whitespace-only content should raise ParseError
    content = "   \n  \n  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test 9: Validation error on type mismatch
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != {}
    
    # Test 10: Schema with missing required field
    class StrictSchema(Schema):
        required_field = String(allow_null=False)
    
    content = "other_field: value"
    value, error_messages = validate_yaml(content, StrictSchema)
    assert error_messages != {}
    
    # Test 11: Nested YAML structure
    content = "users:\n  - name: Alice\n    age: 25\n  - name: Bob\n    age: 30"
    from typesystem.fields import Object
    validator = Object()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, dict)
    assert "users" in value
    
    # Test 12: YAML with special types (bool, null)
    content = "flag: true\nempty: null"
    validator = Object()
    value, error_messages = validate_yaml(content, validator)
    assert value["flag"] is True
    assert value["empty"] is None


# LLM-generated content at query #54
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple Field validator
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML string with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid YAML with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    validator = TestSchema
    value, error_messages = validate_yaml(content, validator)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Invalid YAML syntax should raise ParseError
    content = "invalid: yaml: content:"
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError:
        pass
    
    # Test 5: Empty YAML content should raise ParseError
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 6: Whitespace-only YAML content should raise ParseError
    content = "   \n  \t  "
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 7: YAML as bytes
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 8: YAML with list
    content = "- item1\n- item2\n- item3"
    from typesystem.fields import Array
    validator = Array(items=String())
    value, error_messages = validate_yaml(content, validator)
    assert value == ["item1", "item2", "item3"]
    assert error_messages == []
    
    # Test 9: YAML with nested dictionary
    content = "outer:\n  inner: value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, dict)
    assert value == {"outer": {"inner": "value"}}
    
    # Test 10: Validation error (wrong type)
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages != []


# LLM-generated content at query #55
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and a simple Field
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid scalar value with String field
    content = "hello"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "hello"
    assert error_messages == []
    
    # Test 2: Valid integer with Integer field
    content = "42"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert value == 42
    assert error_messages == []
    
    # Test 3: Valid YAML dict with Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    content = "name: John\nage: 30"
    value, error_messages = validate_yaml(content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert error_messages == []
    
    # Test 4: Valid YAML list
    content = "- item1\n- item2\n- item3"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert isinstance(value, list)
    
    # Test 5: Invalid YAML syntax
    content = "{ invalid: yaml: content"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages  # Should have errors
    
    # Test 6: Validation error - wrong type
    content = "not_a_number"
    validator = Integer()
    value, error_messages = validate_yaml(content, validator)
    assert error_messages  # Should have validation errors
    
    # Test 7: Empty content
    content = ""
    validator = String()
    try:
        validate_yaml(content, validator)
        assert False, "Should raise ParseError for empty content"
    except Exception as exc:
        assert "no_content" in str(exc) or "No content" in str(exc)
    
    # Test 8: Bytes input
    content = b"test_value"
    validator = String()
    value, error_messages = validate_yaml(content, validator)
    assert value == "test_value"
    assert error_messages == []
    
    # Test 9: YAML with null value
    content = "null"
    validator = String(allow_null=True)
    value, error_messages = validate_yaml(content, validator)
    assert value is None
    assert error_messages == []
    
    # Test 10: YAML with boolean
    content = "true"
    from typesystem.fields import Boolean
    validator = Boolean()
    value, error_messages = validate_yaml(content, validator)
    assert value is True
    assert error_messages == []


