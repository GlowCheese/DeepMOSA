####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and valid schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test with invalid YAML
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test with valid YAML but invalid schema
    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test with empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test with bytes content
    yaml_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}, "items": ["item1", "item2"]}
    assert errors == []

    # Test with positional error reporting
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].position.line_no == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: invalid_int"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: 123\nage: thirty"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) == 2

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with bytes input
    bytes_content = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_content, TestSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []


# LLM-generated content at query #3
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML string
    yaml_str = "key: value"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty YAML string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"

    # Test YAML with list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML with nested structures
    nested_yaml = "outer:\n  inner: value"
    token = tokenize_yaml(nested_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"outer": {"inner": "value"}}


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with correct schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: thirty"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test valid YAML with incorrect schema
    yaml_content = "name: John\nage: 30"
    class WrongSchema(Schema):
        name = Field(int)  # Expecting int, but got str
    result, errors = validate_yaml(yaml_content, WrongSchema)
    assert result is None
    assert len(errors) > 0

    # Test empty YAML
    with pytest.raises(ParseError):
        validate_yaml("", TestSchema)

    # Test YAML with bytes content
    yaml_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user: {name: John, age: 30}\nitems: [1, 2, 3]"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}, "items": [1, 2, 3]}
    assert errors == []

    # Test YAML with special characters
    yaml_content = "name: 'John Doe'\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John Doe", "age": 30}
    assert errors == []


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str, required=True)
        age = Field(int, required=True)

    yaml_content = """
name: John Doe
age: 30
"""
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John Doe", "age": 30}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = """
name: John Doe
age: thirty
"""
    result, errors = validate_yaml(invalid_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test missing required field
    incomplete_yaml = """
name: John Doe
"""
    result, errors = validate_yaml(incomplete_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test empty YAML
    empty_yaml = ""
    result, errors = validate_yaml(empty_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test bytes input
    bytes_yaml = b"""
name: Jane Doe
age: 25
"""
    result, errors = validate_yaml(bytes_yaml, TestSchema)
    assert result == {"name": "Jane Doe", "age": 25}
    assert errors == []

    # Test nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict, required=True)
        settings = Field(list, required=True)

    nested_yaml = """
user:
  name: Alice
  role: admin
settings:
  - dark_mode: true
  - notifications: false
"""
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "role": "admin"},
        "settings": [{"dark_mode": True}, {"notifications": False}]
    }
    assert errors == []


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    schema = Schema(fields={"name": str, "age": int})
    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (syntax error)
    content = "name: John\nage: thirty"
    with pytest.raises(ParseError):
        validate_yaml(content, schema)

    # Test YAML with validation errors
    content = "name: John\nage: -5"
    result, errors = validate_yaml(content, schema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, schema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    schema = Schema(fields={"user": {"name": str, "roles": list}})
    content = "user:\n  name: Admin\n  roles:\n    - admin\n    - user"
    result, errors = validate_yaml(content, schema)
    assert result == {"user": {"name": "Admin", "roles": ["admin", "user"]}}
    assert errors == []

    # Test YAML with bytes input
    content = b"name: John\nage: 30"
    result, errors = validate_yaml(content, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: :invalid"
    with pytest.raises(ParseError):
        validate_yaml(content, TestSchema)

    # Test YAML that fails schema validation
    class TestSchemaWithRequired(Schema):
        name = Field(str, required=True)
        age = Field(int, required=True)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchemaWithRequired)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: John
        age: 30
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with type validation
    class TypeSchema(Schema):
        count = Field(int)
        active = Field(bool)
        ratio = Field(float)

    content = """
    count: 42
    active: true
    ratio: 3.14
    """
    result, errors = validate_yaml(content, TypeSchema)
    assert result == {"count": 42, "active": True, "ratio": 3.14}
    assert errors == []

    # Test YAML with bytes content
    content_bytes = b"name: John"
    result, errors = validate_yaml(content_bytes, TestSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML that doesn't match schema
    yaml_content = "age: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict, properties={
            "name": Field(str),
            "age": Field(int)
        })

    yaml_content = """
    user:
        name: Alice
        age: 25
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 25}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(str))

    yaml_content = """
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    schema = Schema(fields={"name": str, "age": int})
    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    schema = Schema(fields={"name": str})
    content = "name: [John"  # Invalid YAML
    with pytest.raises(ParseError):
        validate_yaml(content, schema)

    # Test valid YAML with validation errors
    schema = Schema(fields={"name": str, "age": int})
    content = "name: John\nage: thirty"  # Invalid age type
    result, errors = validate_yaml(content, schema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    schema = Schema(fields={"name": str})
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, schema)
    assert exc_info.value.code == "no_content"

    # Test YAML with extra fields not in schema
    schema = Schema(fields={"name": str})
    content = "name: John\nextra: field"
    result, errors = validate_yaml(content, schema)
    assert result == {"name": "John"}
    assert len(errors) == 0  # Assuming schema allows extra fields by default

    # Test YAML with missing required field
    schema = Schema(fields={"name": str, "age": int})
    content = "name: John"
    result, errors = validate_yaml(content, schema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test YAML with nested structures
    schema = Schema(fields={"user": {"name": str, "age": int}})
    content = "user:\n  name: John\n  age: 30"
    result, errors = validate_yaml(content, schema)
    assert result == {"user": {"name": "John", "age": 30}}
    assert errors == []

    # Test YAML with list of items
    schema = Schema(fields={"tags": [str]})
    content = "tags:\n  - python\n  - testing"
    result, errors = validate_yaml(content, schema)
    assert result == {"tags": ["python", "testing"]}
    assert errors == []


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML with a simple schema
    yaml_content = "name: 123"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "type_error"

    # Test empty YAML content
    yaml_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test invalid YAML syntax
    yaml_content = "name: [John"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, SimpleSchema)
    assert exc_info.value.code == "parse_error"

    # Test nested YAML with a nested schema
    class NestedSchema(Schema):
        user = Field(dict)

    yaml_content = "user:\n  name: John\n  age: 30"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}}
    assert errors == []

    # Test YAML with a list
    class ListSchema(Schema):
        items = Field(list)

    yaml_content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with bytes content
    yaml_content = b"name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test YAML with multiple errors
    class ComplexSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: 123\nage: abc"
    result, errors = validate_yaml(yaml_content, ComplexSchema)
    assert result is None
    assert len(errors) == 2


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: 30\ninvalid: [unclosed"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML that fails validation
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Bob"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with type validation
    class TypedSchema(Schema):
        count = Field(int)
        active = Field(bool)
        price = Field(float)

    yaml_content = """
    count: 42
    active: true
    price: 19.99
    """
    result, errors = validate_yaml(yaml_content, TypedSchema)
    assert result == {"count": 42, "active": True, "price": 19.99}
    assert errors == []

    # Test YAML with null values
    class NullableSchema(Schema):
        value = Field(str, allow_null=True)

    yaml_content = "value: null"
    result, errors = validate_yaml(yaml_content, NullableSchema)
    assert result == {"value": None}
    assert errors == []

    # Test YAML with bytes input
    yaml_bytes = b"name: John"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test YAML with positional error reporting
    class PositionalSchema(Schema):
        name = Field(str)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, PositionalSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].position is not None


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = """
    name: John Doe
    age: 30
    """
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert errors == []
    assert result == {"name": "John Doe", "age": 30}

    # Test invalid YAML (syntax error)
    invalid_yaml = """
    name: John Doe
    age: 30
    invalid: [unclosed
    """
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = """
    name: 123
    age: not_a_number
    """
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert len(errors) == 2
    assert any("name" in error.msg for error in errors)
    assert any("age" in error.msg for error in errors)

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = """
    user:
        name: Jane
        age: 25
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert result["user"]["name"] == "Jane"
    assert result["items"] == ["apple", "banana"]

    # Test YAML with bytes content
    bytes_content = b"name: Test\nage: 20"
    result, errors = validate_yaml(bytes_content, TestSchema)
    assert errors == []
    assert result == {"name": "Test", "age": 20}


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test valid YAML mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test valid YAML sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 12

    # Test YAML with different data types
    token = tokenize_yaml("int: 1\nfloat: 1.5\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.5, "bool": True, "null": None}

    # Test invalid YAML
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("invalid: yaml: content")
    assert excinfo.value.code == "parse_error"

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test nested structures
    token = tokenize_yaml("list:\n  - item1\n  - item2\nnested:\n  key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"list": ["item1", "item2"], "nested": {"key": "value"}}


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John Doe"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John Doe"}
    assert errors == []

    # Test invalid YAML with a simple schema
    yaml_content = "name: 123"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test YAML with missing required field
    yaml_content = "age: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test empty YAML content
    yaml_content = ""
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test invalid YAML syntax
    yaml_content = "name: John Doe\nage: 30\ninvalid: [unclosed"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = """
user:
  name: John Doe
  age: 30
items:
  - item1
  - item2
"""
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "John Doe", "age": 30},
        "items": ["item1", "item2"]
    }
    assert errors == []

    # Test YAML with type validation
    class TypedSchema(Schema):
        count = Field(int)
        active = Field(bool)
        ratio = Field(float)

    yaml_content = """
count: 42
active: true
ratio: 3.14
"""
    result, errors = validate_yaml(yaml_content, TypedSchema)
    assert result == {
        "count": 42,
        "active": True,
        "ratio": 3.14
    }
    assert errors == []

    # Test YAML with null values
    yaml_content = """
name: null
"""
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": None}
    assert errors == []


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: invalid_int"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, SimpleSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error.msg for error in errors)

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        tags = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 25\n tags:\n  - python\n  - testing"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 25}, "tags": ["python", "testing"]}
    assert errors == []

    # Test YAML with bytes input
    bytes_yaml = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_yaml, SimpleSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str, min_length=1)
        age = Field(int, minimum=0)

    yaml_content = """
    name: John Doe
    age: 30
    """
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert errors == []
    assert result["name"] == "John Doe"
    assert result["age"] == 30

    # Test invalid YAML (syntax error)
    invalid_yaml = """
    name: John Doe
    age: 30
    invalid: [unclosed bracket
    """
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test valid YAML with validation errors
    invalid_data_yaml = """
    name: J
    age: -5
    """
    result, errors = validate_yaml(invalid_data_yaml, TestSchema)
    assert len(errors) == 2
    assert any("min_length" in str(error) for error in errors)
    assert any("minimum" in str(error) for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert result["user"]["name"] == "Alice"
    assert result["items"] == ["apple", "banana"]

    # Test YAML with different scalar types
    scalar_yaml = """
    bool_val: true
    float_val: 3.14
    null_val: null
    """
    class ScalarSchema(Schema):
        bool_val = Field(bool)
        float_val = Field(float)
        null_val = Field(type=None)

    result, errors = validate_yaml(scalar_yaml, ScalarSchema)
    assert errors == []
    assert result["bool_val"] is True
    assert result["float_val"] == 3.14
    assert result["null_val"] is None


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: 30\ninvalid: yaml: content"
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test YAML validation error
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with list content
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - item1\n  - item2"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": ["item1", "item2"]}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: Alice\n  age: 25"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 25}}
    assert errors == []

    # Test YAML with bytes content
    content = b"name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test YAML with multiple validation errors
    class ComplexSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=18)

    content = "name: Bob\nage: 15"
    result, errors = validate_yaml(content, ComplexSchema)
    assert result is None
    assert len(errors) == 2
    assert {e.code for e in errors} == {"min_length", "minimum"}


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, TestSchema)
    assert "parse_error" in exc_info.value.code

    # Test valid YAML with validation errors
    class TestSchemaWithValidation(Schema):
        age = Field(int, minimum=0)

    content_with_error = "age: -5"
    result, errors = validate_yaml(content_with_error, TestSchemaWithValidation)
    assert result == {"age": -5}
    assert len(errors) == 1
    assert "minimum" in errors[0].message

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert "no_content" in exc_info.value.code

    # Test YAML with bytes content
    byte_content = b"name: Alice"
    result, errors = validate_yaml(byte_content, TestSchema)
    assert result == {"name": "Alice"}
    assert errors == []

    # Test nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_content = """
    user:
        id: 1
        name: Bob
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(nested_content, NestedSchema)
    assert result == {
        "user": {"id": 1, "name": "Bob"},
        "items": ["apple", "banana"]
    }
    assert errors == []


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and valid schema
    yaml_content = "name: John\nage: 30"
    schema = Schema({"name": str, "age": int})
    result = validate_yaml(yaml_content, schema)
    assert result == ({"name": "John", "age": 30}, [])

    # Test with invalid YAML
    invalid_yaml = "name: John\nage: invalid"
    schema = Schema({"name": str, "age": int})
    result = validate_yaml(invalid_yaml, schema)
    assert result[1] != []

    # Test with empty YAML
    empty_yaml = ""
    schema = Schema({"name": str, "age": int})
    try:
        validate_yaml(empty_yaml, schema)
    except ParseError as e:
        assert e.code == "no_content"

    # Test with YAML that doesn't match schema
    yaml_content = "name: John"
    schema = Schema({"name": str, "age": int})
    result = validate_yaml(yaml_content, schema)
    assert result[1] != []

    # Test with bytes input
    yaml_bytes = b"name: John\nage: 30"
    schema = Schema({"name": str, "age": int})
    result = validate_yaml(yaml_bytes, schema)
    assert result == ({"name": "John", "age": 30}, [])

    # Test with nested YAML
    yaml_content = "user:\n  name: John\n  age: 30"
    schema = Schema({"user": {"name": str, "age": int}})
    result = validate_yaml(yaml_content, schema)
    assert result == ({"user": {"name": "John", "age": 30}}, [])


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML (parse error)
    content = "name: John\ninvalid: yaml: content"
    with pytest.raises(ParseError):
        validate_yaml(content, TestSchema)

    # Test valid YAML with validation errors
    class TestSchemaWithValidation(Schema):
        name = Field(str, min_length=5)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchemaWithValidation)
    assert result == {"name": "John"}
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: John\n  age: 30"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with various scalar types
    class ScalarSchema(Schema):
        age = Field(int)
        price = Field(float)
        active = Field(bool)
        description = Field(str, allow_null=True)

    content = "age: 25\nprice: 19.99\nactive: true\ndescription: null"
    result, errors = validate_yaml(content, ScalarSchema)
    assert result == {"age": 25, "price": 19.99, "active": True, "description": None}
    assert errors == []

    # Test YAML with bytes content
    content = b"name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    yaml_content = "key: value"
    schema = Schema(fields={"key": Field(str)})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"key": "value"}
    assert errors == []

    # Test valid YAML with a list schema
    yaml_content = "- item1\n- item2"
    schema = Schema(fields={"items": Field(list)})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == ["item1", "item2"]
    assert errors == []

    # Test invalid YAML syntax
    yaml_content = "key: [unclosed"
    schema = Schema(fields={"key": Field(list)})
    with pytest.raises(ParseError):
        validate_yaml(yaml_content, schema)

    # Test YAML validation error
    yaml_content = "key: 123"
    schema = Schema(fields={"key": Field(str)})
    result, errors = validate_yaml(yaml_content, schema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "type_error"

    # Test empty YAML content
    yaml_content = ""
    schema = Schema(fields={"key": Field(str)})
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, schema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    yaml_content = "outer:\n  inner: value"
    schema = Schema(fields={"outer": Field(dict)})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"outer": {"inner": "value"}}
    assert errors == []

    # Test YAML with different data types
    yaml_content = "int_val: 42\nfloat_val: 3.14\nbool_val: true\nnull_val: null"
    schema = Schema(fields={
        "int_val": Field(int),
        "float_val": Field(float),
        "bool_val": Field(bool),
        "null_val": Field(type(None))
    })
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None
    }
    assert errors == []

    # Test YAML with bytes input
    yaml_content = b"key: value"
    schema = Schema(fields={"key": Field(str)})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"key": "value"}
    assert errors == []


# LLM-generated content at query #22
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML mapping
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML sequence
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML scalar
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test YAML with different data types
    yaml_content = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test invalid YAML
    yaml_content = "key: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"

    # Test bytes content
    yaml_content = b"key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test nested structures
    yaml_content = """
    outer:
      inner:
        - item1
        - item2
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "outer": {
            "inner": ["item1", "item2"]
        }
    }


# LLM-generated content at query #23
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test integer scalar
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1

    # Test float scalar
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3

    # Test boolean scalar
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    # Test null scalar
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test list
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == 0
    assert token.end == 8

    # Test dict
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == 0
    assert token.end == 10

    # Test nested structures
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.start == 0
    assert token.end == 18

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"

    # Test bytes content
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"


# LLM-generated content at query #24
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML string
    yaml_str = "key: value"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test invalid YAML
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"

    # Test YAML list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML scalar
    yaml_scalar = "scalar_value"
    token = tokenize_yaml(yaml_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"

    # Test YAML with different data types
    yaml_types = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_types)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }


# LLM-generated content at query #25
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML string
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test invalid YAML
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"

    # Test YAML with list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML with nested structures
    nested_yaml = "outer:\n  inner: value"
    token = tokenize_yaml(nested_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"outer": {"inner": "value"}}


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and valid schema
    yaml_content = "name: John\nage: 30"
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test with invalid YAML
    invalid_yaml = "name: John\nage: thirty"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test with valid YAML but invalid schema
    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test with empty YAML
    empty_yaml = ""
    with pytest.raises(ParseError):
        validate_yaml(empty_yaml, TestSchema)

    # Test with bytes input
    yaml_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: thirty"  # age should be int
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"  # negative age
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert len(errors) > 0
    assert "age" in errors[0].loc

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with bytes content
    yaml_bytes = b"name: Jane\nage: 25"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert errors == []
    assert result == {"name": "Jane", "age": 25}

    # Test nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user: {name: Alice, age: 28}\nitems: [1, 2, 3]"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert result["user"]["name"] == "Alice"
    assert result["items"] == [1, 2, 3]

    # Test YAML with special characters
    special_yaml = "name: 'John \"The Boss\" Doe'\nage: 40"
    result, errors = validate_yaml(special_yaml, TestSchema)
    assert errors == []
    assert result["name"] == 'John "The Boss" Doe'


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result = validate_yaml(content, SimpleSchema)
    assert result == ({"name": "John"}, [])

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, SimpleSchema)
    assert "parse_error" in str(exc_info.value)

    # Test valid YAML with validation error
    class StrictSchema(Schema):
        age = Field(int, minimum=18)

    content = "age: 15"
    result = validate_yaml(content, StrictSchema)
    assert result[1][0].code == "minimum"

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert "no_content" in str(exc_info.value)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, properties={
            "name": Field(str),
            "age": Field(int)
        })

    content = """
    user:
      name: Alice
      age: 30
    """
    result = validate_yaml(content, NestedSchema)
    assert result[0]["user"]["name"] == "Alice"

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    content = "items: [1, 2, 3]"
    result = validate_yaml(content, ListSchema)
    assert result[0]["items"] == [1, 2, 3]

    # Test bytes input
    content_bytes = b"name: Bob"
    result = validate_yaml(content_bytes, SimpleSchema)
    assert result == ({"name": "Bob"}, [])


# LLM-generated content at query #29
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML mapping
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML sequence
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML scalar
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test YAML with different data types
    yaml_content = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test invalid YAML
    yaml_content = "key: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index == 10

    # Test bytes content
    yaml_content = b"key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test nested structures
    yaml_content = """
    outer:
      inner:
        - item1
        - item2
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "outer": {
            "inner": ["item1", "item2"]
        }
    }


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: :invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML that fails schema validation
    class StrictSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: twenty"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = """
    user:
        name: Alice
        age: 30
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with bytes content
    byte_content = b"name: Bob"
    result, errors = validate_yaml(byte_content, SimpleSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str, required=True)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML (syntax error)
    content = "name: John\ninvalid: yaml: content"
    with pytest.raises(ParseError):
        validate_yaml(content, TestSchema)

    # Test YAML with validation errors
    class TestSchemaWithValidation(Schema):
        name = Field(str, required=True)
        age = Field(int, required=True)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchemaWithValidation)
    assert result == {"name": "John"}
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, required=True)
        items = Field(list, required=True)

    content = """
    user:
        name: John
        age: 30
    items:
        - item1
        - item2
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["item1", "item2"]
    }
    assert errors == []

    # Test YAML with bytes content
    content = b"name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #32
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    yaml_content = "name: John Doe"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John Doe"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: [John Doe"  # Missing closing bracket
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, TestSchema)
    assert "parse_error" in str(exc_info.value)

    # Test YAML with validation errors
    yaml_content = "name: 123"  # Invalid type for name field
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("name" in error.message for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert "no_content" in str(exc_info.value)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = """
    user:
        name: Alice
        age: 30
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: Bob"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #33
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    yaml_content = "name: John\nage: 30"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid_yaml_syntax: 30"
    schema = Schema(fields={"name": str, "age": int})
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, schema)

    # Test YAML with validation errors
    yaml_content = "name: John\nage: thirty"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_content, schema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    empty_yaml = ""
    schema = Schema(fields={"name": str})
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, schema)
    assert exc_info.value.code == "no_content"

    # Test YAML with bytes input
    yaml_bytes = b"name: Jane\nage: 25"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_bytes, schema)
    assert result == {"name": "Jane", "age": 25}
    assert errors == []

    # Test YAML with nested structures
    yaml_content = "user:\n  name: Alice\n  age: 28"
    schema = Schema(fields={"user": dict})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"user": {"name": "Alice", "age": 28}}
    assert errors == []

    # Test YAML with list
    yaml_content = "items:\n  - apple\n  - banana"
    schema = Schema(fields={"items": list})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []


# LLM-generated content at query #34
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str, required=True)
        age = Field(int, required=True)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result == {"name": "John", "age": -5}
    assert len(errors) > 0

    # Test empty YAML
    with pytest.raises(ParseError):
        validate_yaml("", TestSchema)

    # Test YAML with missing required field
    yaml_missing_field = "name: John"
    result, errors = validate_yaml(yaml_missing_field, TestSchema)
    assert result == {"name": "John"}
    assert len(errors) > 0

    # Test YAML with extra fields
    class StrictSchema(Schema):
        name = Field(str, required=True)

    yaml_extra_fields = "name: John\nextra: field"
    result, errors = validate_yaml(yaml_extra_fields, StrictSchema)
    assert result == {"name": "John"}
    assert len(errors) > 0

    # Test YAML bytes input
    yaml_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #35
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and valid schema
    yaml_content = "key: value"
    class TestSchema(Schema):
        key = Field(str)
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"key": "value"}
    assert errors == []

    # Test with invalid YAML
    invalid_yaml = "key: [unclosed"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test with valid YAML but invalid schema
    yaml_content = "key: 123"
    class StrictSchema(Schema):
        key = Field(str)
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test with empty YAML
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test with nested YAML structure
    nested_yaml = """
    outer:
      inner: value
    """
    class NestedSchema(Schema):
        outer = Field(dict)
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"outer": {"inner": "value"}}
    assert errors == []

    # Test with list in YAML
    list_yaml = "items: [1, 2, 3]"
    class ListSchema(Schema):
        items = Field(list)
    result, errors = validate_yaml(list_yaml, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test with bytes input
    bytes_yaml = b"key: value"
    result, errors = validate_yaml(bytes_yaml, TestSchema)
    assert result == {"key": "value"}
    assert errors == []


# LLM-generated content at query #36
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John Doe"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John Doe"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: [John Doe"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, SimpleSchema)
    assert "parse_error" in str(exc_info.value)

    # Test YAML that fails schema validation
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Bob"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].code

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert "no_content" in str(exc_info.value)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = """
user:
  name: Alice
  age: 30
items:
  - apple
  - banana
"""
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with different data types
    class TypeSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)
        tags = Field(list, default=[])

    yaml_content = """
count: 42
price: 19.99
active: true
tags:
  - python
  - yaml
"""
    result, errors = validate_yaml(yaml_content, TypeSchema)
    assert result == {
        "count": 42,
        "price": 19.99,
        "active": True,
        "tags": ["python", "yaml"]
    }
    assert errors == []


# LLM-generated content at query #37
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML that fails schema validation
    yaml_content = "age: 25"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "name" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict, properties={
            "name": Field(str),
            "age": Field(int)
        })

    yaml_content = """
    user:
        name: Alice
        age: 30
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(str))

    yaml_content = """
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with bytes input
    yaml_bytes = b"name: Bob"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #38
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_content = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(invalid_content, SimpleSchema)

    # Test YAML validation error
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    short_name_content = "name: Bob"
    result, errors = validate_yaml(short_name_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].code

    # Test empty YAML content
    empty_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, properties={
            "name": Field(str),
            "age": Field(int)
        })

    nested_content = """
    user:
        name: Alice
        age: 30
    """
    result, errors = validate_yaml(nested_content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    list_content = "items: [1, 2, 3]"
    result, errors = validate_yaml(list_content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with validation error in list
    invalid_list_content = "items: [1, two, 3]"
    result, errors = validate_yaml(invalid_list_content, ListSchema)
    assert result is None
    assert len(errors) == 1


# LLM-generated content at query #39
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    content = "name: John\nage: invalid"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result is None
    assert len(errors) > 0

    # Test empty YAML
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) > 0

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: Alice
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice"},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with type errors
    class TypedSchema(Schema):
        count = Field(int)

    content = "count: not_a_number"
    result, errors = validate_yaml(content, TypedSchema)
    assert result is None
    assert len(errors) > 0

    # Test YAML with missing required field
    class RequiredSchema(Schema):
        required_field = Field(str, required=True)

    content = "other_field: value"
    result, errors = validate_yaml(content, RequiredSchema)
    assert result is None
    assert len(errors) > 0


# LLM-generated content at query #40
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML that doesn't match schema
    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list)

    yaml_content = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        settings = Field(dict)

    yaml_content = "user:\n  name: Alice\n  age: 25\nsettings:\n  theme: dark"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "settings": {"theme": "dark"}
    }
    assert errors == []

    # Test YAML with bytes input
    yaml_bytes = b"name: Bob\nage: 40"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []


# LLM-generated content at query #41
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test valid YAML scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test valid YAML mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test valid YAML sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 12

    # Test valid YAML with different types
    token = tokenize_yaml("int: 1\nfloat: 1.5\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.5, "bool": True, "null": None}

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 22

    # Test bytes content
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test YAML with special characters
    token = tokenize_yaml("key: 'value with spaces'")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value with spaces"}

    # Test nested YAML structures
    token = tokenize_yaml("outer:\n  inner:\n    - item1\n    - item2")
    assert isinstance(token, DictToken)
    assert token.value == {"outer": {"inner": ["item1", "item2"]}}


# LLM-generated content at query #42
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str, min_length=1)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, SimpleSchema)
    assert "parse_error" in str(exc_info.value)

    # Test valid YAML with validation errors
    content = "name: J"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert "no_content" in str(exc_info.value)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, schema={
            "name": Field(str),
            "age": Field(int)
        })

    content = """
    user:
        name: Alice
        age: 30
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, item_type=Field(str))

    content = """
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with type mismatch
    content = """
    items:
        - apple
        - 123
    """
    result, errors = validate_yaml(content, ListSchema)
    assert result is None
    assert len(errors) == 1


# LLM-generated content at query #43
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML (parse error)
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Bob"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "name" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = """
    user:
        name: Alice
        age: 30
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with various scalar types
    class ScalarSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)
        description = Field(str, allow_null=True)

    yaml_content = """
    count: 42
    price: 19.99
    active: true
    description: null
    """
    result, errors = validate_yaml(yaml_content, ScalarSchema)
    assert result == {
        "count": 42,
        "price": 19.99,
        "active": True,
        "description": None
    }
    assert errors == []

    # Test bytes input
    yaml_bytes = b"name: John"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #44
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error.msg for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with missing required field
    yaml_missing_field = "name: John"
    result, errors = validate_yaml(yaml_missing_field, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error.msg for error in errors)

    # Test YAML with extra fields (if schema doesn't allow them)
    class StrictSchema(Schema):
        name = Field(str)
        age = Field(int)
        strict = True

    yaml_extra_field = "name: John\nage: 30\nextra: field"
    result, errors = validate_yaml(yaml_extra_field, StrictSchema)
    assert result is None
    assert len(errors) > 0
    assert any("extra" in error.msg for error in errors)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_nested = "user: {name: John, age: 30}\nitems: [1, 2, 3]"
    result, errors = validate_yaml(yaml_nested, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}, "items": [1, 2, 3]}
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #45
#--------------------------

```python
def test_validate_yaml():
    # Test successful validation
    yaml_content = "key: value"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test validation error
    yaml_content = "key: 123"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "type_error"

    # Test parse error
    yaml_content = "key: [unclosed"
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test empty content
    yaml_content = ""
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test bytes content
    yaml_content = b"key: value"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []


# LLM-generated content at query #46
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML with a schema
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML
    yaml_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test invalid YAML syntax
    yaml_content = "name: John\nage: 30\ninvalid: yaml: content"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, TestSchema)
    assert exc_info.value.code == "parse_error"

    # Test YAML with a field
    field = Field(int)
    yaml_content = "42"
    result, errors = validate_yaml(yaml_content, field)
    assert result == 42
    assert errors == []

    # Test invalid YAML with a field
    field = Field(int)
    yaml_content = "forty-two"
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) == 1

    # Test YAML bytes content
    yaml_content = b"name: Jane\nage: 25"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "Jane", "age": 25}
    assert errors == []

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user: {name: Alice, age: 28}\nitems: [1, 2, 3]"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 28}, "items": [1, 2, 3]}
    assert errors == []


# LLM-generated content at query #47
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    content = "name: John\nage: :invalid"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test YAML that fails validation
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Jo"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: Alice
        age: 30
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with different data types
    class TypesSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)
        tags = Field(list, default=[])

    content = """
    count: 42
    price: 19.99
    active: true
    tags: [python, testing]
    """
    result, errors = validate_yaml(content, TypesSchema)
    assert result == {
        "count": 42,
        "price": 19.99,
        "active": True,
        "tags": ["python", "testing"]
    }
    assert errors == []


# LLM-generated content at query #48
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    assert exc_info.value.position.char_index == 0

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 12

    # Test nested structure
    token = tokenize_yaml("key:\n  - item1\n  - item2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": ["item1", "item2"]}
    assert token.start == 0
    assert token.end == 18

    # Test various scalar types
    token = tokenize_yaml("int: 1\nfloat: 1.5\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.5, "bool": True, "null": None}

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value: extra")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index >= 0


# LLM-generated content at query #49
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: 30\ninvalid: [unclosed"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "No content." not in e.text

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=18)

    content = "name: John\nage: 17"
    result, errors = validate_yaml(content, StrictSchema)
    assert result == {"name": "John", "age": 17}
    assert len(errors) == 2
    assert any("minimum" in error.code for error in errors)
    assert any("min_length" in error.code for error in errors)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = "user:\n  name: Alice\n  age: 30\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with bytes content
    content = b"name: Bob"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #50
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test integer scalar
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1

    # Test float scalar
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3

    # Test boolean scalar
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    # Test null scalar
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test list
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == 0
    assert token.end == 8

    # Test dict
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == 0
    assert token.end == 10

    # Test nested structures
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.start == 0
    assert token.end == 18

    # Test multiline content
    content = """key: value
list:
  - item1
  - item2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "list": ["item1", "item2"]}
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test bytes content
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None


# LLM-generated content at query #51
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result = validate_yaml(content, SimpleSchema)
    assert result[0] == {"name": "John"}
    assert result[1] == []

    # Test invalid YAML with a simple schema
    content = "name: 123"
    result = validate_yaml(content, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) > 0

    # Test YAML with a list schema
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - 1\n  - 2\n  - 3"
    result = validate_yaml(content, ListSchema)
    assert result[0] == {"items": [1, 2, 3]}
    assert result[1] == []

    # Test YAML with a nested schema
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: John\n  age: 30"
    result = validate_yaml(content, NestedSchema)
    assert result[0] == {"user": {"name": "John", "age": 30}}
    assert result[1] == []

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid YAML syntax
    content = "name: John\ninvalid: yaml: content"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test YAML with bytes content
    content = b"name: John"
    result = validate_yaml(content, SimpleSchema)
    assert result[0] == {"name": "John"}
    assert result[1] == []


# LLM-generated content at query #52
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: 30\ninvalid: [unclosed"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test YAML that fails schema validation
    class StrictSchema(Schema):
        name = Field(str, min_length=3)
        age = Field(int)

    content = "name: Jo\nage: thirty"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 2
    assert any("minimum length" in e.text for e in errors)
    assert any("valid integer" in e.text for e in errors)

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
user:
  name: Alice
  age: 25
items:
  - apple
  - banana
  - cherry
"""
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana", "cherry"]
    }
    assert errors == []

    # Test YAML with bytes content
    content = b"name: Bob"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "Bob"}
    assert errors == []

    # Test YAML with special characters
    content = "description: This is a test with 'quotes' and \"double quotes\""
    class SpecialCharsSchema(Schema):
        description = Field(str)

    result, errors = validate_yaml(content, SpecialCharsSchema)
    assert result == {"description": "This is a test with 'quotes' and \"double quotes\""}
    assert errors == []

    # Test YAML with boolean and null values
    class MixedSchema(Schema):
        active = Field(bool)
        value = Field(typing.Any, allow_null=True)

    content = "active: true\nvalue: null"
    result, errors = validate_yaml(content, MixedSchema)
    assert result == {"active": True, "value": None}
    assert errors == []


# LLM-generated content at query #53
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test valid YAML mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test valid YAML sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 12

    # Test YAML with different types
    token = tokenize_yaml("int: 1\nfloat: 1.5\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.5, "bool": True, "null": None}

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content")
    assert exc_info.value.code == "parse_error"

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test position tracking
    token = tokenize_yaml("line1: value1\nline2: value2")
    assert isinstance(token, DictToken)
    assert token.start == 0
    assert token.end == 21


# LLM-generated content at query #54
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML with a simple schema
    yaml_content = "name: 123"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test empty YAML content
    yaml_content = ""
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test invalid YAML syntax
    yaml_content = "name: [John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test nested YAML with a nested schema
    class NestedSchema(Schema):
        user = Field(dict)
        age = Field(int)

    yaml_content = """
    user:
        name: Jane
        email: jane@example.com
    age: 30
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "Jane", "email": "jane@example.com"},
        "age": 30
    }
    assert errors == []

    # Test YAML with a list
    class ListSchema(Schema):
        items = Field(list)

    yaml_content = """
    items:
        - apple
        - banana
        - cherry
    """
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana", "cherry"]}
    assert errors == []

    # Test YAML with bytes content
    yaml_content = b"name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #55
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML string
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty YAML string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test invalid YAML string
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index >= 0

    # Test YAML list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML scalar
    yaml_scalar = "scalar_value"
    token = tokenize_yaml(yaml_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"

    # Test YAML with different data types
    yaml_types = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_types)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }


# LLM-generated content at query #56
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: 30\ninvalid: [unclosed"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=18)

    content = "name: Jo\nage: 15"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 2
    assert any("minimum length" in error.text for error in errors)
    assert any("minimum value" in error.text for error in errors)

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, properties={
            "name": Field(str),
            "age": Field(int)
        })

    content = "user:\n  name: Alice\n  age: 25"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 25}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    content = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with bytes content
    content = b"name: Bob"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #57
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML (parse error)
    content = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        age = Field(int, minimum=18)

    content = "age: 15"
    result, errors = validate_yaml(content, StrictSchema)
    assert result == {"age": 15}
    assert len(errors) == 1
    assert errors[0].code == "minimum"

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, keys=Field(str), values=Field(int))

    content = "user:\n  age: 30\n  score: 100"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"age": 30, "score": 100}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    content = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with bytes content
    content = b"name: Alice"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "Alice"}
    assert errors == []


# LLM-generated content at query #58
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML with a simple schema
    yaml_content = "name: 123"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test empty YAML content
    yaml_content = ""
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test invalid YAML syntax
    yaml_content = "name: [John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test nested YAML with a complex schema
    class AddressSchema(Schema):
        street = Field(str)
        city = Field(str)

    class PersonSchema(Schema):
        name = Field(str)
        age = Field(int)
        address = Field(AddressSchema)

    yaml_content = """
    name: Jane
    age: 30
    address:
        street: 123 Main St
        city: New York
    """
    result, errors = validate_yaml(yaml_content, PersonSchema)
    assert result == {
        "name": "Jane",
        "age": 30,
        "address": {
            "street": "123 Main St",
            "city": "New York"
        }
    }
    assert errors == []

    # Test nested YAML with validation errors
    yaml_content = """
    name: Jane
    age: thirty
    address:
        street: 123 Main St
        city: New York
    """
    result, errors = validate_yaml(yaml_content, PersonSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test YAML with a list
    class ListSchema(Schema):
        items = Field(list)

    yaml_content = """
    items:
        - item1
        - item2
        - item3
    """
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["item1", "item2", "item3"]}
    assert errors == []

    # Test YAML with a list containing validation errors
    class ItemSchema(Schema):
        name = Field(str)
        value = Field(int)

    class ListOfItemsSchema(Schema):
        items = Field(list, items=ItemSchema)

    yaml_content = """
    items:
        - name: item1
          value: 100
        - name: item2
          value: two hundred
    """
    result, errors = validate_yaml(yaml_content, ListOfItemsSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"


# LLM-generated content at query #59
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result = validate_yaml(content, TestSchema)
    assert result == ({"name": "John"}, [])

    # Test invalid YAML with a simple schema
    content = "name: 123"
    result = validate_yaml(content, TestSchema)
    assert result == (None, [{"text": "Must be a string.", "code": "invalid_type", "position": {"line_no": 1, "column_no": 6, "char_index": 5}}])

    # Test YAML with a list schema
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    content = "items:\n  - 1\n  - 2\n  - 3"
    result = validate_yaml(content, ListSchema)
    assert result == ({"items": [1, 2, 3]}, [])

    # Test YAML with a list schema and invalid items
    content = "items:\n  - 1\n  - two\n  - 3"
    result = validate_yaml(content, ListSchema)
    assert result == (None, [{"text": "Must be an integer.", "code": "invalid_type", "position": {"line_no": 3, "column_no": 3, "char_index": 12}}])

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, TestSchema)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == {"line_no": 1, "column_no": 1, "char_index": 0}

    # Test invalid YAML syntax
    content = "name: John\ninvalid: yaml: content"
    try:
        validate_yaml(content, TestSchema)
    except ParseError as e:
        assert e.text.endswith(".")
        assert e.code == "parse_error"
        assert e.position is not None

    # Test YAML with nested schema
    class NestedSchema(Schema):
        user = Field(dict, schema=TestSchema)

    content = "user:\n  name: Jane"
    result = validate_yaml(content, NestedSchema)
    assert result == ({"user": {"name": "Jane"}}, [])

    # Test YAML with nested schema and invalid nested field
    content = "user:\n  name: 123"
    result = validate_yaml(content, NestedSchema)
    assert result == (None, [{"text": "Must be a string.", "code": "invalid_type", "position": {"line_no": 2, "column_no": 7, "char_index": 10}}])

    # Test YAML with bytes content
    content = b"name: John"
    result = validate_yaml(content, TestSchema)
    assert result == ({"name": "John"}, [])

    # Test YAML with bytes content and invalid type
    content = b"name: 123"
    result = validate_yaml(content, TestSchema)
    assert result == (None, [{"text": "Must be a string.", "code": "invalid_type", "position": {"line_no": 1, "column_no": 6, "char_index": 5}}])


# LLM-generated content at query #60
#--------------------------

```python
def test_validate_yaml():
    # Test successful validation
    yaml_content = "key: value"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test validation error
    yaml_content = "key: 123"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) > 0

    # Test parse error
    yaml_content = "key: [unclosed"
    field = Field(str)
    with pytest.raises(ParseError):
        validate_yaml(yaml_content, field)

    # Test empty content
    yaml_content = ""
    field = Field(str)
    with pytest.raises(ParseError):
        validate_yaml(yaml_content, field)

    # Test schema validation
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test schema validation error
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) > 0


# LLM-generated content at query #61
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML string with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    yaml_content = "name: John Doe"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John Doe"}
    assert errors == []

    # Test invalid YAML string
    invalid_yaml = "name: [John Doe"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_content = "name: 123"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "name" in errors[0].text

    # Test empty YAML string
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)

    yaml_content = "user:\n  name: Jane\n  age: 30"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Jane", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list)

    yaml_content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test bytes input
    yaml_bytes = b"name: Test User"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Test User"}
    assert errors == []


# LLM-generated content at query #62
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML mapping
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML sequence
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML scalar
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test YAML with different data types
    yaml_content = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test invalid YAML
    yaml_content = "invalid: yaml: content:"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"

    # Test bytes content
    yaml_content = b"key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}


# LLM-generated content at query #63
#--------------------------

```python
def test_validate_yaml():
    # Test successful validation
    yaml_content = "key: value"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test validation error
    yaml_content = "key: 123"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) > 0

    # Test parse error
    yaml_content = "key: [unclosed"
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) > 0

    # Test empty content
    yaml_content = ""
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) > 0

    # Test bytes content
    yaml_content = b"key: value"
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test schema validation
    class TestSchema(Schema):
        name = Field(str, min_length=3)

    yaml_content = "name: ab"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) > 0

    yaml_content = "name: abc"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "abc"}
    assert errors == []


# LLM-generated content at query #64
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple field
    valid_yaml = "key: value"
    field = Field(str)
    result, errors = validate_yaml(valid_yaml, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test invalid YAML (parse error)
    invalid_yaml = "key: value\n  - invalid yaml"
    field = Field(str)
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, field)

    # Test valid YAML with schema validation
    class TestSchema(Schema):
        name = Field(str, min_length=3)
        age = Field(int, minimum=0)

    valid_schema_yaml = "name: Alice\nage: 30"
    result, errors = validate_yaml(valid_schema_yaml, TestSchema)
    assert result == {"name": "Alice", "age": 30}
    assert errors == []

    # Test YAML with schema validation errors
    invalid_schema_yaml = "name: Al\nage: -5"
    result, errors = validate_yaml(invalid_schema_yaml, TestSchema)
    assert result is None
    assert len(errors) == 2
    assert any("name" in error.message for error in errors)
    assert any("age" in error.message for error in errors)

    # Test empty YAML content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, Field(str))
    assert exc_info.value.code == "no_content"

    # Test YAML with positional validation errors
    positional_yaml = "name: Bob\nage: twenty"
    result, errors = validate_yaml(positional_yaml, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].message


# LLM-generated content at query #65
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    with pytest.raises(ParseError):
        validate_yaml("name: [unclosed", SimpleSchema)

    # Test YAML that fails schema validation
    class NumberSchema(Schema):
        age = Field(int)

    content = "age: not_a_number"
    result, errors = validate_yaml(content, NumberSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, children={
            "name": Field(str),
            "age": Field(int)
        })

    content = """
    user:
        name: Alice
        age: 30
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, children=[Field(int)])

    content = "items: [1, 2, 3]"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with bytes content
    content_bytes = b"name: Bob"
    result, errors = validate_yaml(content_bytes, SimpleSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #66
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: invalid_int"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, SimpleSchema)
    assert len(errors) > 0
    assert "age" in str(errors[0])

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user: {name: Alice, age: 25}\nitems: [1, 2, 3]"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert result == {"user": {"name": "Alice", "age": 25}, "items": [1, 2, 3]}

    # Test YAML with bytes input
    bytes_yaml = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_yaml, SimpleSchema)
    assert errors == []
    assert result == {"name": "Bob", "age": 40}


# LLM-generated content at query #67
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    content = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test YAML with validation errors
    class NumberSchema(Schema):
        age = Field(int)

    content = "age: twenty"
    result, errors = validate_yaml(content, NumberSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test empty YAML
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: Alice\n  age: 30"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []


# LLM-generated content at query #68
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John Doe"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John Doe"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John Doe\ninvalid: yaml: content"
    with pytest.raises(ParseError):
        validate_yaml(content, TestSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=18)

    content = "name: Joe\nage: 17"
    result, errors = validate_yaml(content, StrictSchema)
    assert result == {"name": "Joe", "age": 17}
    assert len(errors) == 2
    assert any("minimum length" in str(e) for e in errors)
    assert any("minimum value" in str(e) for e in errors)

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: Alice
        age: 30
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test bytes content
    content = b"name: Bob"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #69
#--------------------------

```python
def test_validate_yaml():
    # Test successful validation
    yaml_content = "key: value"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test validation error
    yaml_content = "key: 123"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "type_error"

    # Test parse error
    yaml_content = "key: [unclosed"
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test empty content
    yaml_content = ""
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test bytes content
    yaml_content = b"key: value"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test schema validation
    class TestSchema(Schema):
        name = Field(str, max_length=10)

    yaml_content = "name: validname"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "validname"}
    assert errors == []

    yaml_content = "name: verylongname"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "max_length"


# LLM-generated content at query #70
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    schema = Schema({"name": str, "age": int})
    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    schema = Schema({"name": str})
    content = "name: [John"  # Missing closing bracket
    with pytest.raises(ParseError):
        validate_yaml(content, schema)

    # Test YAML with validation errors
    schema = Schema({"name": str, "age": int})
    content = "name: John\nage: thirty"  # Invalid age type
    result, errors = validate_yaml(content, schema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    schema = Schema({"name": str})
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, schema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    schema = Schema({"user": {"name": str, "roles": list}})
    content = "user:\n  name: Alice\n  roles:\n    - admin\n    - user"
    result, errors = validate_yaml(content, schema)
    assert result == {"user": {"name": "Alice", "roles": ["admin", "user"]}}
    assert errors == []

    # Test YAML with bytes content
    schema = Schema({"name": str})
    content = b"name: Bob"
    result, errors = validate_yaml(content, schema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #71
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test simple mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test simple sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 12

    # Test nested structure
    token = tokenize_yaml("key:\n  - item1\n  - item2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": ["item1", "item2"]}
    assert token.start == 0
    assert token.end == 18

    # Test various scalar types
    token = tokenize_yaml("int: 1\nfloat: 1.5\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.5, "bool": True, "null": None}

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value\n  invalid yaml")
    assert exc_info.value.code == "parse_error"
    assert "mapping values are not allowed" in exc_info.value.text


# LLM-generated content at query #72
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: 30\ninvalid: [unclosed"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=18)

    yaml_content = "name: Jo\nage: 15"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result == {"name": "Jo", "age": 15}
    assert len(errors) == 2
    assert any("minimum length" in error.text for error in errors)
    assert any("minimum value" in error.text for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with different data types
    class MixedSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)
        tags = Field(list, default=[])

    yaml_content = """
    count: 42
    price: 19.99
    active: true
    tags: [python, testing]
    """
    result, errors = validate_yaml(yaml_content, MixedSchema)
    assert result == {
        "count": 42,
        "price": 19.99,
        "active": True,
        "tags": ["python", "testing"]
    }
    assert errors == []

    # Test YAML with null values
    class NullableSchema(Schema):
        optional = Field(str, allow_null=True)

    yaml_content = "optional: null"
    result, errors = validate_yaml(yaml_content, NullableSchema)
    assert result == {"optional": None}
    assert errors == []

    # Test bytes input
    yaml_bytes = b"name: Test"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "Test"}
    assert errors == []


# LLM-generated content at query #73
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    yaml_content = "name: John\nage: 30"
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: "
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML that fails schema validation
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1

    # Test empty YAML content
    with pytest.raises(ParseError):
        validate_yaml("", TestSchema)

    # Test YAML with nested structures
    yaml_content = "user:\n  name: John\n  age: 30"
    class NestedSchema(Schema):
        user = Field(dict)
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}}
    assert errors == []

    # Test YAML with list
    yaml_content = "items:\n  - apple\n  - banana"
    class ListSchema(Schema):
        items = Field(list)
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []


# LLM-generated content at query #74
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML with scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test valid YAML with mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test valid YAML with sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 12

    # Test valid YAML with nested structures
    token = tokenize_yaml("key:\n  - item1\n  - item2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": ["item1", "item2"]}
    assert token.start == 0
    assert token.end == 18

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: [value")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.text.endswith(".")
    assert exc_info.value.position is not None

    # Test bytes content
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test YAML with different types
    token = tokenize_yaml("int: 1\nfloat: 1.5\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.5, "bool": True, "null": None}


# LLM-generated content at query #75
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty string
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.text == "No content."
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test integer scalar
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1

    # Test float scalar
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3

    # Test boolean scalar
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    # Test null scalar
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test list
    token = tokenize_yaml("- a\n- b\n- c")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b", "c"]
    assert token.start == 0
    assert token.end == 9

    # Test dict
    token = tokenize_yaml("a: 1\nb: 2\nc: 3")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2, "c": 3}
    assert token.start == 0
    assert token.end == 10

    # Test nested structure
    token = tokenize_yaml("a:\n  b:\n    - c\n    - d")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": ["c", "d"]}}
    assert token.start == 0
    assert token.end == 15

    # Test bytes input
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test invalid YAML
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("a: [\n  b: c\n]")
    assert excinfo.value.code == "parse_error"
    assert excinfo.value.text.endswith(".")
    assert excinfo.value.position is not None


# LLM-generated content at query #76
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML (parse error)
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        age = Field(int, minimum=18)

    yaml_content = "age: 15"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result == {"age": 15}
    assert len(errors) == 1
    assert errors[0].code == "minimum"

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, keys=Field(str), values=Field(int))

    yaml_content = "user:\n  age: 30\n  score: 100"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"age": 30, "score": 100}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(str))

    yaml_content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with bytes input
    yaml_bytes = b"name: Alice"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "Alice"}
    assert errors == []


# LLM-generated content at query #77
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    with pytest.raises(ParseError):
        validate_yaml("invalid: yaml: content", SimpleSchema)

    # Test YAML that fails validation
    class StrictSchema(Schema):
        age = Field(int, minimum=0)

    content = "age: -5"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "minimum"

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict, properties={
            "name": Field(str),
            "age": Field(int)
        })

    content = """
    user:
        name: Alice
        age: 30
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    content = "items: [1, 2, 3]"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with validation error in nested structure
    content = """
    user:
        name: Bob
        age: -10
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "minimum"


# LLM-generated content at query #78
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=18)

    yaml_with_errors = "name: Bob\nage: 15"
    result, errors = validate_yaml(yaml_with_errors, StrictSchema)
    assert result == {"name": "Bob", "age": 15}
    assert len(errors) == 2
    assert any("minimum length" in error.message for error in errors)
    assert any("minimum value" in error.message for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert "No content" in str(exc_info.value)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with bytes content
    bytes_content = b"name: John\nage: 30"
    result, errors = validate_yaml(bytes_content, SimpleSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #79
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with valid schema
    yaml_content = "name: John\nage: 30"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid_yaml_syntax:"
    schema = Schema(fields={"name": str, "age": int})
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, schema)

    # Test valid YAML with invalid schema
    yaml_content = "name: John\nage: thirty"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_content, schema)
    assert result is None
    assert len(errors) > 0

    # Test empty YAML content
    empty_yaml = ""
    schema = Schema(fields={"name": str})
    with pytest.raises(ParseError):
        validate_yaml(empty_yaml, schema)

    # Test YAML with extra fields not in schema
    yaml_content = "name: John\nage: 30\ncity: New York"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"name": "John", "age": 30}
    assert len(errors) > 0

    # Test YAML with missing required fields
    yaml_content = "name: John"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_content, schema)
    assert result is None
    assert len(errors) > 0

    # Test YAML with nested structures
    yaml_content = "user:\n  name: John\n  age: 30"
    schema = Schema(fields={"user": dict})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"user": {"name": "John", "age": 30}}
    assert errors == []

    # Test YAML with list
    yaml_content = "items:\n  - apple\n  - banana"
    schema = Schema(fields={"items": list})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with bytes content
    yaml_content = b"name: John\nage: 30"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test YAML with boolean values
    yaml_content = "active: true\nverified: false"
    schema = Schema(fields={"active": bool, "verified": bool})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"active": True, "verified": False}
    assert errors == []

    # Test YAML with null values
    yaml_content = "name: John\nmiddle_name: null"
    schema = Schema(fields={"name": str, "middle_name": typing.Optional[str]})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"name": "John", "middle_name": None}
    assert errors == []


# LLM-generated content at query #80
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML string
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty YAML string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML string
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"

    # Test YAML list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML scalar
    yaml_scalar = "scalar_value"
    token = tokenize_yaml(yaml_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"

    # Test YAML with different data types
    yaml_mixed = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_mixed)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None
    }


# LLM-generated content at query #81
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: 30\ninvalid: yaml: content"
    with pytest.raises(ParseError):
        validate_yaml(content, TestSchema)

    # Test YAML validation error
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].code

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with type validation
    class TypedSchema(Schema):
        count = Field(int)
        active = Field(bool)
        price = Field(float)

    content = """
    count: 42
    active: true
    price: 19.99
    """
    result, errors = validate_yaml(content, TypedSchema)
    assert result == {"count": 42, "active": True, "price": 19.99}
    assert errors == []

    # Test YAML with null values
    class NullableSchema(Schema):
        optional = Field(str, allow_null=True)

    content = "optional: null"
    result, errors = validate_yaml(content, NullableSchema)
    assert result == {"optional": None}
    assert errors == []

    # Test bytes input
    content_bytes = b"name: Jane"
    result, errors = validate_yaml(content_bytes, TestSchema)
    assert result == {"name": "Jane"}
    assert errors == []


# LLM-generated content at query #82
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a schema
    class TestSchema(Schema):
        name = Field(str, min_length=1)
        age = Field(int, minimum=0)

    yaml_content = """
name: John
age: 30
"""
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John\nage: invalid_yaml_syntax:"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test valid YAML with validation errors
    yaml_with_errors = """
name: J
age: -5
"""
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) == 2
    assert any("min_length" in error.code for error in errors)
    assert any("minimum" in error.code for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with missing required field
    yaml_missing_field = """
name: John
"""
    result, errors = validate_yaml(yaml_missing_field, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test YAML with extra fields (if schema doesn't allow extra)
    class StrictSchema(Schema):
        name = Field(str, min_length=1)
        extra = False

    yaml_extra_field = """
name: John
extra_field: value
"""
    result, errors = validate_yaml(yaml_extra_field, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "extra_field"


# LLM-generated content at query #83
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with valid schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: :invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test valid YAML with invalid schema
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with missing required field
    yaml_content = "age: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "name" in errors[0].text

    # Test YAML with extra fields
    yaml_content = "name: John\nage: 30\nextra: field"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}, "items": ["item1", "item2"]}
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #84
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_content = "name: John\nage: 30\ninvalid: [unclosed"
    with pytest.raises(ParseError):
        validate_yaml(invalid_content, SimpleSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    short_name_content = "name: Bob"
    result, errors = validate_yaml(short_name_content, StrictSchema)
    assert result == {"name": "Bob"}
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML content
    empty_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_content = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(nested_content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with type validation
    class TypedSchema(Schema):
        count = Field(int)
        active = Field(bool)
        price = Field(float)

    typed_content = """
    count: 42
    active: true
    price: 19.99
    """
    result, errors = validate_yaml(typed_content, TypedSchema)
    assert result == {"count": 42, "active": True, "price": 19.99}
    assert errors == []

    # Test YAML with bytes content
    bytes_content = b"name: Jane"
    result, errors = validate_yaml(bytes_content, SimpleSchema)
    assert result == {"name": "Jane"}
    assert errors == []


# LLM-generated content at query #85
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John Doe"
    result = validate_yaml(content, SimpleSchema)
    assert result[0]["name"] == "John Doe"
    assert result[1] == []

    # Test invalid YAML (syntax error)
    content = "name: John Doe\nage: invalid_yaml: 30"
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=0)

    content = "name: Jo\nage: -5"
    result = validate_yaml(content, StrictSchema)
    assert result[0] == {"name": "Jo", "age": -5}
    assert len(result[1]) == 2
    assert any("min_length" in str(err) for err in result[1])
    assert any("minimum" in str(err) for err in result[1])

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
user:
  name: Alice
  age: 25
items:
  - apple
  - banana
"""
    result = validate_yaml(content, NestedSchema)
    assert result[0]["user"]["name"] == "Alice"
    assert result[0]["items"] == ["apple", "banana"]
    assert result[1] == []

    # Test YAML with type mismatches
    class TypedSchema(Schema):
        count = Field(int)
        active = Field(bool)

    content = "count: not_a_number\nactive: yes"
    result = validate_yaml(content, TypedSchema)
    assert result[0] == {"count": "not_a_number", "active": True}
    assert len(result[1]) == 1
    assert any("int" in str(err).lower() for err in result[1])


# LLM-generated content at query #86
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: "
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test YAML validation error
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].code

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict, schema={
            "name": Field(str),
            "age": Field(int)
        })

    content = """
user:
  name: Alice
  age: 30
"""
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, item_type=Field(str))

    content = """
items:
  - apple
  - banana
  - cherry
"""
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": ["apple", "banana", "cherry"]}
    assert errors == []

    # Test YAML with type mismatch
    content = """
items:
  - 1
  - 2
  - 3
"""
    result, errors = validate_yaml(content, ListSchema)
    assert result is None
    assert len(errors) == 3
    for error in errors:
        assert "type" in error.code


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str, min_length=1)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML with a simple schema
    content = "name:"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test valid YAML with a complex schema
    class ComplexSchema(Schema):
        name = Field(str, min_length=1)
        age = Field(int, minimum=0)
        hobbies = Field(list, items=Field(str))

    content = """
    name: Jane
    age: 25
    hobbies:
      - reading
      - hiking
    """
    result, errors = validate_yaml(content, ComplexSchema)
    assert result == {"name": "Jane", "age": 25, "hobbies": ["reading", "hiking"]}
    assert errors == []

    # Test invalid YAML with a complex schema
    content = """
    name: Jane
    age: -5
    hobbies:
      - reading
      - 123
    """
    result, errors = validate_yaml(content, ComplexSchema)
    assert result is None
    assert len(errors) == 2
    assert any(error.code == "minimum" for error in errors)
    assert any(error.code == "type" for error in errors)

    # Test empty YAML content
    content = ""
    result, errors = validate_yaml(content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test invalid YAML syntax
    content = "name: John\nage: 25\ninvalid: yaml: content"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test YAML with bytes content
    content = b"name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test integer scalar
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1

    # Test float scalar
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3

    # Test boolean scalar
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    # Test null scalar
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test list
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == 0
    assert token.end == 8

    # Test dict
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == 0
    assert token.end == 10

    # Test nested structure
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.start == 0
    assert token.end == 17

    # Test multiline YAML
    content = """
    key:
      - item1
      - item2
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": ["item1", "item2"]}

    # Test bytes input
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 24


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert errors == []
    assert result == {"name": "John"}

    # Test invalid YAML with a simple schema
    yaml_content = "name: 123"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict, children={
            "name": Field(str),
            "age": Field(int)
        })

    yaml_content = """
user:
  name: Alice
  age: 30
"""
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert errors == []
    assert result == {"user": {"name": "Alice", "age": 30}}

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, children=Field(int))

    yaml_content = """
items:
  - 1
  - 2
  - 3
"""
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert errors == []
    assert result == {"items": [1, 2, 3]}

    # Test invalid YAML syntax
    yaml_content = "name: [John"
    try:
        validate_yaml(yaml_content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test empty YAML content
    yaml_content = ""
    try:
        validate_yaml(yaml_content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with bytes content
    yaml_content = b"name: Bob"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert errors == []
    assert result == {"name": "Bob"}


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    content = "name: John\nage: invalid_yaml: 30"
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Jo"
    result, errors = validate_yaml(content, StrictSchema)
    assert result == {"name": "Jo"}
    assert len(errors) == 1
    assert errors[0].text == "Must be at least 5 characters."

    # Test empty YAML
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: John\n  age: 30"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test bytes input
    content = b"name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: :invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML that fails schema validation
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    short_name_yaml = "name: Bob"
    result, errors = validate_yaml(short_name_yaml, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, schema={
            "name": Field(str),
            "age": Field(int)
        })

    nested_yaml = """
    user:
      name: Alice
      age: 30
    """
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, item_type=Field(int))

    list_yaml = """
    items:
      - 1
      - 2
      - 3
    """
    result, errors = validate_yaml(list_yaml, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with multiple validation errors
    class MultiFieldSchema(Schema):
        name = Field(str, min_length=3)
        age = Field(int, minimum=18)

    multi_error_yaml = """
    name: Bo
    age: 15
    """
    result, errors = validate_yaml(multi_error_yaml, MultiFieldSchema)
    assert result is None
    assert len(errors) == 2
    error_texts = [error.text for error in errors]
    assert any("min_length" in text for text in error_texts)
    assert any("minimum" in text for text in error_texts)


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML string
    yaml_str = "key: value"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(yaml_str) - 1

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test YAML list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML scalar
    yaml_scalar = "scalar"
    token = tokenize_yaml(yaml_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar"

    # Test YAML int
    yaml_int = "42"
    token = tokenize_yaml(yaml_int)
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test YAML float
    yaml_float = "3.14"
    token = tokenize_yaml(yaml_float)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test YAML bool
    yaml_bool = "true"
    token = tokenize_yaml(yaml_bool)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test YAML null
    yaml_null = "null"
    token = tokenize_yaml(yaml_null)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test invalid YAML
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.text.endswith(".")


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML mapping
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML sequence
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML scalar
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test YAML with different data types
    yaml_content = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test invalid YAML
    yaml_content = "invalid: yaml: content: :"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index >= 0

    # Test bytes content
    yaml_content_bytes = b"key: value"
    token = tokenize_yaml(yaml_content_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test nested structures
    yaml_content = """
    nested:
      mapping:
        key: value
      sequence:
        - item1
        - item2
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "nested": {
            "mapping": {"key": "value"},
            "sequence": ["item1", "item2"],
        }
    }


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid_int"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=18)

    yaml_with_errors = "name: John\nage: 17"
    result, errors = validate_yaml(yaml_with_errors, StrictSchema)
    assert len(errors) == 2
    assert any("minimum length" in error.text for error in errors)
    assert any("minimum value" in error.text for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        tags = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 25\ntags:\n  - python\n  - testing"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "tags": ["python", "testing"]
    }

    # Test YAML with bytes content
    bytes_content = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_content, SimpleSchema)
    assert errors == []
    assert result == {"name": "Bob", "age": 40}


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with valid schema
    class TestSchema(Schema):
        name = Field(str, min_length=1)
        age = Field(int, minimum=0)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    # Test valid YAML with invalid schema
    yaml_content = "name: J\nage: -5"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert len(errors) == 2
    assert result == {"name": "J", "age": -5}

    # Test invalid YAML
    yaml_content = "name: John\nage: :invalid"
    with pytest.raises(ParseError):
        validate_yaml(yaml_content, TestSchema)

    # Test empty YAML
    yaml_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user:\n  name: John\n  age: 30\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert errors == []
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["apple", "banana"]
    }

    # Test YAML with bytes input
    yaml_content = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with valid schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test valid YAML with invalid schema
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test invalid YAML
    yaml_content = "name: John\nage: 30\ninvalid: yaml: content"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test empty YAML
    yaml_content = ""
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test YAML with extra fields
    class StrictSchema(Schema):
        name = Field(str)
        age = Field(int)
        strict = True

    yaml_content = "name: John\nage: 30\nextra: field"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "extra" in errors[0].text

    # Test YAML with missing required field
    yaml_content = "age: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "name" in errors[0].text

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["item1", "item2"]
    }
    assert errors == []

    # Test YAML with bytes content
    yaml_content = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #11
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML mapping
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML list
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML scalar
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test YAML with different data types
    yaml_content = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test invalid YAML
    yaml_content = "invalid: yaml: content: :"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test bytes content
    yaml_content = b"key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test nested YAML structures
    yaml_content = """
    nested:
      mapping:
        key: value
      list:
        - item1
        - item2
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "nested": {
            "mapping": {"key": "value"},
            "list": ["item1", "item2"],
        }
    }


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result = validate_yaml(yaml_content, TestSchema)
    assert result == ({"name": "John"}, [])

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: :invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML that fails schema validation
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    short_name_yaml = "name: Bob"
    result = validate_yaml(short_name_yaml, StrictSchema)
    assert result[0] is None
    assert len(result[1]) > 0
    assert "min_length" in str(result[1][0])

    # Test empty YAML content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = """
    user:
        name: Alice
        age: 30
    items:
        - apple
        - banana
    """
    result = validate_yaml(nested_yaml, NestedSchema)
    assert result[0]["user"]["name"] == "Alice"
    assert result[0]["items"][0] == "apple"

    # Test YAML with different data types
    class TypesSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)
        tags = Field(list, default=[])

    types_yaml = """
    count: 42
    price: 19.99
    active: true
    tags:
        - python
        - testing
    """
    result = validate_yaml(types_yaml, TypesSchema)
    assert result[0]["count"] == 42
    assert result[0]["price"] == 19.99
    assert result[0]["active"] is True
    assert "python" in result[0]["tags"]

    # Test YAML with null values
    class NullSchema(Schema):
        value = Field(str, allow_null=True)

    null_yaml = "value: null"
    result = validate_yaml(null_yaml, NullSchema)
    assert result[0]["value"] is None


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML validation error
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Bob"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "Shorter than minimum length" in errors[0].text

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test complex YAML structure
    class ComplexSchema(Schema):
        users = Field(list)
        config = Field(dict)

    yaml_content = """
    users:
      - Alice
      - Bob
    config:
      debug: true
      timeout: 30
    """
    result, errors = validate_yaml(yaml_content, ComplexSchema)
    assert result == {
        "users": ["Alice", "Bob"],
        "config": {"debug": True, "timeout": 30}
    }
    assert errors == []

    # Test bytes input
    yaml_bytes = b"name: Jane"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "Jane"}
    assert errors == []


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    with pytest.raises(ParseError):
        validate_yaml("name: [John", TestSchema)

    # Test YAML that fails validation
    class StrictSchema(Schema):
        age = Field(int, minimum=18)

    content = "age: 17"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "must be greater than or equal to 18" in errors[0].text

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with bytes input
    content_bytes = b"name: Alice"
    result, errors = validate_yaml(content_bytes, TestSchema)
    assert result == {"name": "Alice"}
    assert errors == []

    # Test complex YAML structure
    class ComplexSchema(Schema):
        users = Field(list, items=Field(dict, keys=Field(str), values=Field(str)))

    content = """
    users:
      - name: John
        role: admin
      - name: Jane
        role: user
    """
    result, errors = validate_yaml(content, ComplexSchema)
    assert result == {
        "users": [
            {"name": "John", "role": "admin"},
            {"name": "Jane", "role": "user"}
        ]
    }
    assert errors == []


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid_int"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test empty YAML content
    with pytest.raises(ParseError):
        validate_yaml("", TestSchema)

    # Test YAML with missing required field
    incomplete_yaml = "name: John"
    result, errors = validate_yaml(incomplete_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test YAML with extra fields
    yaml_with_extra = "name: John\nage: 30\nextra: field"
    result, errors = validate_yaml(yaml_with_extra, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert len(errors) > 0

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict, schema=TestSchema)

    nested_yaml = "user:\n  name: Jane\n  age: 25"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Jane", "age": 25}}
    assert errors == []

    # Test YAML with list of items
    class ListSchema(Schema):
        items = Field(list, item_type=Field(str))

    list_yaml = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(list_yaml, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test integer scalar
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1

    # Test float scalar
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3

    # Test boolean scalar
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    # Test null scalar
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test list
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == 0
    assert token.end == 8

    # Test dict
    token = tokenize_yaml("{\"a\": 1}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}
    assert token.start == 0
    assert token.end == 6

    # Test bytes content
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.text.endswith(".")
    assert exc_info.value.position is not None


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_content = "name: John\nage: 30\ninvalid: [unclosed"
    with pytest.raises(ParseError):
        validate_yaml(invalid_content, SimpleSchema)

    # Test YAML validation error
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    short_name_content = "name: Bob"
    result, errors = validate_yaml(short_name_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].code

    # Test empty YAML content
    empty_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with bytes content
    bytes_content = b"name: Alice"
    result, errors = validate_yaml(bytes_content, SimpleSchema)
    assert result == {"name": "Alice"}
    assert errors == []

    # Test nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_content = """
    user:
        name: John
        age: 30
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(nested_content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with various scalar types
    class ScalarSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)
        description = Field(str, allow_null=True)

    scalar_content = """
    count: 42
    price: 19.99
    active: true
    description: null
    """
    result, errors = validate_yaml(scalar_content, ScalarSchema)
    assert result == {
        "count": 42,
        "price": 19.99,
        "active": True,
        "description": None
    }
    assert errors == []


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John\nage: invalid_int"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result == {"name": "John", "age": -5}
    assert len(errors) > 0

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with bytes input
    yaml_bytes = b"name: Jane\nage: 25"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Jane", "age": 25}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 28\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 28}, "items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with special characters
    special_yaml = "name: 'John \"The Boss\" Doe'\nage: 30"
    result, errors = validate_yaml(special_yaml, TestSchema)
    assert result == {"name": 'John "The Boss" Doe', "age": 30}
    assert errors == []


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John\nage: invalid_int"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error for error in errors)

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with missing required field
    incomplete_yaml = "name: John"
    result, errors = validate_yaml(incomplete_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error for error in errors)

    # Test YAML with extra fields (if schema doesn't allow them)
    class StrictSchema(Schema):
        name = Field(str)

    extra_fields_yaml = "name: John\nextra: field"
    result, errors = validate_yaml(extra_fields_yaml, StrictSchema)
    assert result is None
    assert len(errors) > 0
    assert any("extra" in error for error in errors)


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = """
name: John Doe
age: 30
"""
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert errors == []
    assert result["name"] == "John Doe"
    assert result["age"] == 30

    # Test invalid YAML (parse error)
    invalid_yaml = """
name: John Doe
age: 30
invalid: [unclosed bracket
"""
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test valid YAML with validation errors
    yaml_with_errors = """
name: John Doe
age: thirty
"""
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert len(errors) > 0
    assert any("age" in error.msg for error in errors)

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with extra fields (should pass if schema allows)
    yaml_extra_fields = """
name: John Doe
age: 30
extra: field
"""
    result, errors = validate_yaml(yaml_extra_fields, TestSchema)
    assert errors == []
    assert result["name"] == "John Doe"
    assert result["age"] == 30

    # Test YAML with missing required fields
    yaml_missing_fields = """
name: John Doe
"""
    result, errors = validate_yaml(yaml_missing_fields, TestSchema)
    assert len(errors) > 0
    assert any("age" in error.msg for error in errors)

    # Test YAML with bytes input
    yaml_bytes = b"name: John Doe\nage: 30"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert errors == []
    assert result["name"] == "John Doe"
    assert result["age"] == 30


# LLM-generated content at query #21
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML mapping
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid YAML sequence
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test valid YAML scalar
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"

    # Test YAML with different data types
    yaml_content = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test empty YAML content
    yaml_content = ""
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "no_content"

    # Test invalid YAML content
    yaml_content = "invalid: yaml: content"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"

    # Test YAML with bytes content
    yaml_content = b"key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test YAML with nested structures
    yaml_content = """
    outer:
      inner:
        - item1
        - item2
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "outer": {
            "inner": ["item1", "item2"]
        }
    }


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: "
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test YAML validation error
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: John"
    result, errors = validate_yaml(content, StrictSchema)
    assert result == {"name": "John"}
    assert len(errors) == 1
    assert "min_length" in errors[0].code

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: John\n  age: 30"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}}
    assert errors == []

    # Test YAML with bytes content
    content = b"name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test YAML with special characters
    content = "name: John O'Brien"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John O'Brien"}
    assert errors == []


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result == {"name": "John", "age": -5}
    assert len(errors) > 0
    assert any("age" in error["loc"] for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 25}, "items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with special characters
    special_yaml = "name: 'John \"The Boss\" Doe'\nage: 40"
    result, errors = validate_yaml(special_yaml, TestSchema)
    assert result == {"name": 'John "The Boss" Doe', "age": 40}
    assert errors == []

    # Test YAML with boolean and null values
    class ExtendedSchema(Schema):
        active = Field(bool)
        optional = Field(typing.Optional[str])

    bool_yaml = "active: true\noptional: null"
    result, errors = validate_yaml(bool_yaml, ExtendedSchema)
    assert result == {"active": True, "optional": None}
    assert errors == []

    # Test YAML with bytes input
    bytes_yaml = b"name: Jane\nage: 28"
    result, errors = validate_yaml(bytes_yaml, TestSchema)
    assert result == {"name": "Jane", "age": 28}
    assert errors == []


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: invalid_yaml: 30"
    with pytest.raises(ParseError):
        validate_yaml(content, TestSchema)

    # Test YAML with validation errors
    class TestSchemaWithValidation(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=18)

    content = "name: Jo\nage: 17"
    result, errors = validate_yaml(content, TestSchemaWithValidation)
    assert result == {"name": "Jo", "age": 17}
    assert len(errors) == 2
    assert any("minimum length is 5" in error.text for error in errors)
    assert any("must be greater than or equal to 18" in error.text for error in errors)

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with bytes content
    content = b"name: Bob"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, SimpleSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error.msg for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        tags = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 25\ntags:\n  - python\n  - testing"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 25}, "tags": ["python", "testing"]}
    assert errors == []

    # Test YAML with bytes content
    bytes_content = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_content, SimpleSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: "
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML that fails schema validation
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert errors == []
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }

    # Test YAML with bytes input
    yaml_bytes = b"name: Bob\nage: 40"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert errors == []
    assert result == {"name": "Bob", "age": 40}


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML with a simple schema
    content = "name: 123"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test YAML with a list schema
    class ListSchema(Schema):
        items = Field(list, default=[])

    content = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: 30\ninvalid: yaml: content"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: John\n  age: 30"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}}
    assert errors == []

    # Test YAML with bytes content
    content = b"name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("name: [John", SimpleSchema)
    assert exc_info.value.code == "parse_error"

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
user:
  name: Alice
  age: 30
items:
  - apple
  - banana
"""
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with type validation
    class TypedSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)

    content = """
count: 42
price: 19.99
active: true
"""
    result, errors = validate_yaml(content, TypedSchema)
    assert result == {"count": 42, "price": 19.99, "active": True}
    assert errors == []

    # Test YAML with null values
    class NullableSchema(Schema):
        value = Field(str, allow_null=True)

    content = "value: null"
    result, errors = validate_yaml(content, NullableSchema)
    assert result == {"value": None}
    assert errors == []


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    content = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(content, TestSchema)

    # Test YAML with validation errors
    class TestSchemaWithValidation(Schema):
        name = Field(str, min_length=5)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchemaWithValidation)
    assert result == {"name": "John"}
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
user:
  name: John
  age: 30
items:
  - apple
  - banana
"""
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with type mismatches
    class TypeSchema(Schema):
        count = Field(int)

    content = "count: not_a_number"
    result, errors = validate_yaml(content, TypeSchema)
    assert result == {"count": "not_a_number"}
    assert len(errors) == 1
    assert errors[0].code == "type"


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result = validate_yaml(content, SimpleSchema)
    assert result == ({"name": "John"}, [])

    # Test invalid YAML with a simple schema
    content = "name: 123"
    result = validate_yaml(content, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) > 0

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with list content
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - 1\n  - 2\n  - 3"
    result = validate_yaml(content, ListSchema)
    assert result == ({"items": [1, 2, 3]}, [])

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: Alice\n  age: 30"
    result = validate_yaml(content, NestedSchema)
    assert result == ({"user": {"name": "Alice", "age": 30}}, [])

    # Test YAML with bytes content
    content = b"name: Bob"
    result = validate_yaml(content, SimpleSchema)
    assert result == ({"name": "Bob"}, [])

    # Test YAML with invalid syntax
    content = "name: John\nage: 30\ninvalid: yaml: content"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test with invalid YAML
    invalid_yaml = "name: John\nage: thirty"
    result, errors = validate_yaml(invalid_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert errors[0].code == "invalid_type"

    # Test with empty YAML
    empty_yaml = ""
    result, errors = validate_yaml(empty_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert errors[0].code == "no_content"

    # Test with malformed YAML
    malformed_yaml = "name: John\nage: 30\ninvalid: :"
    result, errors = validate_yaml(malformed_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert errors[0].code == "parse_error"

    # Test with bytes content
    bytes_content = b"name: Jane\nage: 25"
    result, errors = validate_yaml(bytes_content, TestSchema)
    assert result == {"name": "Jane", "age": 25}
    assert errors == []

    # Test with nested schema
    class NestedSchema(Schema):
        user = Field(TestSchema)

    nested_yaml = "user:\n  name: Alice\n  age: 28"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 28}}
    assert errors == []

    # Test with list in YAML
    class ListSchema(Schema):
        items = Field(list)

    list_yaml = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(list_yaml, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test with missing required field
    missing_field_yaml = "name: Bob"
    result, errors = validate_yaml(missing_field_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert errors[0].code == "required_field"


# LLM-generated content at query #32
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1

    # Test float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3

    # Test boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test list
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == 0
    assert token.end == 8

    # Test dict
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == 0
    assert token.end == 10

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index >= 0

    # Test bytes input
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test multiline content
    content = "key:\n  - item1\n  - item2"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": ["item1", "item2"]}


# LLM-generated content at query #33
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result = validate_yaml(content, TestSchema)
    assert result[0] == {"name": "John"}
    assert result[1] == []

    # Test invalid YAML syntax
    content = "name: John\nage: 30\ninvalid: yaml: content"
    try:
        validate_yaml(content, TestSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result = validate_yaml(content, StrictSchema)
    assert result[0] is None
    assert len(result[1]) > 0
    assert "min_length" in str(result[1][0])

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, TestSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result = validate_yaml(content, NestedSchema)
    assert result[0] == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert result[1] == []

    # Test YAML with type conversion
    class TypeSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)

    content = """
    count: 42
    price: 19.99
    active: true
    """
    result = validate_yaml(content, TypeSchema)
    assert result[0] == {"count": 42, "price": 19.99, "active": True}
    assert result[1] == []

    # Test YAML with null values
    class NullSchema(Schema):
        value = Field(typing.Optional[str])

    content = "value: null"
    result = validate_yaml(content, NullSchema)
    assert result[0] == {"value": None}
    assert result[1] == []

    # Test bytes input
    content_bytes = b"name: John"
    result = validate_yaml(content_bytes, TestSchema)
    assert result[0] == {"name": "John"}
    assert result[1] == []


# LLM-generated content at query #34
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: 123\nage: thirty"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) == 2
    assert any("name" in error.msg for error in errors)
    assert any("age" in error.msg for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with missing required field
    yaml_missing_field = "name: Jane"
    result, errors = validate_yaml(yaml_missing_field, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].msg

    # Test YAML with extra fields (if schema doesn't allow them)
    class StrictSchema(Schema):
        name = Field(str)
        age = Field(int)
        strict = True

    yaml_extra_field = "name: Bob\nage: 25\nextra: field"
    result, errors = validate_yaml(yaml_extra_field, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "extra" in errors[0].msg

    # Test YAML with correct types but wrong values (e.g., age negative)
    class ValidatedSchema(Schema):
        name = Field(str, min_length=1)
        age = Field(int, minimum=0)

    yaml_wrong_values = "name: \nage: -5"
    result, errors = validate_yaml(yaml_wrong_values, ValidatedSchema)
    assert result is None
    assert len(errors) == 2


# LLM-generated content at query #35
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with valid schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test valid YAML with invalid schema
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test invalid YAML
    yaml_content = "name: John\nage: 30\ninvalid: yaml: content"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test empty YAML
    yaml_content = ""
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test YAML with extra fields
    class StrictSchema(Schema):
        name = Field(str)
        age = Field(int)

        class Options:
            extra = "forbid"

    yaml_content = "name: John\nage: 30\nextra: field"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "extra_forbidden"

    # Test YAML with missing required field
    yaml_content = "age: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["item1", "item2"]
    }
    assert errors == []

    # Test YAML with bytes content
    yaml_content = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #36
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    assert exc_info.value.position.char_index == 0

    # Test valid YAML mapping
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML sequence
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML scalar
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test YAML with different data types
    yaml_content = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test invalid YAML
    invalid_yaml = "key: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index >= 0

    # Test bytes content
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test nested structures
    yaml_content = """
    outer:
      inner:
        - item1
        - item2
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "outer": {
            "inner": ["item1", "item2"]
        }
    }


# LLM-generated content at query #37
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with correct schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid_yaml_syntax:"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error["loc"] for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with missing required field
    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error["loc"] for error in errors)

    # Test YAML with extra fields (if schema doesn't allow them)
    class StrictSchema(Schema):
        name = Field(str)
        age = Field(int)
        allow_extra = False

    yaml_content = "name: John\nage: 30\nextra_field: value"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) > 0
    assert any("extra_field" in error["loc"] for error in errors)

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["item1", "item2"]
    }
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #38
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test integer
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1

    # Test float
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3

    # Test boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test list
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == 0
    assert token.end == 8

    # Test dict
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test bytes input
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert "unclosed" in exc_info.value.text.lower()

    # Test multiline content
    content = """key: value
list:
  - item1
  - item2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "list": ["item1", "item2"]}
    assert token.start == 0
    assert token.end == len(content) - 1


# LLM-generated content at query #39
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: 30\ninvalid: [unclosed"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Bob"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].code

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with bytes input
    yaml_bytes = b"name: Alice"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Alice"}
    assert errors == []

    # Test nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = """
    user:
        name: John
        age: 30
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with various scalar types
    class ScalarSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)
        description = Field(str, allow_null=True)

    yaml_content = """
    count: 42
    price: 3.14
    active: true
    description: null
    """
    result, errors = validate_yaml(yaml_content, ScalarSchema)
    assert result == {
        "count": 42,
        "price": 3.14,
        "active": True,
        "description": None
    }
    assert errors == []


# LLM-generated content at query #40
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML with a simple schema
    yaml_content = "name: 123"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test empty YAML content
    yaml_content = ""
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test invalid YAML syntax
    yaml_content = "name: [John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test nested YAML with a complex schema
    class AddressSchema(Schema):
        street = Field(str)
        city = Field(str)

    class PersonSchema(Schema):
        name = Field(str)
        age = Field(int)
        address = Field(AddressSchema)

    yaml_content = """
    name: Jane
    age: 30
    address:
      street: 123 Main St
      city: New York
    """
    result, errors = validate_yaml(yaml_content, PersonSchema)
    assert result == {
        "name": "Jane",
        "age": 30,
        "address": {"street": "123 Main St", "city": "New York"}
    }
    assert errors == []

    # Test nested YAML with validation errors
    yaml_content = """
    name: Jane
    age: thirty
    address:
      street: 123 Main St
      city: New York
    """
    result, errors = validate_yaml(yaml_content, PersonSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test YAML with a list
    class ListSchema(Schema):
        items = Field(list)

    yaml_content = """
    items:
      - item1
      - item2
      - item3
    """
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["item1", "item2", "item3"]}
    assert errors == []

    # Test YAML with a list containing validation errors
    class ItemSchema(Schema):
        name = Field(str)
        value = Field(int)

    class ListWithSchema(Schema):
        items = Field(list, item_type=ItemSchema)

    yaml_content = """
    items:
      - name: item1
        value: 10
      - name: item2
        value: twenty
    """
    result, errors = validate_yaml(yaml_content, ListWithSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"


# LLM-generated content at query #41
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_with_errors, SimpleSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error["loc"] for error in errors)

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with bytes input
    bytes_yaml = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_yaml, SimpleSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []


# LLM-generated content at query #42
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML mapping
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML sequence
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML scalar
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML with different types
    yaml_content = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test invalid YAML
    yaml_content = "key: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index == 10

    # Test bytes input
    yaml_content = b"key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test nested structures
    yaml_content = """
    nested:
      list:
        - item1
        - item2
      dict:
        key1: value1
        key2: value2
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "nested": {
            "list": ["item1", "item2"],
            "dict": {"key1": "value1", "key2": "value2"},
        }
    }


# LLM-generated content at query #43
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML with scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test valid YAML with mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test valid YAML with sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 12

    # Test valid YAML with nested structures
    token = tokenize_yaml("list:\n  - item1\n  - item2\nnested:\n  key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"list": ["item1", "item2"], "nested": {"key": "value"}}
    assert token.start == 0
    assert token.end == 38

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value: extra")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 10

    # Test YAML with different data types
    token = tokenize_yaml("int: 42\nfloat: 3.14\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 42, "float": 3.14, "bool": True, "null": None}
    assert token.start == 0
    assert token.end == 35

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9


# LLM-generated content at query #44
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML string
    yaml_str = "key: value"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test empty bytes
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(b"")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test invalid YAML
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test YAML list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML scalar
    yaml_scalar = "scalar"
    token = tokenize_yaml(yaml_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar"

    # Test YAML with different data types
    yaml_types = """
    int: 1
    float: 1.1
    bool: true
    null: null
    """
    token = tokenize_yaml(yaml_types)
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.1, "bool": True, "null": None}


# LLM-generated content at query #45
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML with a simple schema
    yaml_content = "name: 123"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict, children={
            "name": Field(str),
            "age": Field(int)
        })

    yaml_content = """
user:
  name: Alice
  age: 30
"""
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, children=Field(int))

    yaml_content = """
items:
  - 1
  - 2
  - 3
"""
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test invalid YAML syntax
    yaml_content = "name: [John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test empty YAML content
    yaml_content = ""
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test YAML with bytes content
    yaml_content = b"name: Bob"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #46
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str, required=True)
        age = Field(int, required=True)

    yaml_content = """
    name: John Doe
    age: 30
    """
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John Doe", "age": 30}
    assert errors == []

    # Test invalid YAML (parse error)
    invalid_yaml = """
    name: John Doe
    age: thirty
    """
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = """
    name: John Doe
    age: 3000
    """
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test empty YAML content
    empty_yaml = ""
    with pytest.raises(ParseError):
        validate_yaml(empty_yaml, TestSchema)

    # Test YAML with missing required field
    yaml_missing_field = """
    name: John Doe
    """
    result, errors = validate_yaml(yaml_missing_field, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test YAML with bytes content
    yaml_bytes = b"""
    name: John Doe
    age: 30
    """
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "John Doe", "age": 30}
    assert errors == []


# LLM-generated content at query #47
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John Doe"
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] == {"name": "John Doe"}
    assert result[1] == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John Doe\ninvalid: [unclosed"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int)

    yaml_content = "name: Jo\nage: 25"
    result = validate_yaml(yaml_content, StrictSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert "name" in result[1][0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with list content
    class ListSchema(Schema):
        items = Field(list)

    yaml_content = "items:\n  - item1\n  - item2"
    result = validate_yaml(yaml_content, ListSchema)
    assert result[0] == {"items": ["item1", "item2"]}
    assert result[1] == []

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        settings = Field(dict)

    yaml_content = """
    user:
      name: Alice
      age: 30
    settings:
      theme: dark
      notifications: true
    """
    result = validate_yaml(yaml_content, NestedSchema)
    assert result[0]["user"]["name"] == "Alice"
    assert result[0]["settings"]["notifications"] is True
    assert result[1] == []

    # Test YAML with type mismatches
    class TypedSchema(Schema):
        count = Field(int)
        active = Field(bool)

    yaml_content = "count: not_a_number\nactive: yes"
    result = validate_yaml(yaml_content, TypedSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert "count" in result[1][0].text


# LLM-generated content at query #48
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid_int"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML validation error
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list)

    yaml_content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)

    yaml_content = "user:\n  name: Alice\n  email: alice@example.com"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Alice", "email": "alice@example.com"}}
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: Bob\nage: 25"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Bob", "age": 25}
    assert errors == []


# LLM-generated content at query #49
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    content = "name: John\nage: invalid_type"
    result, errors = validate_yaml(content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "extra_field"

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Jo"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML
    content = ""
    result, errors = validate_yaml(content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test malformed YAML
    content = "name: John\nage: 30\ninvalid: yaml: content"
    result, errors = validate_yaml(content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, schema=TestSchema)

    content = "user:\n  name: Jane"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "Jane"}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, item_type=str)

    content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with bytes content
    content = b"name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test YAML with positional validation
    class PositionalSchema(Schema):
        name = Field(str, max_length=10)

    content = "name: ThisNameIsTooLong"
    result, errors = validate_yaml(content, PositionalSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    assert errors[0].position is not None


# LLM-generated content at query #50
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML string
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty YAML string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test invalid YAML syntax
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index >= 0

    # Test YAML with list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML with nested structure
    nested_yaml = "outer:\n  inner: value"
    token = tokenize_yaml(nested_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"outer": {"inner": "value"}}


# LLM-generated content at query #51
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    value, errors = validate_yaml(yaml_content, SimpleSchema)
    assert value == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: invalid_int"
    try:
        validate_yaml(invalid_yaml, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    value, errors = validate_yaml(yaml_with_errors, SimpleSchema)
    assert value == {"name": "John", "age": -5}
    assert len(errors) > 0
    assert any("age" in error for error in errors)

    # Test empty YAML
    try:
        validate_yaml("", SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user:\n  name: Alice\nitems:\n  - 1\n  - 2"
    value, errors = validate_yaml(nested_yaml, NestedSchema)
    assert value == {"user": {"name": "Alice"}, "items": [1, 2]}
    assert errors == []

    # Test YAML with bytes input
    bytes_yaml = b"name: Bob\nage: 25"
    value, errors = validate_yaml(bytes_yaml, SimpleSchema)
    assert value == {"name": "Bob", "age": 25}
    assert errors == []


# LLM-generated content at query #52
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: thirty"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error for error in errors)

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user: {name: Alice, age: 25}\nitems: [1, 2, 3]"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 25}, "items": [1, 2, 3]}
    assert errors == []

    # Test YAML with bytes input
    bytes_yaml = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_yaml, TestSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []


# LLM-generated content at query #53
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML validation error
    yaml_content = "age: 25"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with bytes
    yaml_bytes = b"name: Jane"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Jane"}
    assert errors == []

    # Test nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict)

    yaml_content = "user:\n  name: Alice\n  age: 30"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []


# LLM-generated content at query #54
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML string
    yaml_str = "key: value"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty YAML string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML string
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"

    # Test YAML with list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML with nested structures
    nested_yaml = "outer:\n  inner: value"
    token = tokenize_yaml(nested_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"outer": {"inner": "value"}}


# LLM-generated content at query #55
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test valid YAML mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test valid YAML sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 12

    # Test YAML with different types
    token = tokenize_yaml("int: 1\nfloat: 1.5\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.5, "bool": True, "null": None}

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content")
    assert exc_info.value.code == "parse_error"

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test YAML with nested structures
    token = tokenize_yaml("nested:\n  list:\n    - item1\n    - item2")
    assert isinstance(token, DictToken)
    assert token.value == {"nested": {"list": ["item1", "item2"]}}


# LLM-generated content at query #56
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test valid YAML mapping
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test valid YAML sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 12

    # Test valid YAML with different types
    token = tokenize_yaml("int: 1\nfloat: 1.5\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.5, "bool": True, "null": None}

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content")
    assert exc_info.value.code == "parse_error"

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test position tracking
    token = tokenize_yaml("a: b\nc: d")
    assert isinstance(token, DictToken)
    assert token.start == 0
    assert token.end == 7


# LLM-generated content at query #57
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: invalid_yaml: 30"
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test YAML validation error
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].code

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with different data types
    class TypesSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)
        tags = Field(list, default=[])

    content = """
    count: 42
    price: 19.99
    active: true
    tags: [python, testing]
    """
    result, errors = validate_yaml(content, TypesSchema)
    assert result == {
        "count": 42,
        "price": 19.99,
        "active": True,
        "tags": ["python", "testing"]
    }
    assert errors == []

    # Test YAML with bytes content
    content_bytes = b"name: Jane"
    result, errors = validate_yaml(content_bytes, SimpleSchema)
    assert result == {"name": "Jane"}
    assert errors == []


# LLM-generated content at query #58
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result = validate_yaml(content, TestSchema)
    assert result == ({"name": "John"}, [])

    # Test invalid YAML (syntax error)
    invalid_content = "name: John\nage: 30\ninvalid: [unclosed"
    with pytest.raises(ParseError):
        validate_yaml(invalid_content, TestSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    short_name_content = "name: Bob"
    result = validate_yaml(short_name_content, StrictSchema)
    assert result[0] == {"name": "Bob"}
    assert len(result[1]) == 1
    assert "min_length" in result[1][0].code

    # Test empty YAML content
    empty_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_content = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result = validate_yaml(nested_content, NestedSchema)
    expected = {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert result == (expected, [])

    # Test YAML with type validation
    class TypedSchema(Schema):
        count = Field(int)
        active = Field(bool)
        price = Field(float)

    typed_content = """
    count: 42
    active: true
    price: 19.99
    """
    result = validate_yaml(typed_content, TypedSchema)
    expected = {
        "count": 42,
        "active": True,
        "price": 19.99
    }
    assert result == (expected, [])

    # Test YAML with null values
    class NullableSchema(Schema):
        value = Field(typing.Optional[str])

    null_content = "value: null"
    result = validate_yaml(null_content, NullableSchema)
    assert result == ({"value": None}, [])


# LLM-generated content at query #59
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: 30\ninvalid: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert "parse_error" in str(exc_info.value)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result, errors = validate_yaml(content, StrictSchema)
    assert result == {"name": "Bob"}
    assert len(errors) == 1
    assert "min_length" in errors[0].text

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert "no_content" in str(exc_info.value)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with type validation
    class TypedSchema(Schema):
        count = Field(int)
        active = Field(bool)
        price = Field(float)

    content = """
    count: 42
    active: true
    price: 19.99
    """
    result, errors = validate_yaml(content, TypedSchema)
    assert result == {"count": 42, "active": True, "price": 19.99}
    assert errors == []

    # Test YAML with null values
    class NullableSchema(Schema):
        optional = Field(str, allow_null=True)

    content = "optional: null"
    result, errors = validate_yaml(content, NullableSchema)
    assert result == {"optional": None}
    assert errors == []

    # Test YAML with bytes content
    content_bytes = b"name: John"
    result, errors = validate_yaml(content_bytes, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #60
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    content = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(content, TestSchema)

    # Test YAML with validation errors
    class TestSchemaWithValidation(Schema):
        name = Field(str, min_length=5)

    content = "name: Jo"
    result, errors = validate_yaml(content, TestSchemaWithValidation)
    assert result == {"name": "Jo"}
    assert len(errors) == 1
    assert "min_length" in errors[0].code

    # Test empty YAML
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: John
        age: 30
    items:
        - item1
        - item2
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["item1", "item2"]
    }
    assert errors == []

    # Test YAML with bytes content
    content = b"name: Jane"
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "Jane"}
    assert errors == []


# LLM-generated content at query #61
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with missing required field
    incomplete_yaml = "name: John"
    result, errors = validate_yaml(incomplete_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test YAML with extra fields (if schema doesn't allow them)
    class StrictSchema(Schema):
        name = Field(str)

    extra_fields_yaml = "name: John\nage: 30"
    result, errors = validate_yaml(extra_fields_yaml, StrictSchema)
    assert result is None
    assert len(errors) > 0

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user:\n  name: John\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "John"}, "items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with bytes content
    bytes_content = b"name: John\nage: 30"
    result, errors = validate_yaml(bytes_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #62
#--------------------------

```python
def test_validate_yaml():
    # Test successful validation
    yaml_content = "key: value"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test validation error
    yaml_content = "key: 123"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "type_error"

    # Test parse error
    yaml_content = "key: [unclosed"
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test empty content
    yaml_content = ""
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test bytes content
    yaml_content = b"key: value"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test schema validation
    class TestSchema(Schema):
        name = Field(str, min_length=1)
    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test schema validation error
    yaml_content = "name: "
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "min_length"


# LLM-generated content at query #63
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result = validate_yaml(yaml_content, TestSchema)
    assert result[0] == {"name": "John", "age": 30}
    assert result[1] == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: thirty"
    result = validate_yaml(yaml_with_errors, TestSchema)
    assert result[0] is None
    assert len(result[1]) > 0

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user: {name: Alice}\nitems: [1, 2, 3]"
    result = validate_yaml(nested_yaml, NestedSchema)
    assert result[0] == {"user": {"name": "Alice"}, "items": [1, 2, 3]}
    assert result[1] == []

    # Test YAML with bytes input
    bytes_yaml = b"name: Bob\nage: 25"
    result = validate_yaml(bytes_yaml, TestSchema)
    assert result[0] == {"name": "Bob", "age": 25}
    assert result[1] == []

    # Test YAML with special characters
    special_yaml = "name: 'John Doe'\nage: 30"
    result = validate_yaml(special_yaml, TestSchema)
    assert result[0] == {"name": "John Doe", "age": 30}
    assert result[1] == []


# LLM-generated content at query #64
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] == {"name": "John"}
    assert result[1] == []

    # Test invalid YAML with a simple schema
    yaml_content = "name: 123"
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) > 0

    # Test YAML with a list schema
    class ListSchema(Schema):
        items = Field(list, items=Field(str))

    yaml_content = "items:\n  - apple\n  - banana"
    result = validate_yaml(yaml_content, ListSchema)
    assert result[0] == {"items": ["apple", "banana"]}
    assert result[1] == []

    # Test invalid YAML with a list schema
    yaml_content = "items:\n  - apple\n  - 123"
    result = validate_yaml(yaml_content, ListSchema)
    assert result[0] is None
    assert len(result[1]) > 0

    # Test empty YAML content
    yaml_content = ""
    try:
        validate_yaml(yaml_content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test malformed YAML content
    yaml_content = "name: John\ninvalid: yaml: content"
    try:
        validate_yaml(yaml_content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test YAML with nested schema
    class NestedSchema(Schema):
        user = Field(dict, schema=Field(dict, keys=Field(str), values=Field(int)))

    yaml_content = "user:\n  age: 30\n  height: 180"
    result = validate_yaml(yaml_content, NestedSchema)
    assert result[0] == {"user": {"age": 30, "height": 180}}
    assert result[1] == []

    # Test invalid nested YAML
    yaml_content = "user:\n  age: thirty\n  height: 180"
    result = validate_yaml(yaml_content, NestedSchema)
    assert result[0] is None
    assert len(result[1]) > 0

    # Test YAML with bytes content
    yaml_content = b"name: John"
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] == {"name": "John"}
    assert result[1] == []


# LLM-generated content at query #65
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML
    with pytest.raises(ParseError):
        validate_yaml("invalid: yaml: content", SimpleSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        age = Field(int, minimum=0)

    yaml_content = "age: -5"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "minimum" in errors[0].text

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, properties={
            "name": Field(str),
            "age": Field(int)
        })

    yaml_content = """
    user:
      name: Alice
      age: 30
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    yaml_content = "items: [1, 2, 3]"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test bytes input
    yaml_bytes = b"name: Bob"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #66
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML with a simple schema
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test empty YAML content
    yaml_content = ""
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test malformed YAML content
    yaml_content = "name: John\nage: 30\ninvalid: yaml: content"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test YAML with missing required field
    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required_field"

    # Test YAML with extra fields
    class StrictSchema(Schema):
        name = Field(str)
        age = Field(int)
        strict = True

    yaml_content = "name: John\nage: 30\nextra: field"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "extra_field"

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        settings = Field(list)

    yaml_content = "user:\n  name: John\n  age: 30\nsettings:\n  - dark_mode: true\n  - notifications: false"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "settings": [{"dark_mode": True}, {"notifications": False}]
    }
    assert errors == []

    # Test YAML with bytes content
    yaml_content = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #67
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: invalid_yaml: 30"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test YAML validation error
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
    user:
        name: Alice
        age: 25
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with type validation
    class TypedSchema(Schema):
        count = Field(int)
        active = Field(bool)
        price = Field(float)

    content = """
    count: 42
    active: true
    price: 19.99
    """
    result, errors = validate_yaml(content, TypedSchema)
    assert result == {"count": 42, "active": True, "price": 19.99}
    assert errors == []

    # Test YAML with null values
    class NullableSchema(Schema):
        optional = Field(str, allow_null=True)

    content = "optional: null"
    result, errors = validate_yaml(content, NullableSchema)
    assert result == {"optional": None}
    assert errors == []

    # Test YAML bytes input
    content_bytes = b"name: John"
    result, errors = validate_yaml(content_bytes, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #68
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str, required=True)
        age = Field(int, required=True)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML that fails schema validation
    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with bytes content
    yaml_bytes = b"name: Jane\nage: 25"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Jane", "age": 25}
    assert errors == []

    # Test nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict, required=True)
        settings = Field(list, required=True)

    yaml_content = "user:\n  name: Alice\n  role: admin\nsettings:\n  - dark_mode: true\n  - notifications: false"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "role": "admin"},
        "settings": [{"dark_mode": True}, {"notifications": False}]
    }
    assert errors == []


# LLM-generated content at query #69
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: 30\ninvalid: [unclosed"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "No closing" in e.text

    # Test YAML validation error
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Jo"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "min_length"
    assert errors[0].position is not None

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - a\n  - b\n  - c"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": ["a", "b", "c"]}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        settings = Field(dict)

    content = "user:\n  name: Alice\n  age: 25\nsettings:\n  theme: dark"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "settings": {"theme": "dark"}
    }
    assert errors == []

    # Test YAML with bytes content
    content = b"name: Bob"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "Bob"}
    assert errors == []

    # Test YAML with type validation
    class TypedSchema(Schema):
        count = Field(int)

    content = "count: 42"
    result, errors = validate_yaml(content, TypedSchema)
    assert result == {"count": 42}
    assert errors == []

    # Test YAML with invalid type
    content = "count: not_a_number"
    result, errors = validate_yaml(content, TypedSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"


# LLM-generated content at query #70
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML validation error
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list)

    yaml_content = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)

    yaml_content = "user:\n  name: Alice\n  email: alice@example.com"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Alice", "email": "alice@example.com"}}
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: Bob\nage: 25"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "Bob", "age": 25}
    assert errors == []

    # Test YAML with special characters
    yaml_content = "name: José\nage: 40"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "José", "age": 40}
    assert errors == []


# LLM-generated content at query #71
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result = validate_yaml(content, SimpleSchema)
    assert result == ({"name": "John"}, [])

    # Test invalid YAML with a simple schema
    content = "name: 123"
    result = validate_yaml(content, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) > 0

    # Test YAML with a list schema
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - 1\n  - 2"
    result = validate_yaml(content, ListSchema)
    assert result == ({"items": [1, 2]}, [])

    # Test invalid YAML syntax
    content = "name: John\ninvalid: yaml: content"
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: John\n  age: 30"
    result = validate_yaml(content, NestedSchema)
    assert result == ({"user": {"name": "John", "age": 30}}, [])

    # Test YAML with positional validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result = validate_yaml(content, StrictSchema)
    assert result[0] is None
    assert len(result[1]) > 0


# LLM-generated content at query #72
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with correct schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: thirty"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=18)

    yaml_with_errors = "name: Tom\nage: 15"
    result, errors = validate_yaml(yaml_with_errors, StrictSchema)
    assert result == {"name": "Tom", "age": 15}
    assert len(errors) == 2
    assert any("minimum length" in error["text"] for error in errors)
    assert any("minimum value" in error["text"] for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with bytes input
    bytes_yaml = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_yaml, TestSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []


# LLM-generated content at query #73
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a schema
    class TestSchema(Schema):
        name = Field(str, min_length=1)
        age = Field(int, minimum=0)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John\nage: invalid_int"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "name: J\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result == {"name": "J", "age": -5}
    assert len(errors) == 2
    assert any("min_length" in str(error) for error in errors)
    assert any("minimum" in str(error) for error in errors)

    # Test empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with bytes input
    yaml_bytes = b"name: Alice\nage: 25"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Alice", "age": 25}
    assert errors == []

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user:\n  name: Bob\n  age: 40\nitems:\n  - apple\n  - banana"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Bob", "age": 40}, "items": ["apple", "banana"]}
    assert errors == []


# LLM-generated content at query #74
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    yaml_content = "name: John Doe"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John Doe"}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = "name: [John Doe"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML that fails schema validation
    class NumberSchema(Schema):
        age = Field(int)

    yaml_content = "age: twenty"
    result, errors = validate_yaml(yaml_content, NumberSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, properties={
            "name": Field(str),
            "age": Field(int)
        })

    yaml_content = """
    user:
      name: Alice
      age: 30
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    yaml_content = "items: [1, 2, 3]"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with validation errors in list
    yaml_content = "items: [1, two, 3]"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result is None
    assert len(errors) == 1


# LLM-generated content at query #75
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML (parse error)
    content = "name: John\nage: :invalid"
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
user:
  name: Alice
  age: 30
items:
  - apple
  - banana
"""
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with different scalar types
    class TypesSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)
        tag = Field(str, allow_null=True)

    content = """
count: 42
price: 19.99
active: true
tag: null
"""
    result, errors = validate_yaml(content, TypesSchema)
    assert result == {
        "count": 42,
        "price": 19.99,
        "active": True,
        "tag": None
    }
    assert errors == []

    # Test YAML bytes input
    content = b"name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #76
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert errors == []
    assert result["name"] == "John"
    assert result["age"] == 30

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML validation error
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert errors != []
    assert "age" in errors[0].loc

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with bytes content
    yaml_bytes = b"name: Jane\nage: 25"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert errors == []
    assert result["name"] == "Jane"
    assert result["age"] == 25

    # Test nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user:\n  name: Alice\n  age: 28\nitems:\n  - item1\n  - item2"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert errors == []
    assert result["user"]["name"] == "Alice"
    assert result["items"] == ["item1", "item2"]

    # Test YAML with special characters
    yaml_content = "name: 'John Doe'\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert errors == []
    assert result["name"] == "John Doe"

    # Test YAML with boolean and null values
    class ExtendedSchema(Schema):
        active = Field(bool)
        optional = Field(typing.Optional[str])

    yaml_content = "active: true\noptional: null"
    result, errors = validate_yaml(yaml_content, ExtendedSchema)
    assert errors == []
    assert result["active"] is True
    assert result["optional"] is None


# LLM-generated content at query #77
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)

    content = "name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John\nage: "
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "parse_error"

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Bob"
    result, errors = validate_yaml(content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML content
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    content = """
user:
  name: Alice
  age: 30
items:
  - apple
  - banana
"""
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 30},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with type mismatches
    class TypedSchema(Schema):
        count = Field(int)

    content = "count: not_a_number"
    result, errors = validate_yaml(content, TypedSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "type"

    # Test bytes input
    content = b"name: John"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #78
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test with invalid YAML
    invalid_yaml = "name: John\nage: invalid_age"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test with YAML that doesn't match schema
    yaml_content = "name: John"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test with empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test with bytes content
    yaml_bytes = b"name: Jane\nage: 25"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Jane", "age": 25}
    assert errors == []


# LLM-generated content at query #79
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML mapping
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML sequence
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test valid YAML scalar
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"

    # Test YAML with different data types
    yaml_content = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test invalid YAML
    yaml_content = "key: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"

    # Test bytes content
    yaml_content = b"key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test nested structures
    yaml_content = """
    outer:
      inner:
        - item1
        - item2
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {
        "outer": {
            "inner": ["item1", "item2"]
        }
    }


# LLM-generated content at query #80
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML string
    yaml_str = "key: value"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"

    # Test invalid YAML
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"

    # Test YAML list
    yaml_list = "- item1\n- item2"
    token = tokenize_yaml(yaml_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML scalar
    yaml_scalar = "scalar"
    token = tokenize_yaml(yaml_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar"

    # Test YAML with different data types
    yaml_types = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(yaml_types)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test nested YAML structures
    nested_yaml = """
    outer:
      inner:
        - item1
        - item2
    """
    token = tokenize_yaml(nested_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {
        "outer": {
            "inner": ["item1", "item2"]
        }
    }


