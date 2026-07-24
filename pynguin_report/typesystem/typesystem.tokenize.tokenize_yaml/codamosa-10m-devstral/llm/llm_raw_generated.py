####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    assert len(result[1]) == 1
    assert "name" in result[1][0].text

    # Test YAML with a list schema
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - 1\n  - 2\n  - 3"
    result = validate_yaml(content, ListSchema)
    assert result[0] == {"items": [1, 2, 3]}
    assert result[1] == []

    # Test invalid YAML syntax
    content = "name: John\ninvalid: yaml: content"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert "parse_error" in e.code

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert "no_content" in e.code

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: John\n  age: 30"
    result = validate_yaml(content, NestedSchema)
    assert result[0] == {"user": {"name": "John", "age": 30}}
    assert result[1] == []

    # Test YAML with bytes content
    content = b"name: John"
    result = validate_yaml(content, SimpleSchema)
    assert result[0] == {"name": "John"}
    assert result[1] == []


# LLM-generated content at query #2
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
        tokenize_yaml("invalid: yaml: content")
    assert exc_info.value.code == "parse_error"

    # Test bytes content
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test YAML with nested structures
    token = tokenize_yaml("list:\n  - item1\n  - item2\nnested:\n  key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"list": ["item1", "item2"], "nested": {"key": "value"}}


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = """
name: John
age: 30
"""
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (parse error)
    invalid_yaml = """
name: John
age: thirty
"""
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = """
name: John
age: -5
"""
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error["field"] for error in errors)

    # Test empty YAML
    empty_yaml = ""
    with pytest.raises(ParseError):
        validate_yaml(empty_yaml, TestSchema)

    # Test YAML with missing required field
    incomplete_yaml = """
name: John
"""
    result, errors = validate_yaml(incomplete_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error["field"] for error in errors)

    # Test YAML with extra fields (if schema doesn't allow them)
    yaml_with_extra = """
name: John
age: 30
extra_field: value
"""
    result, errors = validate_yaml(yaml_with_extra, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []  # Assuming schema ignores extra fields

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        settings = Field(list)

    nested_yaml = """
user:
  name: John
  age: 30
settings:
  - dark_mode: true
  - notifications: false
"""
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "settings": [{"dark_mode": True}, {"notifications": False}]
    }
    assert errors == []

    # Test YAML with bytes input
    bytes_yaml = b"""
name: John
age: 30
"""
    result, errors = validate_yaml(bytes_yaml, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    yaml_content = "invalid: yaml: content: :"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test bytes content
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
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


# LLM-generated content at query #2
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

    # Test YAML with a list schema
    class ListSchema(Schema):
        items = Field(list, items=Field(str))

    yaml_content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test invalid YAML with a list schema
    yaml_content = "items:\n  - 123\n  - banana"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test YAML with a nested schema
    class NestedSchema(Schema):
        user = Field(dict, schema=Field(dict, keys=Field(str), values=Field(str)))

    yaml_content = "user:\n  name: John\n  age: '30'"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "John", "age": "30"}}
    assert errors == []

    # Test invalid YAML with a nested schema
    yaml_content = "user:\n  name: John\n  age: 30"
    result, errors = validate_yaml(yaml_content, NestedSchema)
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
    yaml_content = "name: John\ninvalid: yaml: content"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test YAML with bytes content
    yaml_content = b"name: John"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []

    # Test YAML with bytes content and invalid type
    yaml_content = b"name: 123"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"


# LLM-generated content at query #3
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

    # Test invalid YAML syntax
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"
    assert "mapping values are not allowed here" in exc_info.value.text

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


# LLM-generated content at query #4
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
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test invalid YAML syntax
    yaml_content = "name: [John"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, SimpleSchema)
    assert exc_info.value.code == "parse_error"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, properties={"name": Field(str), "age": Field(int)})

    yaml_content = """
    user:
        name: Jane
        age: 30
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Jane", "age": 30}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    yaml_content = """
    items:
        - 1
        - 2
        - 3
    """
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with validation errors in nested structures
    yaml_content = """
    user:
        name: Jane
        age: thirty
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test YAML with validation errors in lists
    yaml_content = """
    items:
        - 1
        - two
        - 3
    """
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"


# LLM-generated content at query #5
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

    # Test YAML with different types
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
    assert exc_info.value.position.char_index == len(yaml_content) - 1

    # Test bytes content
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


# LLM-generated content at query #6
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

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John Doe\nage: :invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int)

    yaml_content = "name: Jo\nage: 25"
    result, errors = validate_yaml(yaml_content, StrictSchema)
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

    # Test YAML with different scalar types
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


# LLM-generated content at query #7
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

    # Test invalid YAML (syntax error)
    content = "name: John\nage: 30\ninvalid: yaml: content"
    try:
        validate_yaml(content, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

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
    tags: [python, yaml]
    """
    result, errors = validate_yaml(content, TypesSchema)
    assert result == {
        "count": 42,
        "price": 19.99,
        "active": True,
        "tags": ["python", "yaml"]
    }
    assert errors == []

    # Test YAML with bytes content
    content = b"name: Test"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "Test"}
    assert errors == []


# LLM-generated content at query #8
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
    token = tokenize_yaml("{\"a\": 1}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}
    assert token.start == 0
    assert token.end == 7

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.text is not None
    assert exc_info.value.position is not None

    # Test bytes content
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a valid schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test valid YAML with a valid field
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test valid YAML with invalid schema
    class InvalidSchema(Schema):
        name = Field(int)  # Expects int, but YAML has str

    result, errors = validate_yaml(yaml_content, InvalidSchema)
    assert result is None
    assert len(errors) > 0

    # Test empty YAML
    with pytest.raises(ParseError):
        validate_yaml("", TestSchema)

    # Test YAML with positional validation errors
    class NestedSchema(Schema):
        user = Field(dict)
        details = Field(dict)

    nested_yaml = "user:\n  name: Jane\n  age: 25\ndetails:\n  invalid_key: value"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result is None
    assert len(errors) > 0

    # Test YAML with list
    list_yaml = "items:\n  - 1\n  - 2\n  - 3"
    class ListSchema(Schema):
        items = Field(list)

    result, errors = validate_yaml(list_yaml, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with bytes content
    bytes_content = b"name: John\nage: 30"
    result, errors = validate_yaml(bytes_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #10
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

    # Test invalid YAML syntax
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(invalid_yaml, SimpleSchema)
    assert exc_info.value.code == "parse_error"

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Bob"
    result = validate_yaml(yaml_content, StrictSchema)
    assert result[0] == {"name": "Bob"}
    assert len(result[1]) == 1
    assert result[1][0].code == "min_length"

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        tags = Field(list)

    yaml_content = """
    user:
        name: Alice
        age: 30
    tags:
        - python
        - testing
    """
    result = validate_yaml(yaml_content, NestedSchema)
    assert result[0]["user"]["name"] == "Alice"
    assert result[0]["tags"] == ["python", "testing"]
    assert result[1] == []

    # Test YAML with type validation
    class TypedSchema(Schema):
        count = Field(int)
        active = Field(bool)

    yaml_content = """
    count: 42
    active: true
    """
    result = validate_yaml(yaml_content, TypedSchema)
    assert result[0]["count"] == 42
    assert result[0]["active"] is True
    assert result[1] == []

    # Test YAML with type validation error
    yaml_content = """
    count: not_a_number
    active: true
    """
    result = validate_yaml(yaml_content, TypedSchema)
    assert len(result[1]) == 1
    assert result[1][0].code == "invalid_type"


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result == ({"name": "John", "age": 30}, [])

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: "
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML that fails schema validation
    invalid_schema_yaml = "name: John\nage: thirty"
    result = validate_yaml(invalid_schema_yaml, SimpleSchema)
    assert result[1] != []  # Should have validation errors

    # Test empty YAML
    with pytest.raises(ParseError):
        validate_yaml("", SimpleSchema)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user: {name: Alice, age: 25}\nitems: [1, 2, 3]"
    result = validate_yaml(nested_yaml, NestedSchema)
    assert result == ({"user": {"name": "Alice", "age": 25}, "items": [1, 2, 3]}, [])

    # Test YAML with bytes input
    bytes_yaml = b"name: Bob\nage: 40"
    result = validate_yaml(bytes_yaml, SimpleSchema)
    assert result == ({"name": "Bob", "age": 40}, [])


# LLM-generated content at query #12
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
    content = "name: John\nage: :invalid"
    try:
        validate_yaml(content, TestSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "invalid" in e.text.lower()

    # Test YAML that fails schema validation
    class TestSchemaRequired(Schema):
        name = Field(str, required=True)
        age = Field(int, required=True)

    content = "name: John"
    result, errors = validate_yaml(content, TestSchemaRequired)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test empty YAML content
    content = ""
    try:
        validate_yaml(content, TestSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test YAML with list
    class TestSchemaList(Schema):
        items = Field(list)

    content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(content, TestSchemaList)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with nested structure
    class TestSchemaNested(Schema):
        user = Field(dict)

    content = "user:\n  name: John\n  age: 30"
    result, errors = validate_yaml(content, TestSchemaNested)
    assert result == {"user": {"name": "John", "age": 30}}
    assert errors == []

    # Test YAML with type mismatch
    class TestSchemaTypes(Schema):
        count = Field(int)

    content = "count: not_a_number"
    result, errors = validate_yaml(content, TestSchemaTypes)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "type"


# LLM-generated content at query #13
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
    yaml_with_errors = "name: 123\nage: thirty"
    result, errors = validate_yaml(yaml_with_errors, SimpleSchema)
    assert result is None
    assert len(errors) == 2
    assert any("name" in error.msg for error in errors)
    assert any("age" in error.msg for error in errors)

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


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_yaml():
    # Test successful validation with a simple YAML string
    yaml_content = "key: value"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test successful validation with a YAML list
    yaml_content = "- item1\n- item2"
    field = Field(list)
    result, errors = validate_yaml(yaml_content, field)
    assert result == ["item1", "item2"]
    assert errors == []

    # Test validation failure with incorrect type
    yaml_content = "key: value"
    field = Field(int)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) > 0
    assert all(isinstance(error, dict) for error in errors)

    # Test parse error with invalid YAML
    yaml_content = "key: value\n  invalid yaml: [unclosed"
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) > 0
    assert all(isinstance(error, dict) for error in errors)

    # Test empty content error
    yaml_content = ""
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result is None
    assert len(errors) > 0
    assert all(isinstance(error, dict) for error in errors)

    # Test with bytes content
    yaml_content = b"key: value"
    field = Field(dict)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test with Schema validation
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test Schema validation failure
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert all(isinstance(error, dict) for error in errors)


# LLM-generated content at query #15
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
    yaml_content = "key: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"
    assert "end of the stream" in exc_info.value.text

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


# LLM-generated content at query #16
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

    # Test YAML with different scalar types
    scalar_yaml = "int: 42\nfloat: 3.14\nbool: true\nnull: null"
    token = tokenize_yaml(scalar_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"int": 42, "float": 3.14, "bool": True, "null": None}


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    schema = Schema(
        fields={
            "name": Field(str),
            "age": Field(int),
        }
    )
    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, schema)

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_with_errors, schema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", schema)
    assert exc_info.value.code == "no_content"

    # Test YAML with extra fields not in schema
    yaml_extra_fields = "name: John\nage: 30\nextra: field"
    result, errors = validate_yaml(yaml_extra_fields, schema)
    assert result == {"name": "John", "age": 30}
    assert len(errors) == 1
    assert errors[0].code == "extra_field"

    # Test YAML with missing required field
    yaml_missing_field = "age: 30"
    result, errors = validate_yaml(yaml_missing_field, schema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "required_field"

    # Test YAML with nested structure
    nested_schema = Schema(
        fields={
            "user": Field(
                Schema(
                    fields={
                        "name": Field(str),
                        "email": Field(str),
                    }
                )
            )
        }
    )
    nested_yaml = "user:\n  name: Jane\n  email: jane@example.com"
    result, errors = validate_yaml(nested_yaml, nested_schema)
    assert result == {"user": {"name": "Jane", "email": "jane@example.com"}}
    assert errors == []

    # Test YAML with list of items
    list_schema = Schema(
        fields={
            "items": Field(
                typing.List[Field(int)]
            )
        }
    )
    list_yaml = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(list_yaml, list_schema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with bytes content
    bytes_yaml = b"name: John\nage: 30"
    result, errors = validate_yaml(bytes_yaml, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #18
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

    # Test YAML that fails schema validation
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Bob"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "name" in errors[0].loc
    assert "String should have at least 5 characters" in errors[0].text

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

    # Test YAML with different data types
    class TypesSchema(Schema):
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
    result, errors = validate_yaml(yaml_content, TypesSchema)
    assert result == {
        "count": 42,
        "price": 19.99,
        "active": True,
        "tags": ["python", "testing"]
    }
    assert errors == []

    # Test YAML with null values
    class NullableSchema(Schema):
        value = Field(str, allow_null=True)

    yaml_content = "value: null"
    result, errors = validate_yaml(yaml_content, NullableSchema)
    assert result == {"value": None}
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: Test"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "Test"}
    assert errors == []


# LLM-generated content at query #19
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
    invalid_yaml = "name: John\nage: thirty"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML validation error
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
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
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []

    # Test YAML with special characters
    yaml_content = "name: José\nage: 35"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert result == {"name": "José", "age": 35}
    assert errors == []


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and valid schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result = validate_yaml(yaml_content, TestSchema)
    assert result == ({"name": "John", "age": 30}, [])

    # Test with invalid YAML
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test with valid YAML but invalid schema
    class StrictSchema(Schema):
        name = Field(str)
        age = Field(int, min_value=18)

    yaml_content = "name: John\nage: 10"
    result = validate_yaml(yaml_content, StrictSchema)
    assert result[1][0].code == "min_value"

    # Test with empty YAML
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test with bytes input
    yaml_bytes = b"name: Jane\nage: 25"
    result = validate_yaml(yaml_bytes, TestSchema)
    assert result == ({"name": "Jane", "age": 25}, [])


# LLM-generated content at query #21
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
    assert result == {"name": "John Doe", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = """
    name: John Doe
    age: thirty
    """
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = """
    name: John Doe
    age: -5
    """
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test empty YAML content
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", TestSchema)
    assert "No content" in str(excinfo.value)

    # Test YAML with missing required field
    incomplete_yaml = """
    name: John Doe
    """
    result, errors = validate_yaml(incomplete_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test YAML with extra fields
    yaml_with_extra = """
    name: John Doe
    age: 30
    extra_field: extra
    """
    result, errors = validate_yaml(yaml_with_extra, TestSchema)
    assert result == {"name": "John Doe", "age": 30}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        settings = Field(list)

    nested_yaml = """
    user:
        name: John Doe
        age: 30
    settings:
        - dark_mode: true
        - notifications: false
    """
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {
        "user": {"name": "John Doe", "age": 30},
        "settings": [{"dark_mode": True}, {"notifications": False}]
    }
    assert errors == []

    # Test YAML with bytes content
    bytes_yaml = b"""
    name: John Doe
    age: 30
    """
    result, errors = validate_yaml(bytes_yaml, TestSchema)
    assert result == {"name": "John Doe", "age": 30}
    assert errors == []


# LLM-generated content at query #22
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test YAML list parsing
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML scalar parsing
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"

    # Test YAML with different data types
    yaml_content = "int_val: 42\nfloat_val: 3.14\nbool_val: true\nnull_val: null"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"int_val": 42, "float_val": 3.14, "bool_val": True, "null_val": None}

    # Test YAML with nested structures
    yaml_content = "nested:\n  key: value\n  list:\n    - item1\n    - item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"nested": {"key": "value", "list": ["item1", "item2"]}}

    # Test YAML with bytes input
    yaml_content = b"key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty YAML content
    yaml_content = ""
    try:
        tokenize_yaml(yaml_content)
        assert False, "Expected ParseError for empty content"
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"

    # Test invalid YAML content
    yaml_content = "key: value: invalid"
    try:
        tokenize_yaml(yaml_content)
        assert False, "Expected ParseError for invalid YAML"
    except ParseError as e:
        assert e.code == "parse_error"


# LLM-generated content at query #23
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
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test valid YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Bob"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result == {"name": "Bob"}
    assert len(errors) == 1
    assert errors[0].text == "Must be at least 5 characters."

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
        items = Field(list, item_type=Field(str))

    yaml_content = """
    items:
        - apple
        - banana
    """
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: John"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #24
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

    # Test multiline YAML
    yaml_content = """
    key1: value1
    key2:
      - item1
      - item2
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key1": "value1", "key2": ["item1", "item2"]}


# LLM-generated content at query #25
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
    yaml_content = "key: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"
    assert "unclosed" in exc_info.value.text.lower()

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

    # Test YAML with special characters
    yaml_content = "key: 'value with \"quotes\"'"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": 'value with "quotes"'}


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    schema = Schema(fields={"name": str, "age": int})
    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML (parse error)
    yaml_content = "name: John\nage: invalid_yaml:"
    with pytest.raises(ParseError):
        validate_yaml(yaml_content, schema)

    # Test valid YAML with validation errors
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, schema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test empty YAML content
    yaml_content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(yaml_content, schema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    schema = Schema(fields={"user": {"name": str, "age": int}})
    yaml_content = "user:\n  name: Jane\n  age: 25"
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"user": {"name": "Jane", "age": 25}}
    assert errors == []

    # Test YAML with list
    schema = Schema(fields={"tags": [str]})
    yaml_content = "tags:\n  - python\n  - testing"
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"tags": ["python", "testing"]}
    assert errors == []

    # Test YAML with bytes content
    yaml_content_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_content_bytes, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #27
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"

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


# LLM-generated content at query #28
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
    invalid_yaml = "name: John\nage: thirty"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML that fails schema validation
    invalid_schema_yaml = "name: John\nage: -5"
    result, errors = validate_yaml(invalid_schema_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0

    # Test empty YAML content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, schema=TestSchema)

    nested_yaml = "user:\n  name: Jane\n  age: 25"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Jane", "age": 25}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    list_yaml = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(list_yaml, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with bytes input
    bytes_yaml = b"name: John\nage: 30"
    result, errors = validate_yaml(bytes_yaml, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #29
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

    # Test nested structures
    token = tokenize_yaml("key:\n  - item1\n  - item2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": ["item1", "item2"]}

    # Test various scalar types
    token = tokenize_yaml("int: 1\nfloat: 1.5\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.5, "bool": True, "null": None}

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value: extra")
    assert exc_info.value.code == "parse_error"

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test position tracking
    token = tokenize_yaml("key: value")
    assert token.position.line_no == 1
    assert token.position.column_no == 1


# LLM-generated content at query #30
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

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Jo"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "min_length" in errors[0].code

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, schema=SimpleSchema)

    yaml_content = "user:\n  name: Alice"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Alice"}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(str))

    yaml_content = "items:\n  - apple\n  - banana"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": ["apple", "banana"]}
    assert errors == []

    # Test bytes input
    yaml_bytes = b"name: Bob"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #31
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
    invalid_yaml = "name: [John Doe"  # Missing closing bracket
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML validation error
    class StrictSchema(Schema):
        age = Field(int, minimum=0, maximum=120)

    yaml_content = "age: 150"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "must be less than or equal to 120" in errors[0].text

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict, properties={
            "name": Field(str),
            "email": Field(str, format="email")
        })

    yaml_content = """
    user:
      name: Jane Doe
      email: jane@example.com
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {
            "name": "Jane Doe",
            "email": "jane@example.com"
        }
    }
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    yaml_content = "items: [1, 2, 3, 4]"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": [1, 2, 3, 4]}
    assert errors == []

    # Test YAML with validation error in list
    yaml_content = "items: [1, 2, 'three', 4]"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result is None
    assert len(errors) == 1
    assert "must be of type 'integer'" in errors[0].text

    # Test bytes input
    yaml_bytes = b"name: Test User"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "Test User"}
    assert errors == []


# LLM-generated content at query #32
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

    # Test YAML with different types
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
    assert exc_info.value.text is not None
    assert exc_info.value.position is not None

    # Test bytes content
    yaml_content = b"key: value"
    token = tokenize_yaml(yaml_content)
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


# LLM-generated content at query #33
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
    invalid_yaml = "name: [John Doe"  # Missing closing bracket
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        age = Field(int, minimum=0, maximum=120)

    yaml_with_error = "age: 150"
    result, errors = validate_yaml(yaml_with_error, StrictSchema)
    assert result == {"age": 150}
    assert len(errors) == 1
    assert "must be less than or equal to 120" in errors[0].text

    # Test empty YAML content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, SimpleSchema)
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

    # Test YAML with type conversion
    class TypeSchema(Schema):
        count = Field(int)
        price = Field(float)
        active = Field(bool)

    type_yaml = """
    count: 42
    price: 19.99
    active: true
    """
    result, errors = validate_yaml(type_yaml, TypeSchema)
    assert result == {"count": 42, "price": 19.99, "active": True}
    assert errors == []

    # Test YAML with null values
    class NullSchema(Schema):
        value = Field(str, allow_null=True)

    null_yaml = "value: null"
    result, errors = validate_yaml(null_yaml, NullSchema)
    assert result == {"value": None}
    assert errors == []

    # Test YAML with positional error reporting
    class PosSchema(Schema):
        email = Field(str, format="email")

    pos_yaml = "email: not-an-email"
    result, errors = validate_yaml(pos_yaml, PosSchema)
    assert result == {"email": "not-an-email"}
    assert len(errors) == 1
    assert errors[0].position is not None
    assert errors[0].position.char_index == 7  # Position of "not-an-email"


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
    invalid_yaml = "name: John\nage: invalid_int"
    result, errors = validate_yaml(invalid_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error["loc"] for error in errors)

    # Test YAML with missing required field
    incomplete_yaml = "name: Jane"
    result, errors = validate_yaml(incomplete_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("age" in error["loc"] for error in errors)

    # Test empty YAML content
    empty_yaml = ""
    result, errors = validate_yaml(empty_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any(error["code"] == "no_content" for error in errors)

    # Test YAML with extra fields not in schema
    extra_fields_yaml = "name: Bob\nage: 25\nextra: field"
    result, errors = validate_yaml(extra_fields_yaml, TestSchema)
    assert result == {"name": "Bob", "age": 25}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 28\nitems:\n  - item1\n  - item2"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 28}, "items": ["item1", "item2"]}
    assert errors == []

    # Test YAML with type mismatch
    type_mismatch_yaml = "name: 123\nage: thirty"
    result, errors = validate_yaml(type_mismatch_yaml, TestSchema)
    assert result is None
    assert len(errors) > 0
    assert any("name" in error["loc"] for error in errors)
    assert any("age" in error["loc"] for error in errors)


# LLM-generated content at query #35
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and valid schema
    class TestSchema(Schema):
        name = Field(str, min_length=1)
        age = Field(int, minimum=0)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test with valid YAML but invalid schema
    yaml_content = "name: J\nage: -5"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 2
    assert any("min_length" in error.code for error in errors)
    assert any("minimum" in error.code for error in errors)

    # Test with invalid YAML
    yaml_content = "name: John\nage: invalid"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test with empty YAML
    yaml_content = ""
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test with nested YAML structure
    class NestedSchema(Schema):
        user = Field(dict)
        items = Field(list)

    yaml_content = "user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "John", "age": 30}, "items": ["item1", "item2"]}
    assert errors == []

    # Test with bytes input
    yaml_content = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #36
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    content = "name: John Doe"
    value, errors = validate_yaml(content, TestSchema)
    assert value == {"name": "John Doe"}
    assert errors == []

    # Test invalid YAML syntax
    content = "name: John Doe\nage: invalid_yaml: 30"
    value, errors = validate_yaml(content, TestSchema)
    assert value is None
    assert len(errors) > 0
    assert any("parse_error" in error.code for error in errors)

    # Test YAML with validation errors
    class TestSchemaWithValidation(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=0)

    content = "name: Joe\nage: -5"
    value, errors = validate_yaml(content, TestSchemaWithValidation)
    assert value is None
    assert len(errors) > 0
    assert any("min_length" in error.code for error in errors)
    assert any("minimum" in error.code for error in errors)

    # Test empty YAML content
    content = ""
    value, errors = validate_yaml(content, TestSchema)
    assert value is None
    assert len(errors) > 0
    assert any("no_content" in error.code for error in errors)

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
    value, errors = validate_yaml(content, NestedSchema)
    assert value == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with bytes content
    content = b"name: Bob"
    value, errors = validate_yaml(content, TestSchema)
    assert value == {"name": "Bob"}
    assert errors == []


# LLM-generated content at query #37
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
    invalid_yaml = "name: John\nage: thirty"
    result, errors = validate_yaml(invalid_yaml, SimpleSchema)
    assert result is None
    assert len(errors) > 0
    assert errors[0].code == "parse_error"

    # Test YAML with validation errors
    yaml_with_errors = "name: John\nage: -5"
    result, errors = validate_yaml(yaml_with_errors, SimpleSchema)
    assert result is None
    assert len(errors) > 0
    assert errors[0].code == "validation_error"

    # Test empty YAML content
    empty_yaml = ""
    result, errors = validate_yaml(empty_yaml, SimpleSchema)
    assert result is None
    assert len(errors) > 0
    assert errors[0].code == "no_content"

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


# LLM-generated content at query #38
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

    # Test invalid YAML (syntax error)
    invalid_yaml = "name: John\nage: 30\ninvalid: [unclosed"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML that fails schema validation
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Bob"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert "name" in errors[0].text

    # Test empty YAML content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        tags = Field(list)

    yaml_content = """
    user:
        name: Alice
        age: 25
    tags:
        - python
        - testing
    """
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "tags": ["python", "testing"]
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


# LLM-generated content at query #39
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    yaml_content = "name: John\nage: 30"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML with syntax error
    invalid_yaml = "name: John\nage: : 30"
    schema = Schema(fields={"name": str, "age": int})
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, schema)

    # Test YAML with validation errors
    yaml_content = "name: John\nage: thirty"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_content, schema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test empty YAML content
    empty_yaml = ""
    schema = Schema(fields={"name": str})
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, schema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    yaml_content = "user:\n  name: Alice\n  age: 25"
    schema = Schema(fields={"user": {"name": str, "age": int}})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"user": {"name": "Alice", "age": 25}}
    assert errors == []

    # Test YAML with list
    yaml_content = "tags:\n  - python\n  - yaml\n  - test"
    schema = Schema(fields={"tags": [str]})
    result, errors = validate_yaml(yaml_content, schema)
    assert result == {"tags": ["python", "yaml", "test"]}
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: Bob\nage: 40"
    schema = Schema(fields={"name": str, "age": int})
    result, errors = validate_yaml(yaml_bytes, schema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []


# LLM-generated content at query #40
#--------------------------

```python
def test_tokenize_yaml():
    # Test with valid YAML string
    yaml_str = "key: value"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test with empty string
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test with invalid YAML
    invalid_yaml = "key: value: extra"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"

    # Test with nested structures
    nested_yaml = """
    outer:
      inner:
        - item1
        - item2
    """
    token = tokenize_yaml(nested_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"outer": {"inner": ["item1", "item2"]}}

    # Test with various scalar types
    scalar_yaml = """
    int_val: 42
    float_val: 3.14
    bool_val: true
    null_val: null
    """
    token = tokenize_yaml(scalar_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "null_val": None,
    }

    # Test with list at root
    list_yaml = """
    - item1
    - item2
    """
    token = tokenize_yaml(list_yaml)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML content with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML content
    invalid_yaml = "name: John\nage: not_a_number"
    result, errors = validate_yaml(invalid_yaml, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text.lower()

    # Test empty YAML content
    empty_yaml = ""
    result, errors = validate_yaml(empty_yaml, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test YAML with list content
    class ListSchema(Schema):
        items = Field(list)

    yaml_list = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(yaml_list, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        settings = Field(dict)

    yaml_nested = "user:\n  name: Alice\n  age: 25\nsettings:\n  theme: dark"
    result, errors = validate_yaml(yaml_nested, NestedSchema)
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "settings": {"theme": "dark"}
    }
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: Bob\nage: 40"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []

    # Test YAML with malformed structure
    malformed_yaml = "name: John\nage: 30\ninvalid_key"
    result, errors = validate_yaml(malformed_yaml, TestSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = """
    name: John
    age: 30
    """
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []

    # Test invalid YAML syntax
    invalid_yaml = """
    name: John
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
    assert result is None
    assert len(errors) == 2
    assert any("name" in error.msg for error in errors)
    assert any("age" in error.msg for error in errors)

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
    assert result == {
        "user": {"name": "Alice", "age": 25},
        "items": ["apple", "banana"]
    }
    assert errors == []

    # Test YAML with bytes content
    bytes_content = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_content, TestSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []


# LLM-generated content at query #3
#--------------------------

```python
def test_tokenize_yaml():
    # Test valid YAML string
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(yaml_content) - 1

    # Test valid YAML bytes
    yaml_bytes = b"key: value"
    token = tokenize_yaml(yaml_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test empty string
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

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

    # Test YAML with scalar
    yaml_scalar = "scalar_value"
    token = tokenize_yaml(yaml_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"

    # Test YAML with nested structures
    yaml_nested = "outer:\n  inner: value"
    token = tokenize_yaml(yaml_nested)
    assert isinstance(token, DictToken)
    assert token.value == {"outer": {"inner": "value"}}


# LLM-generated content at query #4
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
    assert errors[0].code == "min_length"

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
        items = Field(list, items=Field(int))

    yaml_content = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with bytes input
    yaml_bytes = b"name: Alice"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "Alice"}
    assert errors == []


# LLM-generated content at query #5
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
    assert len(errors) > 0
    assert any("age" in error for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict, schema=SimpleSchema)

    yaml_content = "user:\n  name: Jane\n  age: 25"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {"user": {"name": "Jane", "age": 25}}
    assert errors == []

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list, items=Field(int))

    yaml_content = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(yaml_content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with bytes content
    yaml_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #6
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
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("key: value: extra")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index == 9

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


# LLM-generated content at query #7
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
    invalid_yaml = "name: John\nage: invalid"
    try:
        validate_yaml(invalid_yaml, SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test YAML validation error
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test empty YAML content
    try:
        validate_yaml("", SimpleSchema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

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
    yaml_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}


# LLM-generated content at query #8
#--------------------------

```python
def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

    # Test valid YAML
    yaml_content = "key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test YAML with list
    yaml_content = "- item1\n- item2"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test YAML with scalar
    yaml_content = "scalar_value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == "scalar_value"

    # Test YAML with int
    yaml_content = "42"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test YAML with float
    yaml_content = "3.14"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test YAML with bool
    yaml_content = "true"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test YAML with null
    yaml_content = "null"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test invalid YAML
    yaml_content = "key: value: extra"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"

    # Test bytes content
    yaml_content = b"key: value"
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}


# LLM-generated content at query #9
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

    # Test YAML with validation errors
    class StrictSchema(Schema):
        age = Field(int, minimum=0)

    yaml_with_error = "age: -5"
    result, errors = validate_yaml(yaml_with_error, StrictSchema)
    assert result is None
    assert len(errors) > 0
    assert any("minimum" in error.text for error in errors)

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
        items = Field(list, items=Field(int))

    list_yaml = "items: [1, 2, 3]"
    result, errors = validate_yaml(list_yaml, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML content and a simple field validator
    yaml_content = "key: value"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test with invalid YAML content
    invalid_yaml = "key: value\n  invalid yaml: ["
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, field)

    # Test with empty YAML content
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, field)
    assert exc_info.value.code == "no_content"

    # Test with YAML content that fails validation
    yaml_content = "key: 123"
    field = Field(str)
    result, errors = validate_yaml(yaml_content, field)
    assert result == {"key": 123}
    assert len(errors) > 0

    # Test with bytes content
    yaml_bytes = b"key: value"
    result, errors = validate_yaml(yaml_bytes, field)
    assert result == {"key": "value"}
    assert errors == []

    # Test with a schema validator
    class TestSchema(Schema):
        name = Field(str)

    yaml_content = "name: test"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": "test"}
    assert errors == []

    # Test with a schema validator and invalid data
    yaml_content = "name: 123"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result == {"name": 123}
    assert len(errors) > 0


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

    # Test invalid YAML syntax
    yaml_content = "key: [unclosed"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(yaml_content)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index == 9

    # Test bytes input
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


# LLM-generated content at query #12
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

    # Test YAML that fails schema validation
    yaml_content = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_content, TestSchema)
    assert result is None
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

    yaml_content = "user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2"
    result, errors = validate_yaml(yaml_content, NestedSchema)
    assert result == {
        "user": {"name": "John", "age": 30},
        "items": ["item1", "item2"]
    }
    assert errors == []

    # Test YAML with bytes input
    yaml_bytes = b"name: John\nage: 30"
    result, errors = validate_yaml(yaml_bytes, TestSchema)
    assert result == {"name": "John", "age": 30}
    assert errors == []


# LLM-generated content at query #13
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
    invalid_yaml = "name: John\nage: :30"
    schema = Schema(fields={"name": str, "age": int})
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, schema)

    # Test YAML with validation errors
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


# LLM-generated content at query #14
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

    # Test invalid YAML syntax
    invalid_yaml = """
    name: John Doe
    age: invalid
    """
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = """
    name: ""
    age: -5
    """
    result, errors = validate_yaml(yaml_with_errors, TestSchema)
    assert len(errors) == 2
    assert any("name" in error.msg for error in errors)
    assert any("age" in error.msg for error in errors)

    # Test empty YAML content
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with missing required field
    incomplete_yaml = """
    name: Jane Doe
    """
    result, errors = validate_yaml(incomplete_yaml, TestSchema)
    assert len(errors) == 1
    assert "age" in errors[0].msg

    # Test YAML with extra fields (should be ignored by default)
    yaml_with_extra = """
    name: Bob
    age: 25
    extra_field: "should be ignored"
    """
    result, errors = validate_yaml(yaml_with_extra, TestSchema)
    assert errors == []
    assert result["name"] == "Bob"
    assert result["age"] == 25
    assert "extra_field" not in result

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
        settings = Field(list)

    nested_yaml = """
    user:
      name: Alice
      email: alice@example.com
    settings:
      - dark_mode: true
      - notifications: false
    """
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert result["user"]["name"] == "Alice"
    assert len(result["settings"]) == 2


# LLM-generated content at query #15
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

    # Test YAML with null value
    class NullableSchema(Schema):
        value = Field(str, allow_null=True)

    content = "value: null"
    result, errors = validate_yaml(content, NullableSchema)
    assert result == {"value": None}
    assert errors == []

    # Test YAML with bytes content
    content_bytes = b"name: John"
    result, errors = validate_yaml(content_bytes, SimpleSchema)
    assert result == {"name": "John"}
    assert errors == []


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_yaml():
    # Test valid YAML with a simple schema
    class TestSchema(Schema):
        name = Field(str)

    yaml_content = "name: John"
    result = validate_yaml(yaml_content, TestSchema)
    assert result == ({"name": "John"}, [])

    # Test invalid YAML
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    yaml_with_errors = "age: not_a_number"
    class AgeSchema(Schema):
        age = Field(int)

    result = validate_yaml(yaml_with_errors, AgeSchema)
    assert result[0] is None
    assert len(result[1]) > 0

    # Test empty YAML
    empty_yaml = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(empty_yaml, TestSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with nested structures
    nested_yaml = """
    user:
        name: Alice
        age: 30
    """
    class NestedSchema(Schema):
        user = Field(dict)

    result = validate_yaml(nested_yaml, NestedSchema)
    assert result == ({"user": {"name": "Alice", "age": 30}}, [])

    # Test YAML with list
    list_yaml = """
    items:
        - apple
        - banana
    """
    class ListSchema(Schema):
        items = Field(list)

    result = validate_yaml(list_yaml, ListSchema)
    assert result == ({"items": ["apple", "banana"]}, [])


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
    invalid_yaml = "name: [John"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, SimpleSchema)

    # Test YAML that fails schema validation
    class NumberSchema(Schema):
        age = Field(int)

    content = "age: twenty"
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
        items = Field(list, children=Field(int))

    content = "items: [1, 2, 3]"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with validation errors in nested structure
    content = """
    user:
      name: Bob
      age: not_a_number
    """
    result, errors = validate_yaml(content, NestedSchema)
    assert result is None
    assert len(errors) == 1
    assert "age" in errors[0].text


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

    # Test invalid YAML syntax
    invalid_yaml = "name: John\nage: invalid"
    with pytest.raises(ParseError):
        validate_yaml(invalid_yaml, TestSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)
        age = Field(int, minimum=18)

    yaml_with_errors = "name: Jo\nage: 17"
    result, errors = validate_yaml(yaml_with_errors, StrictSchema)
    assert result == {"name": "Jo", "age": 17}
    assert len(errors) == 2
    assert any("minimum length" in error.text for error in errors)
    assert any("minimum value" in error.text for error in errors)

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

    # Test YAML with bytes content
    bytes_content = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_content, TestSchema)
    assert result == {"name": "Bob", "age": 40}
    assert errors == []


# LLM-generated content at query #19
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

    # Test invalid YAML
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index >= 0

    # Test bytes content
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"


# LLM-generated content at query #20
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

    # Test YAML with positional validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    yaml_content = "name: Bob"
    result, errors = validate_yaml(yaml_content, StrictSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "too_short"


# LLM-generated content at query #21
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
    try:
        validate_yaml(invalid_yaml, SimpleSchema)
    except ParseError as e:
        assert e.code == "parse_error"
        assert "invalid_int" in str(e)

    # Test YAML validation error
    yaml_with_error = "name: John\nage: thirty"
    result, errors = validate_yaml(yaml_with_error, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"
    assert errors[0].field == "age"

    # Test empty YAML content
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, SimpleSchema)
    except ParseError as e:
        assert e.code == "no_content"
        assert "No content" in str(e)

    # Test YAML with nested structures
    class NestedSchema(Schema):
        user = Field(dict)
        tags = Field(list)

    nested_yaml = "user:\n  name: Alice\n  age: 25\n tags: [python, yaml]"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert result == {"user": {"name": "Alice", "age": 25}, "tags": ["python", "yaml"]}

    # Test YAML with bytes input
    bytes_yaml = b"name: Bob\nage: 40"
    result, errors = validate_yaml(bytes_yaml, SimpleSchema)
    assert errors == []
    assert result == {"name": "Bob", "age": 40}


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

    # Test invalid YAML
    content = "name: John\nage: invalid_int"
    with pytest.raises(ParseError):
        validate_yaml(content, SimpleSchema)

    # Test YAML with validation errors
    class StrictSchema(Schema):
        name = Field(str, min_length=5)

    content = "name: Jo"
    result, errors = validate_yaml(content, StrictSchema)
    assert result == {"name": "Jo"}
    assert len(errors) == 1
    assert errors[0].code == "min_length"

    # Test empty YAML
    content = ""
    with pytest.raises(ParseError) as exc_info:
        validate_yaml(content, SimpleSchema)
    assert exc_info.value.code == "no_content"

    # Test YAML with list
    class ListSchema(Schema):
        items = Field(list)

    content = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(content, ListSchema)
    assert result == {"items": [1, 2, 3]}
    assert errors == []

    # Test YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)

    content = "user:\n  name: Alice\n  age: 30"
    result, errors = validate_yaml(content, NestedSchema)
    assert result == {"user": {"name": "Alice", "age": 30}}
    assert errors == []

    # Test YAML with bytes content
    content = b"name: Bob"
    result, errors = validate_yaml(content, SimpleSchema)
    assert result == {"name": "Bob"}
    assert errors == []


