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


