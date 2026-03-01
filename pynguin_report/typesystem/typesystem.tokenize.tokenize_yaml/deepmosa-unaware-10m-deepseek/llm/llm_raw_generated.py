####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer
    
    # Test valid YAML with string field
    string_field = String(max_length=10)
    result = validate_yaml("hello: world", string_field)
    assert result == ("world", [])
    
    # Test invalid YAML with string field (too long)
    result = validate_yaml("hello: thisistoolong", string_field)
    assert len(result[1]) == 1
    assert result[1][0].code == "max_length"
    
    # Test valid YAML with integer field
    int_field = Integer(minimum=0, maximum=100)
    result = validate_yaml("value: 42", int_field)
    assert result == (42, [])
    
    # Test invalid YAML with integer field (out of range)
    result = validate_yaml("value: 150", int_field)
    assert len(result[1]) == 1
    assert result[1][0].code == "maximum"
    
    # Test with schema validation
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    
    class PersonSchema(Schema):
        name = String(max_length=50)
        age = Integer(minimum=0, maximum=150)
    
    # Test valid YAML with schema
    yaml_content = """
    name: John Doe
    age: 30
    """
    result = validate_yaml(yaml_content, PersonSchema)
    assert result == ({"name": "John Doe", "age": 30}, [])
    
    # Test invalid YAML with schema (multiple errors)
    yaml_content = """
    name: This name is way too long to be valid according to our schema
    age: 200
    """
    result = validate_yaml(yaml_content, PersonSchema)
    assert len(result[1]) == 2
    
    # Test empty YAML content
    try:
        validate_yaml("", PersonSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "no_content"
    
    # Test YAML with only whitespace
    try:
        validate_yaml("   \n  \t  \n", PersonSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "no_content"
    
    # Test invalid YAML syntax
    invalid_yaml = """
    name: John
    age: thirty
    extra: [unclosed list
    """
    try:
        validate_yaml(invalid_yaml, PersonSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "parse_error"
    
    # Test bytes input
    bytes_content = b"name: Alice\nage: 25"
    result = validate_yaml(bytes_content, PersonSchema)
    assert result == ({"name": "Alice", "age": 25}, [])
    
    # Test nested structures
    class NestedSchema(Schema):
        person = PersonSchema
        tags = String()
    
    yaml_content = """
    person:
      name: Bob
      age: 40
    tags: developer
    """
    result = validate_yaml(yaml_content, NestedSchema)
    assert result == ({
        "person": {"name": "Bob", "age": 40},
        "tags": "developer"
    }, [])
    
    # Test with list validation
    from typesystem.fields import Array
    
    array_field = Array(items=Integer())
    result = validate_yaml("[1, 2, 3]", array_field)
    assert result == ([1, 2, 3], [])
    
    # Test with invalid list items
    result = validate_yaml("[1, 'two', 3]", array_field)
    assert len(result[1]) == 1
    
    # Test boolean values
    class BoolSchema(Schema):
        active = Boolean()
    
    result = validate_yaml("active: true", BoolSchema)
    assert result == ({"active": True}, [])
    
    result = validate_yaml("active: false", BoolSchema)
    assert result == ({"active": False}, [])
    
    # Test null values
    class OptionalSchema(Schema):
        name = String(allow_null=True)
    
    result = validate_yaml("name: null", OptionalSchema)
    assert result == ({"name": None}, [])
    
    # Test with default values in schema
    class DefaultSchema(Schema):
        name = String(default="Unknown")
        count = Integer(default=0)
    
    result = validate_yaml("{}", DefaultSchema)
    assert result == ({"name": "Unknown", "count": 0}, [])


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_yaml():
    # Mock yaml module to test both import scenarios
    import sys
    from unittest.mock import Mock, patch, MagicMock
    
    # Test 1: yaml not installed
    with patch.dict(sys.modules, {'yaml': None}):
        from typesystem.tokenize.yaml import validate_yaml
        
        try:
            validate_yaml("test: value", Mock())
            assert False, "Should have raised assertion error"
        except AssertionError as e:
            assert "'pyyaml' must be installed." in str(e)
    
    # Restore yaml module for other tests
    import yaml
    
    # Test 2: Valid YAML with simple schema validation
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    content = "name: John\nage: 25"
    value, errors = validate_yaml(content, TestSchema)
    
    assert errors == []
    assert value == {"name": "John", "age": 25}
    
    # Test 3: Invalid YAML - parse error
    invalid_content = "name: John\n  age: 25"  # Invalid indentation
    value, errors = validate_yaml(invalid_content, TestSchema)
    
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert errors[0].text.startswith("mapping values are not allowed here")
    
    # Test 4: Empty content
    empty_content = ""
    value, errors = validate_yaml(empty_content, TestSchema)
    
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    assert errors[0].text == "No content."
    
    # Test 5: Bytes input
    bytes_content = b"name: Alice\nage: 30"
    value, errors = validate_yaml(bytes_content, TestSchema)
    
    assert errors == []
    assert value == {"name": "Alice", "age": 30}
    
    # Test 6: Validation errors
    invalid_data = "name: VeryLongNameExceedsLimit\nage: -5"
    value, errors = validate_yaml(invalid_data, TestSchema)
    
    assert len(errors) == 2
    error_codes = [error.code for error in errors]
    assert "max_length" in error_codes
    assert "minimum" in error_codes
    
    # Test 7: Valid YAML with nested structure
    class NestedSchema(Schema):
        user = TestSchema
        active = Boolean()
    
    from typesystem.fields import Boolean
    
    nested_content = """
    user:
      name: Bob
      age: 40
    active: true
    """
    
    value, errors = validate_yaml(nested_content, NestedSchema)
    
    assert errors == []
    assert value["user"]["name"] == "Bob"
    assert value["user"]["age"] == 40
    assert value["active"] is True
    
    # Test 8: Field validator instead of Schema
    from typesystem.fields import Array
    
    array_field = Array(items=String())
    array_content = "- item1\n- item2\n- item3"
    
    value, errors = validate_yaml(array_content, array_field)
    
    assert errors == []
    assert value == ["item1", "item2", "item3"]
    
    # Test 9: YAML with special types (int, float, bool, null)
    special_content = """
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    """
    
    class SpecialSchema(Schema):
        integer = Integer()
        float = Float()
        boolean = Boolean()
        null_value = Field(allow_null=True)
    
    from typesystem.fields import Float
    
    value, errors = validate_yaml(special_content, SpecialSchema)
    
    assert errors == []
    assert value["integer"] == 42
    assert value["float"] == 3.14
    assert value["boolean"] is True
    assert value["null_value"] is None
    
    # Test 10: Complex YAML error with position
    malformed_content = "key: [unclosed list"
    value, errors = validate_yaml(malformed_content, Mock())
    
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert "did not find expected" in errors[0].text.lower()


# LLM-generated content at query #3
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar types
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 4
    
    # Test integer
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.start == 0
    assert result.end == 1
    
    # Test float
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start == 0
    assert result.end == 3
    
    # Test boolean true
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3
    
    # Test boolean false
    result = tokenize_yaml("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4
    
    # Test null
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3
    
    # Test list
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert isinstance(result.value[0], ScalarToken)
    assert result.value[0].value == "item1"
    assert isinstance(result.value[1], ScalarToken)
    assert result.value[1].value == "item2"
    
    # Test dict
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert "key" in result.value
    assert isinstance(result.value["key"], ScalarToken)
    assert result.value["key"].value == "value"
    
    # Test nested structure
    result = tokenize_yaml("key:\n  nested: 42")
    assert isinstance(result, DictToken)
    nested_dict = result.value["key"]
    assert isinstance(nested_dict, DictToken)
    assert nested_dict.value["nested"].value == 42
    
    # Test bytes input
    result = tokenize_yaml(b"hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test empty string raises ParseError
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0
    
    # Test whitespace only raises ParseError
    try:
        tokenize_yaml("   \n  \t  ")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid YAML raises ParseError
    try:
        tokenize_yaml("key: [unclosed")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "parse_error" in e.code
    
    # Test with multiline string
    result = tokenize_yaml("|\n  line1\n  line2")
    assert isinstance(result, ScalarToken)
    assert result.value == "line1\nline2\n"
    
    # Test complex structure
    yaml_content = """
    name: John
    age: 30
    hobbies:
      - reading
      - hiking
    active: true
    """
    result = tokenize_yaml(yaml_content)
    assert isinstance(result, DictToken)
    assert result.value["name"].value == "John"
    assert result.value["age"].value == 30
    assert result.value["active"].value is True
    hobbies = result.value["hobbies"]
    assert isinstance(hobbies, ListToken)
    assert len(hobbies.value) == 2


# LLM-generated content at query #4
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.start == 0
    assert token.end == 9
    assert token.value == {"key": "value"}
    
    # Test nested structures
    content = """
    outer:
      inner: nested
      list:
        - item1
        - item2
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert "outer" in token.value
    assert isinstance(token.value["outer"], dict)
    assert token.value["outer"]["inner"] == "nested"
    assert isinstance(token.value["outer"]["list"], list)
    
    # Test different data types
    content = """
    string: hello
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    """
    token = tokenize_yaml(content)
    assert token.value["string"] == "hello"
    assert token.value["integer"] == 42
    assert token.value["float"] == 3.14
    assert token.value["boolean"] is True
    assert token.value["null_value"] is None
    
    # Test list parsing
    content = "- first\n- second\n- third"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["first", "second", "third"]
    
    # Test empty string
    content = ""
    try:
        tokenize_yaml(content)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
    
    # Test whitespace-only string
    content = "   \n  \t  \n"
    try:
        tokenize_yaml(content)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test bytes input
    content = b"key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test invalid YAML
    content = "key: [unclosed list"
    try:
        tokenize_yaml(content)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "parse_error" in exc.code
    
    # Test scalar token
    content = "simple string"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "simple string"
    
    # Test complex nested structure with correct positions
    content = "outer:\n  inner: test"
    token = tokenize_yaml(content)
    assert token.start == 0
    assert token.end == len(content) - 1
    
    # Test YAML with comments
    content = """
    # This is a comment
    key: value  # inline comment
    """
    token = tokenize_yaml(content)
    assert token.value == {"key": "value"}
    
    # Test special characters in strings
    content = "special: 'quoted string with \"quotes\"'"
    token = tokenize_yaml(content)
    assert token.value["special"] == 'quoted string with "quotes"'


# LLM-generated content at query #5
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar types
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert result.start == 0
    assert result.end == 9
    assert "key" in result.value
    assert isinstance(result.value["key"], ScalarToken)
    assert result.value["key"].value == "value"
    
    # Test integer
    result = tokenize_yaml("number: 42")
    assert isinstance(result.value["number"], ScalarToken)
    assert result.value["number"].value == 42
    
    # Test float
    result = tokenize_yaml("float: 3.14")
    assert isinstance(result.value["float"], ScalarToken)
    assert result.value["float"].value == 3.14
    
    # Test boolean
    result = tokenize_yaml("flag: true")
    assert isinstance(result.value["flag"], ScalarToken)
    assert result.value["flag"].value is True
    
    # Test null
    result = tokenize_yaml("empty: null")
    assert isinstance(result.value["empty"], ScalarToken)
    assert result.value["empty"].value is None
    
    # Test list
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(item, ScalarToken) for item in result.value)
    
    # Test nested structure
    result = tokenize_yaml("""
    users:
      - name: Alice
        age: 30
      - name: Bob
        age: 25
    """)
    assert isinstance(result, DictToken)
    users = result.value["users"]
    assert isinstance(users, ListToken)
    assert len(users.value) == 2
    assert users.value[0].value["name"].value == "Alice"
    assert users.value[0].value["age"].value == 30
    
    # Test bytes input
    result = tokenize_yaml(b"key: value")
    assert isinstance(result, DictToken)
    assert result.value["key"].value == "value"
    
    # Test empty string
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test whitespace only
    try:
        tokenize_yaml("   \n  \t  \n")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "parse_error" in e.code
    
    # Test malformed YAML
    try:
        tokenize_yaml("key: : value")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test content property
    content = "key: value"
    result = tokenize_yaml(content)
    assert result.content == content
    assert result.value["key"].content == content
    
    # Test position tracking
    content = """first: value1
second: value2"""
    result = tokenize_yaml(content)
    assert result.value["first"].start == 0
    assert result.value["first"].end == 12
    assert result.value["second"].start == 13
    assert result.value["second"].end == 26


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9
    
    # Test nested structures
    content = """
    name: John
    age: 30
    hobbies:
      - reading
      - hiking
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["name"] == "John"
    assert token.value["age"] == 30
    assert isinstance(token.value["hobbies"], list)
    
    # Test different data types
    content = """
    string: hello
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    """
    token = tokenize_yaml(content)
    assert token.value["string"] == "hello"
    assert token.value["integer"] == 42
    assert token.value["float"] == 3.14
    assert token.value["boolean"] is True
    assert token.value["null_value"] is None
    
    # Test list parsing
    content = "- item1\n- item2\n- item3"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2", "item3"]
    
    # Test empty string
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test bytes input
    content = b"key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed list")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test scalar values
    content = "single_value"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "single_value"
    
    # Test complex nested structure
    content = """
    users:
      - name: Alice
        age: 25
        active: true
      - name: Bob
        age: 30
        active: false
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert len(token.value["users"]) == 2
    assert token.value["users"][0]["name"] == "Alice"
    assert token.value["users"][1]["age"] == 30
    
    # Test position information
    content = "first: value1\nsecond: value2"
    token = tokenize_yaml(content)
    assert token.start == 0
    assert token.end == len(content) - 1


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.start == 0
    assert token.end == 9
    assert token.value == {"key": "value"}
    
    # Test nested structures
    content = """
    outer:
      inner: nested
      list:
        - item1
        - item2
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    outer = token.value["outer"]
    assert isinstance(outer, DictToken)
    assert outer.value["inner"].value == "nested"
    assert isinstance(outer.value["list"], ListToken)
    
    # Test different data types
    content = """
    string: hello
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    """
    token = tokenize_yaml(content)
    values = token.value
    assert values["string"].value == "hello"
    assert values["integer"].value == 42
    assert values["float"].value == 3.14
    assert values["boolean"].value is True
    assert values["null_value"].value is None
    
    # Test list parsing
    content = "- first\n- second\n- third"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert token.value[0].value == "first"
    
    # Test empty content
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
    
    # Test bytes input
    content = b"key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "parse_error" in exc.code
    
    # Test scalar values
    content = "single_value"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "single_value"
    
    # Test complex nested structure
    content = """
    users:
      - name: Alice
        age: 30
        active: true
      - name: Bob
        age: 25
        active: false
    """
    token = tokenize_yaml(content)
    users = token.value["users"]
    assert isinstance(users, ListToken)
    assert len(users.value) == 2
    assert users.value[0].value["name"].value == "Alice"
    assert users.value[1].value["age"].value == 25
    
    # Test positions are correctly calculated
    content = "first:\n  second: value"
    token = tokenize_yaml(content)
    assert token.start == 0
    assert token.end == len(content) - 1


# LLM-generated content at query #8
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar types
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 4
    
    # Test integer
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.start == 0
    assert result.end == 1
    
    # Test float
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start == 0
    assert result.end == 3
    
    # Test boolean
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3
    
    # Test null
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3
    
    # Test list
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert isinstance(result.value[0], ScalarToken)
    assert result.value[0].value == "item1"
    assert isinstance(result.value[1], ScalarToken)
    assert result.value[1].value == "item2"
    
    # Test dict
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert "key" in result.value
    assert isinstance(result.value["key"], ScalarToken)
    assert result.value["key"].value == "value"
    
    # Test nested structure
    result = tokenize_yaml("key:\n  nested: 42")
    assert isinstance(result, DictToken)
    nested_dict = result.value["key"].value
    assert isinstance(nested_dict, DictToken)
    assert nested_dict.value["nested"].value == 42
    
    # Test bytes input
    result = tokenize_yaml(b"test")
    assert isinstance(result, ScalarToken)
    assert result.value == "test"
    
    # Test empty string
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0
    
    # Test whitespace only
    try:
        tokenize_yaml("   \n  \t  \n")
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "parse_error" in exc.code
    
    # Test malformed YAML
    try:
        tokenize_yaml("key: : value")
    except ParseError as exc:
        assert exc.code == "parse_error"


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_yaml():
    # Test basic validation with simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}
    
    # Test validation error
    content = "name: John\nage: not_an_int"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type"
    assert errors[0].text == "Must be a number."
    
    # Test with nested schema
    class NestedSchema(Schema):
        user = SimpleSchema
    
    content = "user:\n  name: Alice\n  age: 25"
    value, errors = validate_yaml(content, NestedSchema)
    assert errors == []
    assert value == {"user": {"name": "Alice", "age": 25}}
    
    # Test with list field
    class ListSchema(Schema):
        items = Field(list, items=Field(int))
    
    content = "items:\n  - 1\n  - 2\n  - 3"
    value, errors = validate_yaml(content, ListSchema)
    assert errors == []
    assert value == {"items": [1, 2, 3]}
    
    # Test with required field
    class RequiredSchema(Schema):
        required = Field(str, required=True)
        optional = Field(str, required=False)
    
    content = "optional: test"
    value, errors = validate_yaml(content, RequiredSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"
    
    # Test with bytes input
    content = b"name: Bob\nage: 40"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "Bob", "age": 40}
    
    # Test with empty content
    content = ""
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test with invalid YAML syntax
    content = "name: John\n: invalid"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    
    # Test with Field validator directly
    field = Field(str)
    content = '"test string"'
    value, errors = validate_yaml(content, field)
    assert errors == []
    assert value == "test string"
    
    # Test with boolean field
    class BoolSchema(Schema):
        active = Field(bool)
    
    content = "active: true"
    value, errors = validate_yaml(content, BoolSchema)
    assert errors == []
    assert value == {"active": True}
    
    # Test with null value
    class NullableSchema(Schema):
        value = Field(str, allow_null=True)
    
    content = "value: null"
    value, errors = validate_yaml(content, NullableSchema)
    assert errors == []
    assert value == {"value": None}


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}
    
    # Test with invalid YAML (parse error)
    invalid_content = "name: John\n  age: 30"  # Invalid indentation
    result, errors = validate_yaml(invalid_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert errors[0].position is not None
    
    # Test with validation error
    invalid_data = "name: John\nage: 'thirty'"  # age is string instead of int
    result, errors = validate_yaml(invalid_data, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type"
    assert "age" in errors[0].text
    
    # Test with empty content
    empty_content = ""
    result, errors = validate_yaml(empty_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test with bytes input
    bytes_content = b"name: Alice\nage: 25"
    result, errors = validate_yaml(bytes_content, SimpleSchema)
    assert errors == []
    assert result == {"name": "Alice", "age": 25}
    
    # Test with nested structure
    class NestedSchema(Schema):
        user = Field(dict, properties={"name": Field(str), "details": Field(dict)})
    
    nested_content = "user:\n  name: Bob\n  details:\n    active: true"
    result, errors = validate_yaml(nested_content, NestedSchema)
    assert errors == []
    
    # Test with list validation
    class ListSchema(Schema):
        items = Field(list, items=Field(int))
    
    list_content = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(list_content, ListSchema)
    assert errors == []
    assert result == {"items": [1, 2, 3]}
    
    # Test with Field validator directly (not Schema)
    field_validator = Field(str)
    result, errors = validate_yaml("'simple string'", field_validator)
    assert errors == []
    assert result == "simple string"
    
    # Test with required field missing
    required_content = "name: John"  # age is missing
    result, errors = validate_yaml(required_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer
    
    # Test valid YAML with string field
    string_field = String(max_length=10)
    result = validate_yaml("hello: world", string_field)
    assert result == ("world", [])
    
    # Test invalid YAML with string field (too long)
    result = validate_yaml("hello: thisiswaytoolong", string_field)
    assert len(result[1]) == 1
    assert result[1][0].code == "max_length"
    
    # Test valid YAML with integer field
    int_field = Integer(minimum=0, maximum=100)
    result = validate_yaml("value: 42", int_field)
    assert result == (42, [])
    
    # Test invalid YAML with integer field (out of range)
    result = validate_yaml("value: 150", int_field)
    assert len(result[1]) == 1
    assert result[1][0].code == "maximum"
    
    # Test with schema validation
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    
    class PersonSchema(Schema):
        name = String(max_length=50)
        age = Integer(minimum=0, maximum=150)
    
    # Test valid YAML with schema
    yaml_content = """
    name: John Doe
    age: 30
    """
    result = validate_yaml(yaml_content, PersonSchema)
    assert result[0] == {"name": "John Doe", "age": 30}
    assert result[1] == []
    
    # Test invalid YAML with schema (multiple errors)
    yaml_content = """
    name: This name is way too long to be valid according to our schema
    age: 200
    """
    result = validate_yaml(yaml_content, PersonSchema)
    assert len(result[1]) == 2
    error_codes = {error.code for error in result[1]}
    assert "max_length" in error_codes
    assert "maximum" in error_codes
    
    # Test empty YAML content
    try:
        validate_yaml("", PersonSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "no_content"
    
    # Test YAML with only whitespace
    try:
        validate_yaml("   \n  \t  \n", PersonSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "no_content"
    
    # Test invalid YAML syntax
    invalid_yaml = """
    name: John
    age: thirty
      extra: indent
    """
    try:
        validate_yaml(invalid_yaml, PersonSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "parse_error"
    
    # Test bytes input
    bytes_content = b"name: Alice\nage: 25"
    result = validate_yaml(bytes_content, PersonSchema)
    assert result[0] == {"name": "Alice", "age": 25}
    assert result[1] == []
    
    # Test nested structures with schema
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class UserSchema(Schema):
        name = String()
        address = AddressSchema
    
    yaml_content = """
    name: Bob
    address:
      street: 123 Main St
      city: Anytown
    """
    result = validate_yaml(yaml_content, UserSchema)
    assert result[0]["name"] == "Bob"
    assert result[0]["address"]["street"] == "123 Main St"
    assert result[0]["address"]["city"] == "Anytown"
    assert result[1] == []
    
    # Test with required fields
    class RequiredSchema(Schema):
        required_field = String(required=True)
        optional_field = String(required=False)
    
    # Missing required field
    yaml_content = "optional_field: test"
    result = validate_yaml(yaml_content, RequiredSchema)
    assert len(result[1]) == 1
    assert result[1][0].code == "required"
    
    # Test YAML with null value
    yaml_content = "required_field: null"
    result = validate_yaml(yaml_content, RequiredSchema)
    assert len(result[1]) == 1
    assert result[1][0].code == "required"
    
    # Test YAML with boolean values
    class BoolSchema(Schema):
        active = Boolean()
    
    yaml_content = "active: true"
    result = validate_yaml(yaml_content, BoolSchema)
    assert result[0]["active"] is True
    assert result[1] == []
    
    # Test YAML with float values
    class FloatSchema(Schema):
        price = Float(minimum=0.0)
    
    yaml_content = "price: 19.99"
    result = validate_yaml(yaml_content, FloatSchema)
    assert result[0]["price"] == 19.99
    assert result[1] == []
    
    # Test YAML with list values
    class ListSchema(Schema):
        tags = Array(items=String())
    
    yaml_content = """
    tags:
      - python
      - testing
      - yaml
    """
    result = validate_yaml(yaml_content, ListSchema)
    assert result[0]["tags"] == ["python", "testing", "yaml"]
    assert result[1] == []


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_yaml():
    # Test basic validation with simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int, minimum=0)
    
    # Test valid YAML
    valid_content = "name: John\nage: 25"
    value, errors = validate_yaml(valid_content, SimpleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 25}
    
    # Test invalid YAML - validation error
    invalid_content = "name: John\nage: -5"
    value, errors = validate_yaml(invalid_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "minimum"
    assert errors[0].position.line_no == 2
    
    # Test invalid YAML - parse error
    parse_error_content = "name: John\n  age: 25"  # Invalid indentation
    try:
        validate_yaml(parse_error_content, SimpleSchema)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.line_no == 2
    
    # Test empty content
    empty_content = ""
    try:
        validate_yaml(empty_content, SimpleSchema)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
    
    # Test with bytes input
    bytes_content = b"name: Alice\nage: 30"
    value, errors = validate_yaml(bytes_content, SimpleSchema)
    assert errors == []
    assert value == {"name": "Alice", "age": 30}
    
    # Test with nested schema
    class NestedSchema(Schema):
        user = SimpleSchema
        active = Field(bool)
    
    nested_content = "user:\n  name: Bob\n  age: 40\nactive: true"
    value, errors = validate_yaml(nested_content, NestedSchema)
    assert errors == []
    assert value == {"user": {"name": "Bob", "age": 40}, "active": True}
    
    # Test with Field validator directly
    field_validator = Field(str, max_length=5)
    value, errors = validate_yaml("short", field_validator)
    assert errors == []
    assert value == "short"
    
    value, errors = validate_yaml("too_long", field_validator)
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test with list validation
    class ListSchema(Schema):
        items = Field(list)
    
    list_content = "items:\n  - apple\n  - banana\n  - cherry"
    value, errors = validate_yaml(list_content, ListSchema)
    assert errors == []
    assert value == {"items": ["apple", "banana", "cherry"]}


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}
    
    # Test with invalid YAML (parse error)
    invalid_content = "name: John\n  age: 30"  # Invalid indentation
    value, errors = validate_yaml(invalid_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert errors[0].position is not None
    
    # Test with valid YAML but validation failure
    content = "name: John\nage: 'thirty'"  # age is string instead of int
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type"
    assert "age" in errors[0].text
    
    # Test with empty content
    empty_content = ""
    value, errors = validate_yaml(empty_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test with bytes input
    bytes_content = b"name: Alice\nage: 25"
    value, errors = validate_yaml(bytes_content, SimpleSchema)
    assert errors == []
    assert value == {"name": "Alice", "age": 25}
    
    # Test with nested structure
    class NestedSchema(Schema):
        user = Field(dict, properties={"name": Field(str), "scores": Field(list)})
    
    content = "user:\n  name: Bob\n  scores: [1, 2, 3]"
    value, errors = validate_yaml(content, NestedSchema)
    assert errors == []
    assert value == {"user": {"name": "Bob", "scores": [1, 2, 3]}}
    
    # Test with Field validator directly
    field_validator = Field(dict, properties={"title": Field(str)})
    content = "title: Test"
    value, errors = validate_yaml(content, field_validator)
    assert errors == []
    assert value == {"title": "Test"}
    
    # Test with complex YAML types
    content = """
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    list:
      - item1
      - item2
    """
    
    class ComplexSchema(Schema):
        integer = Field(int)
        float = Field(float)
        boolean = Field(bool)
        null_value = Field(type=None, allow_null=True)
        list = Field(list)
    
    value, errors = validate_yaml(content, ComplexSchema)
    assert errors == []
    assert value["integer"] == 42
    assert value["float"] == 3.14
    assert value["boolean"] is True
    assert value["null_value"] is None
    assert value["list"] == ["item1", "item2"]


# LLM-generated content at query #14
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.start == 0
    assert token.end == 9
    assert token.value == {"key": "value"}
    
    # Test nested structures
    content = """
    name: John
    age: 30
    hobbies:
      - reading
      - swimming
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert "name" in token.value
    assert "hobbies" in token.value
    assert isinstance(token.value["hobbies"], list)
    
    # Test different data types
    content = """
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    string: hello
    """
    token = tokenize_yaml(content)
    assert token.value["integer"] == 42
    assert token.value["float"] == 3.14
    assert token.value["boolean"] is True
    assert token.value["null_value"] is None
    assert token.value["string"] == "hello"
    
    # Test list parsing
    content = "- item1\n- item2\n- item3"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2", "item3"]
    
    # Test scalar value
    content = "simple string"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "simple string"
    
    # Test bytes input
    content = b"key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test empty content
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
    
    # Test whitespace-only content
    try:
        tokenize_yaml("   \n  \t  \n")
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed list")
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "parse_error" in exc.code
    
    # Test malformed YAML
    try:
        tokenize_yaml("key: : value")
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test complex nested structure
    content = """
    users:
      - name: Alice
        age: 25
        active: true
      - name: Bob
        age: 30
        active: false
    settings:
      theme: dark
      notifications: true
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert len(token.value["users"]) == 2
    assert token.value["users"][0]["name"] == "Alice"
    assert token.value["settings"]["theme"] == "dark"
    
    # Test positions are correctly calculated
    content = "first:\n  second: value"
    token = tokenize_yaml(content)
    assert token.start == 0
    assert token.end == len(content) - 1
    
    # Test with special characters
    content = "message: 'Hello, \"world\"!'"
    token = tokenize_yaml(content)
    assert token.value["message"] == 'Hello, "world"!'
    
    # Test multiline string
    content = """
    description: |
      This is a
      multiline
      string
    """
    token = tokenize_yaml(content)
    assert "This is a\nmultiline\nstring\n" in token.value["description"]


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with string field
    content = "name: John"
    field = String(max_length=10)
    value, errors = validate_yaml(content, field)
    assert value == "John"
    assert errors == []
    
    # Test 2: Invalid YAML with string field (too long)
    content = "name: Johnathan"
    field = String(max_length=5)
    value, errors = validate_yaml(content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test 3: Valid YAML with schema
    class PersonSchema(Schema):
        name = String(max_length=20)
        age = Integer(minimum=0)
    
    content = """
    name: Alice
    age: 25
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert value == {"name": "Alice", "age": 25}
    assert errors == []
    
    # Test 4: Invalid YAML with schema (multiple errors)
    content = """
    name: Bob
    age: -5
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "minimum"
    
    # Test 5: Empty YAML content
    content = ""
    field = String()
    try:
        validate_yaml(content, field)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "no_content"
    
    # Test 6: Invalid YAML syntax
    content = "name: [unclosed list"
    field = String()
    try:
        validate_yaml(content, field)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "parse_error"
    
    # Test 7: YAML with nested structure
    class NestedSchema(Schema):
        items = ListToken
    
    content = """
    items:
      - first
      - second
    """
    value, errors = validate_yaml(content, NestedSchema)
    assert isinstance(value, dict)
    assert "items" in value
    assert len(value["items"]) == 2
    
    # Test 8: Bytes input
    content = b"name: Charlie"
    field = String()
    value, errors = validate_yaml(content, field)
    assert value == "Charlie"
    assert errors == []
    
    # Test 9: YAML with boolean values
    content = "active: true"
    field = String()
    value, errors = validate_yaml(content, field)
    assert value is True
    assert errors == []
    
    # Test 10: YAML with null value
    content = "value: null"
    field = String(allow_null=True)
    value, errors = validate_yaml(content, field)
    assert value is None
    assert errors == []


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_yaml():
    # Mock yaml to test ImportError handling
    import sys
    original_yaml = sys.modules.get('yaml')
    
    # Test 1: yaml not installed
    sys.modules['yaml'] = None
    try:
        from typesystem.tokenize.yaml import validate_yaml
        import pytest
        with pytest.raises(AssertionError) as exc_info:
            validate_yaml("test: value", Field(type="string"))
        assert "'pyyaml' must be installed." in str(exc_info.value)
    finally:
        if original_yaml:
            sys.modules['yaml'] = original_yaml
        else:
            del sys.modules['yaml']
    
    # Test 2: Valid YAML with simple field validation
    from typesystem.fields import String
    from typesystem.tokenize.yaml import validate_yaml
    
    content = "name: John"
    field = String(max_length=10)
    value, errors = validate_yaml(content, field)
    assert value == {"name": "John"}
    assert errors == []
    
    # Test 3: Invalid YAML - parse error
    invalid_content = "name: John\n  age: 30"  # Bad indentation
    field = String()
    value, errors = validate_yaml(invalid_content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    
    # Test 4: Valid YAML with schema validation
    from typesystem.schemas import Schema
    from typesystem.fields import Integer
    
    class PersonSchema(Schema):
        name = String(max_length=20)
        age = Integer(minimum=0)
    
    valid_person = "name: Alice\nage: 25"
    value, errors = validate_yaml(valid_person, PersonSchema)
    assert value == {"name": "Alice", "age": 25}
    assert errors == []
    
    # Test 5: Invalid YAML with schema validation
    invalid_person = "name: Alice\nage: -5"
    value, errors = validate_yaml(invalid_person, PersonSchema)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "minimum"
    
    # Test 6: Empty content
    empty_content = ""
    value, errors = validate_yaml(empty_content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test 7: Bytes input
    bytes_content = b"name: Bob"
    value, errors = validate_yaml(bytes_content, field)
    assert value == {"name": "Bob"}
    assert errors == []
    
    # Test 8: Complex nested structure validation
    from typesystem.fields import Array
    
    class NestedSchema(Schema):
        items = Array(items=String())
    
    nested_content = "items:\n  - first\n  - second"
    value, errors = validate_yaml(nested_content, NestedSchema)
    assert value == {"items": ["first", "second"]}
    assert errors == []
    
    # Test 9: Multiple validation errors
    class StrictSchema(Schema):
        name = String(required=True, max_length=5)
        age = Integer(required=True, minimum=18)
    
    invalid_strict = "name: Jonathan\nage: 16"
    value, errors = validate_yaml(invalid_strict, StrictSchema)
    assert value is None
    assert len(errors) == 2
    error_codes = [error.code for error in errors]
    assert "max_length" in error_codes
    assert "minimum" in error_codes
    
    # Test 10: Valid YAML with null values
    nullable_content = "name: null"
    nullable_field = String(allow_null=True)
    value, errors = validate_yaml(nullable_content, nullable_field)
    assert value == {"name": None}
    assert errors == []


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml():
    from typesystem.fields import String, Integer, Boolean
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0, maximum=150)
        active = Boolean()
    
    # Test valid YAML
    valid_yaml = """
    name: John
    age: 30
    active: true
    """
    
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30, "active": True}
    
    # Test invalid YAML - parse error
    invalid_yaml = """
    name: John
    age: thirty  # not a number
    active: true
    """
    
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"
    assert errors[0].text == "Must be a number."
    
    # Test validation error - name too long
    invalid_yaml2 = """
    name: JohnathanDoe
    age: 30
    active: true
    """
    
    value, errors = validate_yaml(invalid_yaml2, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    assert errors[0].text == "Must have no more than 10 characters."
    
    # Test multiple validation errors
    invalid_yaml3 = """
    name: JohnathanDoe
    age: 200
    active: maybe
    """
    
    value, errors = validate_yaml(invalid_yaml3, TestSchema)
    assert len(errors) == 3
    
    # Test empty YAML
    empty_yaml = ""
    
    try:
        validate_yaml(empty_yaml, TestSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "no_content"
    
    # Test YAML with bytes input
    bytes_yaml = b"""
    name: Alice
    age: 25
    active: false
    """
    
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert errors == []
    assert value == {"name": "Alice", "age": 25, "active": False}
    
    # Test with Field validator directly
    field_validator = String(max_length=5)
    yaml_content = "test_value"
    
    value, errors = validate_yaml(yaml_content, field_validator)
    assert errors == []
    assert value == "test_value"
    
    # Test invalid YAML syntax
    malformed_yaml = """
    name: John
    age: 30
      extra: indented
    """
    
    try:
        validate_yaml(malformed_yaml, TestSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "parse_error"
    
    # Test YAML with only whitespace
    whitespace_yaml = "   \n   \t   \n"
    
    try:
        validate_yaml(whitespace_yaml, TestSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "no_content"


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validation
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert result["name"] == "John"
    assert result["age"] == 30

    # Test with invalid YAML (parse error)
    invalid_yaml = "name: John\n  age: 30"  # Invalid indentation
    result, errors = validate_yaml(invalid_yaml, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert errors[0].position is not None

    # Test with validation error
    invalid_content = "name: John\nage: 'thirty'"  # age is string, should be int
    result, errors = validate_yaml(invalid_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type"
    assert "age" in errors[0].text

    # Test with empty content
    empty_content = ""
    result, errors = validate_yaml(empty_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test with bytes input
    bytes_content = b"name: Alice\nage: 25"
    result, errors = validate_yaml(bytes_content, SimpleSchema)
    assert errors == []
    assert result["name"] == "Alice"
    assert result["age"] == 25

    # Test with nested schema validation
    class NestedSchema(Schema):
        title = Field(str)
        metadata = Field(dict)

    nested_content = "title: Test\nmetadata:\n  author: Bob\n  version: 1.0"
    result, errors = validate_yaml(nested_content, NestedSchema)
    assert errors == []
    assert result["title"] == "Test"
    assert result["metadata"]["author"] == "Bob"

    # Test with required field missing
    result, errors = validate_yaml("name: John", SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test with complex YAML structure
    complex_content = """
    users:
      - name: Alice
        age: 25
      - name: Bob
        age: 30
    """
    
    class UserSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    class ComplexSchema(Schema):
        users = Field(list, items=UserSchema)
    
    result, errors = validate_yaml(complex_content, ComplexSchema)
    assert errors == []
    assert len(result["users"]) == 2
    assert result["users"][0]["name"] == "Alice"

    # Test with invalid nested data
    invalid_nested = """
    users:
      - name: Alice
        age: "twenty five"
    """
    result, errors = validate_yaml(invalid_nested, ComplexSchema)
    assert len(errors) == 1
    assert "age" in errors[0].text

    # Test with YAML containing null values
    null_content = "name: null\nage: 30"
    result, errors = validate_yaml(null_content, SimpleSchema)
    assert len(errors) == 1
    assert "name" in errors[0].text

    # Test with boolean values
    class BoolSchema(Schema):
        active = Field(bool)
    
    bool_content = "active: true"
    result, errors = validate_yaml(bool_content, BoolSchema)
    assert errors == []
    assert result["active"] is True

    # Test with float values
    class FloatSchema(Schema):
        price = Field(float)
    
    float_content = "price: 19.99"
    result, errors = validate_yaml(float_content, FloatSchema)
    assert errors == []
    assert result["price"] == 19.99


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer
    
    # Test valid YAML with string field
    string_field = String(max_length=10)
    result = validate_yaml("hello", string_field)
    assert result == "hello"
    
    # Test invalid YAML with string field (too long)
    result = validate_yaml("this_string_is_too_long", string_field)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].code == "max_length"
    
    # Test valid YAML with integer field
    int_field = Integer(minimum=0, maximum=100)
    result = validate_yaml("42", int_field)
    assert result == 42
    
    # Test invalid YAML with integer field (out of range)
    result = validate_yaml("150", int_field)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].code == "max_value"
    
    # Test with schema validation
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    
    class PersonSchema(Schema):
        name = String(max_length=50)
        age = Integer(minimum=0)
    
    # Test valid YAML with schema
    yaml_content = """
    name: John Doe
    age: 30
    """
    result = validate_yaml(yaml_content, PersonSchema)
    assert isinstance(result, dict)
    assert result["name"] == "John Doe"
    assert result["age"] == 30
    
    # Test invalid YAML with schema (missing required field)
    yaml_content = """
    name: John Doe
    """
    result = validate_yaml(yaml_content, PersonSchema)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].code == "required"
    
    # Test invalid YAML with schema (invalid field value)
    yaml_content = """
    name: John Doe
    age: -5
    """
    result = validate_yaml(yaml_content, PersonSchema)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].code == "min_value"
    
    # Test empty YAML content
    result = validate_yaml("", PersonSchema)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].code == "no_content"
    
    # Test YAML with parse error
    invalid_yaml = """
    name: John Doe
    age: [unclosed list
    """
    result = validate_yaml(invalid_yaml, PersonSchema)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].code == "parse_error"
    
    # Test bytes input
    bytes_content = b"name: Alice\nage: 25"
    result = validate_yaml(bytes_content, PersonSchema)
    assert isinstance(result, dict)
    assert result["name"] == "Alice"
    assert result["age"] == 25
    
    # Test complex nested structure
    class NestedSchema(Schema):
        items = String()
    
    yaml_content = """
    items:
      - first
      - second
      - third
    """
    result = validate_yaml(yaml_content, NestedSchema)
    assert isinstance(result, dict)
    assert result["items"] == ["first", "second", "third"]
    
    # Test with boolean values
    class BooleanSchema(Schema):
        active = Boolean()
    
    from typesystem.fields import Boolean
    yaml_content = "active: true"
    result = validate_yaml(yaml_content, BooleanSchema)
    assert result["active"] is True
    
    # Test with null values
    class NullableSchema(Schema):
        value = String(allow_null=True)
    
    yaml_content = "value: null"
    result = validate_yaml(yaml_content, NullableSchema)
    assert result["value"] is None
    
    # Test with float values
    class FloatSchema(Schema):
        price = Float()
    
    from typesystem.fields import Float
    yaml_content = "price: 19.99"
    result = validate_yaml(yaml_content, FloatSchema)
    assert result["price"] == 19.99


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] == {"name": "John", "age": 30}
    assert result[1] == []

    # Test with invalid YAML (parse error)
    invalid_yaml = "name: John\n  age: 30"  # Invalid indentation
    result = validate_yaml(invalid_yaml, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "parse_error"

    # Test with validation error
    yaml_content = "name: John\nage: 'thirty'"  # age is string, should be int
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert "age" in str(result[1][0])

    # Test with empty content
    result = validate_yaml("", SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "no_content"

    # Test with bytes input
    yaml_bytes = b"name: Alice\nage: 25"
    result = validate_yaml(yaml_bytes, SimpleSchema)
    assert result[0] == {"name": "Alice", "age": 25}
    assert result[1] == []

    # Test with nested schema
    class NestedSchema(Schema):
        user = SimpleSchema
        active = Field(bool)

    yaml_content = "user:\n  name: Bob\n  age: 40\nactive: true"
    result = validate_yaml(yaml_content, NestedSchema)
    assert result[0] == {"user": {"name": "Bob", "age": 40}, "active": True}
    assert result[1] == []

    # Test with list validation
    class ListSchema(Schema):
        items = Field(list)

    yaml_content = "items:\n  - apple\n  - banana\n  - cherry"
    result = validate_yaml(yaml_content, ListSchema)
    assert result[0] == {"items": ["apple", "banana", "cherry"]}
    assert result[1] == []

    # Test with required field missing
    yaml_content = "name: John"  # age is missing
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert "age" in str(result[1][0])

    # Test with Field validator directly
    field_validator = Field(str)
    yaml_content = "'hello world'"
    result = validate_yaml(yaml_content, field_validator)
    assert result[0] == "hello world"
    assert result[1] == []

    # Test with invalid type for Field validator
    field_validator = Field(int)
    yaml_content = "'not a number'"
    result = validate_yaml(yaml_content, field_validator)
    assert result[0] is None
    assert len(result[1]) == 1


# LLM-generated content at query #21
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.start == 0
    assert token.end == 9
    assert token.value == {"key": "value"}
    
    # Test nested structures
    content = """
    name: John
    age: 30
    hobbies:
      - reading
      - swimming
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["name"] == "John"
    assert token.value["age"] == 30
    assert isinstance(token.value["hobbies"], list)
    
    # Test different data types
    content = """
    string: hello
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    """
    token = tokenize_yaml(content)
    assert token.value["string"] == "hello"
    assert token.value["integer"] == 42
    assert token.value["float"] == 3.14
    assert token.value["boolean"] is True
    assert token.value["null_value"] is None
    
    # Test empty string
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test bytes input
    content_bytes = b"key: value"
    token = tokenize_yaml(content_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed list")
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test list token
    content = "- item1\n- item2"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    
    # Test scalar token
    content = "simple string"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "simple string"
    
    # Test complex nested structure
    content = """
    users:
      - name: Alice
        age: 25
      - name: Bob
        age: 30
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert len(token.value["users"]) == 2
    assert token.value["users"][0]["name"] == "Alice"
    
    # Test with special characters
    content = "message: 'Hello, World!'"
    token = tokenize_yaml(content)
    assert token.value["message"] == "Hello, World!"


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String
    
    field = String(max_length=5)
    
    # Valid case
    result = validate_yaml("name: abc", field)
    assert result == ("abc", [])
    
    # Invalid case - too long
    result = validate_yaml("name: abcdef", field)
    assert len(result[1]) == 1
    assert result[1][0].code == "max_length"
    
    # Test with schema validation
    from typesystem.schemas import Schema
    from typesystem.fields import Integer
    
    class PersonSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    # Valid schema case
    result = validate_yaml("name: John\nage: 25", PersonSchema)
    assert isinstance(result[0], dict)
    assert result[0]["name"] == "John"
    assert result[0]["age"] == 25
    assert result[1] == []
    
    # Invalid schema case - multiple errors
    result = validate_yaml("name: Johnathan\nage: -5", PersonSchema)
    assert len(result[1]) == 2
    error_codes = [error.code for error in result[1]]
    assert "max_length" in error_codes
    assert "minimum" in error_codes
    
    # Test empty content
    from typesystem.base import ParseError
    
    try:
        validate_yaml("", field)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.text == "No content."
    
    # Test with only whitespace
    try:
        validate_yaml("   \n  \t\n", field)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test invalid YAML syntax
    try:
        validate_yaml("name: [1, 2", field)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test bytes input
    result = validate_yaml(b"name: test", field)
    assert result == ("test", [])
    
    # Test nested structures
    from typesystem.fields import Array
    
    array_field = Array(items=String())
    result = validate_yaml("- item1\n- item2\n- item3", array_field)
    assert result[0] == ["item1", "item2", "item3"]
    assert result[1] == []
    
    # Test with complex nested schema
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class CompanySchema(Schema):
        name = String()
        address = AddressSchema
    
    result = validate_yaml(
        "name: Acme\naddress:\n  street: Main St\n  city: Metropolis",
        CompanySchema
    )
    assert result[0]["name"] == "Acme"
    assert result[0]["address"]["street"] == "Main St"
    assert result[0]["address"]["city"] == "Metropolis"
    assert result[1] == []
    
    # Test validation error positions
    result = validate_yaml("name: John\nage: not_a_number", PersonSchema)
    assert len(result[1]) == 1
    assert result[1][0].code == "type"
    assert hasattr(result[1][0], 'position')


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}
    
    # Test with invalid YAML (parse error)
    invalid_content = "name: John\n  age: 30"  # Invalid indentation
    result, errors = validate_yaml(invalid_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert errors[0].position is not None
    
    # Test with validation error
    invalid_data = "name: John\nage: 'thirty'"  # age is string instead of int
    result, errors = validate_yaml(invalid_data, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type"
    assert "age" in errors[0].text
    
    # Test with empty content
    empty_content = ""
    result, errors = validate_yaml(empty_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test with bytes input
    bytes_content = b"name: Alice\nage: 25"
    result, errors = validate_yaml(bytes_content, SimpleSchema)
    assert errors == []
    assert result == {"name": "Alice", "age": 25}
    
    # Test with nested structure
    class NestedSchema(Schema):
        user = Field(dict, properties={"name": Field(str), "scores": Field(list)})
    
    nested_content = "user:\n  name: Bob\n  scores: [1, 2, 3]"
    result, errors = validate_yaml(nested_content, NestedSchema)
    assert errors == []
    assert result == {"user": {"name": "Bob", "scores": [1, 2, 3]}}
    
    # Test with Field validator directly
    field_validator = Field(dict, properties={"id": Field(int), "active": Field(bool)})
    field_content = "id: 123\nactive: true"
    result, errors = validate_yaml(field_content, field_validator)
    assert errors == []
    assert result == {"id": 123, "active": True}
    
    # Test with invalid boolean value
    bool_content = "id: 123\nactive: 'yes'"
    result, errors = validate_yaml(bool_content, field_validator)
    assert len(errors) == 1
    assert "active" in errors[0].text
    
    # Test with list at root level
    class ListSchema(Schema):
        items = Field(list, items=Field(int))
    
    list_content = "items: [1, 2, 3]"
    result, errors = validate_yaml(list_content, ListSchema)
    assert errors == []
    assert result == {"items": [1, 2, 3]}
    
    # Test with null values
    class NullableSchema(Schema):
        value = Field(str, allow_null=True)
    
    null_content = "value: null"
    result, errors = validate_yaml(null_content, NullableSchema)
    assert errors == []
    assert result == {"value": None}


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with string field
    content = "name: John"
    field = String(max_length=10)
    value, errors = validate_yaml(content, field)
    assert value == "John"
    assert errors == []
    
    # Test 2: Invalid YAML with string field (too long)
    content = "name: Johnathan"
    field = String(max_length=5)
    value, errors = validate_yaml(content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test 3: Valid YAML with schema
    class PersonSchema(Schema):
        name = String(max_length=20)
        age = Integer(minimum=0)
    
    content = """
    name: Alice
    age: 25
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert value == {"name": "Alice", "age": 25}
    assert errors == []
    
    # Test 4: Invalid YAML with schema (multiple errors)
    content = """
    name: Bob
    age: -5
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "minimum"
    
    # Test 5: Invalid YAML syntax
    content = "name: [unclosed list"
    field = String()
    try:
        validate_yaml(content, field)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == "parse_error"
    
    # Test 6: Empty YAML content
    content = ""
    field = String()
    try:
        validate_yaml(content, field)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == "no_content"
    
    # Test 7: YAML with nested structure
    class NestedSchema(Schema):
        items = String()
    
    content = """
    items: |
      - item1
      - item2
    """
    value, errors = validate_yaml(content, NestedSchema)
    assert "item1\n- item2\n" in value["items"]
    assert errors == []
    
    # Test 8: Bytes input
    content = b"number: 42"
    field = Integer()
    value, errors = validate_yaml(content, field)
    assert value == 42
    assert errors == []
    
    # Test 9: YAML with boolean
    content = "active: true"
    field = String()
    value, errors = validate_yaml(content, field)
    assert value == "true"
    assert errors == []
    
    # Test 10: YAML with null
    content = "value: null"
    field = String(allow_null=True)
    value, errors = validate_yaml(content, field)
    assert value is None
    assert errors == []


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_yaml():
    # Test basic validation with simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}
    
    # Test validation error
    content = "name: John\nage: 'thirty'"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"
    
    # Test empty content
    content = ""
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test bytes input
    content = b"name: Alice\nage: 25"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "Alice", "age": 25}
    
    # Test nested validation
    class NestedSchema(Schema):
        user = Field(dict)
    
    class UserSchema(Schema):
        name = Field(str)
        email = Field(str)
    
    content = "user:\n  name: Bob\n  email: bob@example.com"
    value, errors = validate_yaml(content, NestedSchema)
    assert errors == []
    
    # Test with Field directly
    field = Field(str)
    content = "hello"
    value, errors = validate_yaml(content, field)
    assert errors == []
    assert value == "hello"
    
    # Test invalid YAML syntax
    content = "name: John\n  age: 30"  # Bad indentation
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    
    # Test list validation
    class ListSchema(Schema):
        items = Field(list)
    
    content = "items:\n  - apple\n  - banana\n  - cherry"
    value, errors = validate_yaml(content, ListSchema)
    assert errors == []
    assert value == {"items": ["apple", "banana", "cherry"]}
    
    # Test boolean validation
    class BoolSchema(Schema):
        active = Field(bool)
    
    content = "active: true"
    value, errors = validate_yaml(content, BoolSchema)
    assert errors == []
    assert value == {"active": True}
    
    # Test null/None handling
    class NullableSchema(Schema):
        value = Field(str, allow_null=True)
    
    content = "value: null"
    value, errors = validate_yaml(content, NullableSchema)
    assert errors == []
    assert value == {"value": None}


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int, minimum=0)

    content = "name: John\nage: 25"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 25}

    # Test with validation errors
    content = "name: John\nage: -5"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "minimum"
    assert errors[0].position.line_no == 2

    # Test with parse error (invalid YAML)
    content = "name: John\n  age: 25"  # Invalid indentation
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test with empty content
    content = ""
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    # Test with bytes input
    content = b"name: Alice\nage: 30"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "Alice", "age": 30}

    # Test with nested structures
    class NestedSchema(Schema):
        items = Field(list)
        config = Field(dict)

    content = "items:\n  - apple\n  - banana\nconfig:\n  enabled: true"
    value, errors = validate_yaml(content, NestedSchema)
    assert errors == []
    assert value == {"items": ["apple", "banana"], "config": {"enabled": True}}

    # Test with field validator directly
    from typesystem.fields import Integer

    int_field = Integer(minimum=10, maximum=100)
    content = "50"
    value, errors = validate_yaml(content, int_field)
    assert errors == []
    assert value == 50

    # Test with field validator error
    content = "5"
    value, errors = validate_yaml(content, int_field)
    assert len(errors) == 1
    assert errors[0].code == "minimum"

    # Test with complex YAML types
    content = "score: 95.5\nactive: true\nnickname: null"
    value, errors = validate_yaml(content, Schema)
    assert errors == []
    assert value == {"score": 95.5, "active": True, "nickname": None}


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_yaml():
    from typesystem.fields import String, Integer, Boolean
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0, maximum=150)
        active = Boolean()
    
    # Test valid YAML
    valid_content = """
    name: John
    age: 30
    active: true
    """
    value, errors = validate_yaml(valid_content, TestSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30, "active": True}
    
    # Test invalid YAML (parse error)
    invalid_yaml = """
    name: John
    age: 30
    active: true
      extra: indented
    """
    try:
        validate_yaml(invalid_yaml, TestSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code in ['parse_error', 'no_content']
    
    # Test validation error
    invalid_content = """
    name: JohnathanTooLong
    age: 200
    active: maybe
    """
    value, errors = validate_yaml(invalid_content, TestSchema)
    assert len(errors) == 3
    error_codes = [error.code for error in errors]
    assert "max_length" in error_codes
    assert "maximum" in error_codes
    assert "type" in error_codes
    
    # Test empty content
    try:
        validate_yaml("", TestSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert e.code == "no_content"
    
    # Test bytes input
    bytes_content = b"""
    name: Alice
    age: 25
    active: false
    """
    value, errors = validate_yaml(bytes_content, TestSchema)
    assert errors == []
    assert value == {"name": "Alice", "age": 25, "active": False}
    
    # Test with Field validator directly
    field_validator = String(max_length=5)
    content = "name: Bob"
    value, errors = validate_yaml(content, field_validator)
    assert errors == []
    assert value == {"name": "Bob"}
    
    # Test nested validation
    class NestedSchema(Schema):
        id = Integer()
        data = TestSchema
    
    nested_content = """
    id: 1
    data:
      name: Eve
      age: 28
      active: true
    """
    value, errors = validate_yaml(nested_content, NestedSchema)
    assert errors == []
    assert value["id"] == 1
    assert value["data"]["name"] == "Eve"


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with string field
    content = "name: John"
    field = String(max_length=10)
    value, errors = validate_yaml(content, field)
    assert value == "John"
    assert errors == []
    
    # Test 2: Invalid YAML with string field (too long)
    content = "name: Johnathan"
    field = String(max_length=5)
    value, errors = validate_yaml(content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test 3: Valid YAML with schema
    class PersonSchema(Schema):
        name = String(max_length=20)
        age = Integer(minimum=0)
    
    content = """
    name: Alice
    age: 25
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert value == {"name": "Alice", "age": 25}
    assert errors == []
    
    # Test 4: Invalid YAML with schema (multiple errors)
    content = """
    name: Bob
    age: -5
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "minimum"
    
    # Test 5: Invalid YAML parse error
    content = "name: [unclosed: list"
    field = String()
    try:
        validate_yaml(content, field)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == "parse_error"
    
    # Test 6: Empty content
    content = ""
    field = String()
    try:
        validate_yaml(content, field)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == "no_content"
    
    # Test 7: Bytes input
    content = b"name: Charlie"
    field = String()
    value, errors = validate_yaml(content, field)
    assert value == "Charlie"
    assert errors == []
    
    # Test 8: Nested structure validation
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class UserSchema(Schema):
        name = String()
        address = AddressSchema
    
    content = """
    name: David
    address:
      street: Main St
      city: Boston
    """
    value, errors = validate_yaml(content, UserSchema)
    assert value == {"name": "David", "address": {"street": "Main St", "city": "Boston"}}
    assert errors == []
    
    # Test 9: List validation
    from typesystem.fields import Array
    
    content = """
    - apple
    - banana
    - cherry
    """
    field = Array(items=String())
    value, errors = validate_yaml(content, field)
    assert value == ["apple", "banana", "cherry"]
    assert errors == []
    
    # Test 10: Complex nested validation with errors
    content = """
    users:
      - name: Eve
        age: 30
      - name: Frank
        age: -1
    """
    
    class UserItem(Schema):
        name = String()
        age = Integer(minimum=0)
    
    class UserList(Schema):
        users = Array(items=UserItem)
    
    value, errors = validate_yaml(content, UserList)
    assert value is None
    assert len(errors) == 1
    assert "age" in str(errors[0])


# LLM-generated content at query #29
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar tokenization
    result = tokenize_yaml("hello: world")
    assert isinstance(result, DictToken)
    assert result.start == 0
    assert result.end == 12
    assert "hello" in result.value
    assert isinstance(result.value["hello"], ScalarToken)
    assert result.value["hello"].value == "world"
    
    # Test integer tokenization
    result = tokenize_yaml("number: 42")
    assert isinstance(result.value["number"], ScalarToken)
    assert result.value["number"].value == 42
    
    # Test float tokenization
    result = tokenize_yaml("pi: 3.14")
    assert isinstance(result.value["pi"], ScalarToken)
    assert result.value["pi"].value == 3.14
    
    # Test boolean tokenization
    result = tokenize_yaml("flag: true")
    assert isinstance(result.value["flag"], ScalarToken)
    assert result.value["flag"].value is True
    
    # Test null tokenization
    result = tokenize_yaml("empty: null")
    assert isinstance(result.value["empty"], ScalarToken)
    assert result.value["empty"].value is None
    
    # Test list tokenization
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(item, ScalarToken) for item in result.value)
    assert result.value[0].value == "item1"
    assert result.value[1].value == "item2"
    
    # Test nested structure
    result = tokenize_yaml("""
    users:
      - name: Alice
        age: 30
      - name: Bob
        age: 25
    """)
    assert isinstance(result, DictToken)
    users = result.value["users"]
    assert isinstance(users, ListToken)
    assert len(users.value) == 2
    assert isinstance(users.value[0], DictToken)
    assert users.value[0].value["name"].value == "Alice"
    assert users.value[0].value["age"].value == 30
    
    # Test bytes input
    result = tokenize_yaml(b"key: value")
    assert isinstance(result, DictToken)
    assert result.value["key"].value == "value"
    
    # Test empty string
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test whitespace-only string
    try:
        tokenize_yaml("   \n  \t  \n")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed list")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "parse_error" in str(e)
    
    # Test malformed YAML
    try:
        tokenize_yaml("key: : value")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test correct position tracking
    content = "first: line\nsecond: value"
    result = tokenize_yaml(content)
    assert result.value["first"].start == 7
    assert result.value["first"].end == 11
    assert result.value["second"].start == 19
    assert result.value["second"].end == 24
    
    # Test complex nested structure with mixed types
    content = """
    config:
      enabled: true
      timeout: 30.5
      retries: 3
      servers:
        - host: "server1"
          port: 8080
        - host: "server2"
          port: 8081
    """
    result = tokenize_yaml(content)
    config = result.value["config"]
    assert config.value["enabled"].value is True
    assert config.value["timeout"].value == 30.5
    assert config.value["retries"].value == 3
    servers = config.value["servers"]
    assert len(servers.value) == 2
    assert servers.value[0].value["host"].value == "server1"
    assert servers.value[0].value["port"].value == 8080


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    valid_yaml = "name: John\nage: 30"
    result, errors = validate_yaml(valid_yaml, SimpleSchema)
    assert errors == []
    assert result["name"] == "John"
    assert result["age"] == 30
    
    # Test with invalid YAML (parse error)
    invalid_yaml = "name: John\nage: "
    result, errors = validate_yaml(invalid_yaml, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    
    # Test with validation error
    invalid_data = "name: 123\nage: thirty"
    result, errors = validate_yaml(invalid_data, SimpleSchema)
    assert len(errors) == 2
    assert any("name" in str(error) for error in errors)
    assert any("age" in str(error) for error in errors)
    
    # Test with empty content
    empty_yaml = ""
    result, errors = validate_yaml(empty_yaml, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test with bytes input
    bytes_yaml = b"name: Alice\nage: 25"
    result, errors = validate_yaml(bytes_yaml, SimpleSchema)
    assert errors == []
    assert result["name"] == "Alice"
    assert result["age"] == 25
    
    # Test with nested schema
    class NestedSchema(Schema):
        user = SimpleSchema
        active = Field(bool)
    
    nested_yaml = "user:\n  name: Bob\n  age: 40\nactive: true"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert result["user"]["name"] == "Bob"
    assert result["user"]["age"] == 40
    assert result["active"] is True
    
    # Test with Field validator directly
    int_field = Field(int)
    result, errors = validate_yaml("42", int_field)
    assert errors == []
    assert result == 42
    
    # Test with invalid type using Field validator
    result, errors = validate_yaml("not_a_number", int_field)
    assert len(errors) == 1
    
    # Test with list validation
    class ListSchema(Schema):
        numbers = Field(list, items=Field(int))
    
    list_yaml = "numbers:\n  - 1\n  - 2\n  - 3"
    result, errors = validate_yaml(list_yaml, ListSchema)
    assert errors == []
    assert result["numbers"] == [1, 2, 3]
    
    # Test with malformed YAML structure
    malformed_yaml = "name: John\n: invalid"
    result, errors = validate_yaml(malformed_yaml, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    
    # Test with YAML containing null value
    null_yaml = "name: null\nage: 25"
    result, errors = validate_yaml(null_yaml, SimpleSchema)
    assert len(errors) == 1  # name should be string, not null
    
    # Test with boolean values
    bool_yaml = "flag: true\nactive: false"
    class BoolSchema(Schema):
        flag = Field(bool)
        active = Field(bool)
    
    result, errors = validate_yaml(bool_yaml, BoolSchema)
    assert errors == []
    assert result["flag"] is True
    assert result["active"] is False
    
    # Test with float values
    float_yaml = "price: 19.99\ntax: 0.07"
    class FloatSchema(Schema):
        price = Field(float)
        tax = Field(float)
    
    result, errors = validate_yaml(float_yaml, FloatSchema)
    assert errors == []
    assert result["price"] == 19.99
    assert result["tax"] == 0.07


# LLM-generated content at query #31
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9
    assert token.content == content
    
    # Test nested structures
    content = """
    name: John
    age: 30
    hobbies:
      - reading
      - swimming
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["name"] == "John"
    assert token.value["age"] == 30
    assert isinstance(token.value["hobbies"], list)
    assert token.value["hobbies"][0] == "reading"
    
    # Test different data types
    content = """
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    string: hello
    """
    token = tokenize_yaml(content)
    assert token.value["integer"] == 42
    assert token.value["float"] == 3.14
    assert token.value["boolean"] is True
    assert token.value["null_value"] is None
    assert token.value["string"] == "hello"
    
    # Test bytes input
    content_bytes = b"key: value"
    token = tokenize_yaml(content_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test empty content
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
    
    # Test whitespace-only content
    try:
        tokenize_yaml("   \n  \t  \n")
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed list")
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "parse_error" in exc.text.lower()
    
    # Test list token
    content = "- item1\n- item2"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    
    # Test scalar token
    content = "simple string"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "simple string"
    
    # Test complex nested structure
    content = """
    users:
      - name: Alice
        age: 25
        active: true
      - name: Bob
        age: 30
        active: false
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    users = token.value["users"]
    assert isinstance(users, list)
    assert users[0]["name"] == "Alice"
    assert users[0]["age"] == 25
    assert users[0]["active"] is True
    
    # Test positions are correctly calculated
    content = "first:\n  second: value"
    token = tokenize_yaml(content)
    assert token.start == 0
    assert token.end == len(content) - 1


# LLM-generated content at query #32
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer
    
    # Test valid YAML with string field
    string_field = String(max_length=10)
    result = validate_yaml("hello: world", string_field)
    assert result == ("world", [])
    
    # Test invalid YAML with string field (too long)
    result = validate_yaml("hello: thisiswaytoolong", string_field)
    assert len(result[1]) == 1
    assert result[1][0].code == "max_length"
    
    # Test valid YAML with integer field
    int_field = Integer(minimum=0, maximum=100)
    result = validate_yaml("value: 42", int_field)
    assert result == (42, [])
    
    # Test invalid YAML with integer field (out of range)
    result = validate_yaml("value: 150", int_field)
    assert len(result[1]) == 1
    assert result[1][0].code == "maximum"
    
    # Test with schema validation
    from typesystem.schemas import Schema
    
    class PersonSchema(Schema):
        name = String(max_length=50)
        age = Integer(minimum=0)
    
    # Test valid YAML with schema
    yaml_content = """
    name: John Doe
    age: 30
    """
    result = validate_yaml(yaml_content, PersonSchema)
    assert result[0]["name"] == "John Doe"
    assert result[0]["age"] == 30
    assert result[1] == []
    
    # Test invalid YAML with schema (missing required field)
    yaml_content = """
    name: John Doe
    """
    result = validate_yaml(yaml_content, PersonSchema)
    assert len(result[1]) == 1
    assert result[1][0].code == "required"
    
    # Test invalid YAML with schema (wrong type)
    yaml_content = """
    name: John Doe
    age: "thirty"
    """
    result = validate_yaml(yaml_content, PersonSchema)
    assert len(result[1]) == 1
    assert result[1][0].code == "type"
    
    # Test empty YAML content
    try:
        validate_yaml("", PersonSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "no_content"
    
    # Test YAML with parse error
    invalid_yaml = """
    name: John Doe
    age: 30
      extra: indented wrong
    """
    try:
        validate_yaml(invalid_yaml, PersonSchema)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "parse_error"
    
    # Test bytes input
    bytes_content = b"name: Jane Doe\nage: 25"
    result = validate_yaml(bytes_content, PersonSchema)
    assert result[0]["name"] == "Jane Doe"
    assert result[0]["age"] == 25
    assert result[1] == []
    
    # Test nested structure validation
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class UserSchema(Schema):
        name = String()
        address = AddressSchema
    
    yaml_content = """
    name: Alice
    address:
      street: 123 Main St
      city: Springfield
    """
    result = validate_yaml(yaml_content, UserSchema)
    assert result[0]["name"] == "Alice"
    assert result[0]["address"]["street"] == "123 Main St"
    assert result[0]["address"]["city"] == "Springfield"
    assert result[1] == []
    
    # Test with list validation
    from typesystem.fields import Array
    
    list_field = Array(items=String())
    result = validate_yaml("- item1\n- item2\n- item3", list_field)
    assert result[0] == ["item1", "item2", "item3"]
    assert result[1] == []
    
    # Test boolean and null values
    class MixedSchema(Schema):
        active = Field(type="boolean")
        value = Field(type="number", allow_null=True)
    
    yaml_content = """
    active: true
    value: null
    """
    result = validate_yaml(yaml_content, MixedSchema)
    assert result[0]["active"] is True
    assert result[0]["value"] is None
    assert result[1] == []


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_yaml():
    # Mock yaml module to test without pyyaml dependency
    import sys
    from unittest.mock import Mock, patch, MagicMock
    
    # Test 1: yaml not installed
    with patch.dict(sys.modules, {'yaml': None}):
        from typesystem.tokenize.yaml import validate_yaml
        
        try:
            validate_yaml("test", Mock())
            assert False, "Should have raised assertion error"
        except AssertionError as e:
            assert str(e) == "'pyyaml' must be installed."
    
    # Test 2: Valid YAML with simple schema validation
    mock_yaml = Mock()
    mock_yaml.load = Mock(return_value="parsed_token")
    mock_validator = Mock()
    mock_validator.validate = Mock(return_value="validated_value")
    
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml.tokenize_yaml') as mock_tokenize:
            mock_tokenize.return_value = Mock()
            
            with patch('typesystem.tokenize.yaml.validate_with_positions') as mock_validate:
                mock_validate.return_value = ("validated_value", [])
                
                result = validate_yaml("key: value", mock_validator)
                
                mock_tokenize.assert_called_once_with("key: value")
                mock_validate.assert_called_once_with(
                    token=mock_tokenize.return_value,
                    validator=mock_validator
                )
                assert result == ("validated_value", [])
    
    # Test 3: YAML parse error during tokenization
    mock_yaml = Mock()
    mock_yaml.load = Mock(side_effect=mock_yaml.scanner.ScannerError("test error"))
    mock_yaml.scanner = Mock()
    mock_yaml.scanner.ScannerError = Exception
    mock_yaml.parser = Mock()
    mock_yaml.parser.ParserError = Exception
    
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml._get_position') as mock_get_pos:
            mock_get_pos.return_value = Mock()
            
            from typesystem.tokenize.yaml import validate_yaml
            
            try:
                validate_yaml("invalid: [yaml", Mock())
                assert False, "Should have raised ParseError"
            except Exception as e:
                assert isinstance(e, Exception)
    
    # Test 4: Empty content
    mock_yaml = Mock()
    
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml.tokenize_yaml') as mock_tokenize:
            mock_tokenize.side_effect = Exception("No content.")
            
            from typesystem.tokenize.yaml import validate_yaml
            
            try:
                validate_yaml("", Mock())
                assert False, "Should have raised exception"
            except Exception as e:
                assert str(e) == "No content."
    
    # Test 5: Bytes input
    mock_yaml = Mock()
    mock_yaml.load = Mock(return_value="parsed_token")
    
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml.tokenize_yaml') as mock_tokenize:
            mock_tokenize.return_value = Mock()
            
            with patch('typesystem.tokenize.yaml.validate_with_positions') as mock_validate:
                mock_validate.return_value = ("validated_value", ["error"])
                
                result = validate_yaml(b"key: value", Mock())
                
                mock_tokenize.assert_called_once_with(b"key: value")
                assert result == ("validated_value", ["error"])
    
    # Test 6: Validator is Schema instance
    mock_yaml = Mock()
    mock_yaml.load = Mock(return_value="parsed_token")
    
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml.tokenize_yaml') as mock_tokenize:
            mock_tokenize.return_value = Mock()
            
            with patch('typesystem.tokenize.yaml.validate_with_positions') as mock_validate:
                mock_validate.return_value = ({"field": "value"}, [])
                
                mock_schema = Mock(spec=Schema)
                result = validate_yaml("field: value", mock_schema)
                
                mock_validate.assert_called_once_with(
                    token=mock_tokenize.return_value,
                    validator=mock_schema
                )
                assert result == ({"field": "value"}, [])
    
    # Test 7: Validator is Field instance
    mock_yaml = Mock()
    mock_yaml.load = Mock(return_value="parsed_token")
    
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml.tokenize_yaml') as mock_tokenize:
            mock_tokenize.return_value = Mock()
            
            with patch('typesystem.tokenize.yaml.validate_with_positions') as mock_validate:
                mock_validate.return_value = ("field_value", [])
                
                mock_field = Mock(spec=Field)
                result = validate_yaml("field_value", mock_field)
                
                mock_validate.assert_called_once_with(
                    token=mock_tokenize.return_value,
                    validator=mock_field
                )
                assert result == ("field_value", [])


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar types
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert "key" in result.value
    assert isinstance(result.value["key"], ScalarToken)
    assert result.value["key"].value == "value"
    
    # Test integer
    result = tokenize_yaml("number: 42")
    assert isinstance(result.value["number"], ScalarToken)
    assert result.value["number"].value == 42
    
    # Test float
    result = tokenize_yaml("pi: 3.14")
    assert isinstance(result.value["pi"], ScalarToken)
    assert result.value["pi"].value == 3.14
    
    # Test boolean
    result = tokenize_yaml("flag: true")
    assert isinstance(result.value["flag"], ScalarToken)
    assert result.value["flag"].value is True
    
    # Test null
    result = tokenize_yaml("nothing: null")
    assert isinstance(result.value["nothing"], ScalarToken)
    assert result.value["nothing"].value is None
    
    # Test list
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(item, ScalarToken) for item in result.value)
    assert result.value[0].value == "item1"
    assert result.value[1].value == "item2"
    
    # Test nested structure
    result = tokenize_yaml("""
    users:
      - name: Alice
        age: 30
      - name: Bob
        age: 25
    """)
    assert isinstance(result, DictToken)
    users = result.value["users"]
    assert isinstance(users, ListToken)
    assert len(users.value) == 2
    assert all(isinstance(user, DictToken) for user in users.value)
    
    # Test bytes input
    result = tokenize_yaml(b"key: value")
    assert isinstance(result, DictToken)
    assert result.value["key"].value == "value"
    
    # Test empty content
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."
    
    # Test whitespace only content
    try:
        tokenize_yaml("   \n  \t  \n")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed list")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test positions are correctly set
    content = "first: value\nsecond: item"
    result = tokenize_yaml(content)
    first_token = result.value["first"]
    assert first_token.start == 0
    assert first_token.end == 10
    assert first_token.content == content
    
    # Test complex scalar with special characters
    result = tokenize_yaml("special: value with spaces & symbols!")
    assert result.value["special"].value == "value with spaces & symbols!"


# LLM-generated content at query #3
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9
    
    # Test nested structures
    content = """
    name: John
    age: 30
    hobbies:
      - reading
      - hiking
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["name"] == "John"
    assert token.value["age"] == 30
    assert len(token.value["hobbies"]) == 2
    
    # Test list parsing
    content = "- item1\n- item2\n- item3"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert token.value[0] == "item1"
    
    # Test scalar types
    content = """
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    string: hello
    """
    token = tokenize_yaml(content)
    assert token.value["integer"] == 42
    assert token.value["float"] == 3.14
    assert token.value["boolean"] is True
    assert token.value["null_value"] is None
    assert token.value["string"] == "hello"
    
    # Test empty content
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."
    
    # Test bytes input
    content_bytes = b"key: value"
    token = tokenize_yaml(content_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test YAML parse error
    invalid_yaml = "key: [unclosed list"
    try:
        tokenize_yaml(invalid_yaml)
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test complex nested structure
    content = """
    users:
      - name: Alice
        age: 25
        active: true
      - name: Bob
        age: 30
        active: false
    metadata:
      count: 2
      timestamp: 2023-01-01
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert len(token.value["users"]) == 2
    assert token.value["users"][0]["name"] == "Alice"
    assert token.value["metadata"]["count"] == 2


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer
    
    # Test valid YAML with string field
    string_field = String(max_length=10)
    result = validate_yaml("hello", string_field)
    assert result == "hello"
    
    # Test invalid YAML with string field (too long)
    try:
        validate_yaml("this_is_too_long", string_field)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert "Must have no more than 10 characters" in str(exc)
    
    # Test valid YAML with integer field
    int_field = Integer(minimum=0, maximum=100)
    result = validate_yaml("42", int_field)
    assert result == 42
    
    # Test invalid YAML with integer field (out of range)
    try:
        validate_yaml("150", int_field)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert "Must be less than or equal to 100" in str(exc)
    
    # Test with schema validation
    from typesystem.schemas import Schema
    
    class PersonSchema(Schema):
        name = String(max_length=50)
        age = Integer(minimum=0)
    
    # Test valid YAML with schema
    valid_yaml = "name: John\nage: 30"
    result = validate_yaml(valid_yaml, PersonSchema)
    assert result == {"name": "John", "age": 30}
    
    # Test invalid YAML with schema (missing required field)
    invalid_yaml = "name: John"
    try:
        validate_yaml(invalid_yaml, PersonSchema)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert "This field is required" in str(exc)
    
    # Test with nested structures
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class UserSchema(Schema):
        name = String()
        address = AddressSchema
    
    nested_yaml = """
    name: Alice
    address:
      street: 123 Main St
      city: Boston
    """
    result = validate_yaml(nested_yaml, UserSchema)
    assert result["name"] == "Alice"
    assert result["address"]["street"] == "123 Main St"
    assert result["address"]["city"] == "Boston"
    
    # Test with bytes input
    bytes_content = b"name: Bob\nage: 25"
    result = validate_yaml(bytes_content, PersonSchema)
    assert result == {"name": "Bob", "age": 25}
    
    # Test empty YAML content
    try:
        validate_yaml("", PersonSchema)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test YAML parse error
    invalid_yaml_syntax = "name: [unclosed: list"
    try:
        validate_yaml(invalid_yaml_syntax, PersonSchema)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test with list validation
    from typesystem.fields import Array
    
    list_field = Array(items=String())
    result = validate_yaml("- item1\n- item2\n- item3", list_field)
    assert result == ["item1", "item2", "item3"]
    
    # Test with boolean and null values
    class MixedSchema(Schema):
        active = Field(type="boolean")
        value = Field(type="number", allow_null=True)
    
    mixed_yaml = "active: true\nvalue: null"
    result = validate_yaml(mixed_yaml, MixedSchema)
    assert result["active"] is True
    assert result["value"] is None
    
    # Test with float values
    float_yaml = "price: 19.99"
    float_field = Field(type="number")
    result = validate_yaml(float_yaml, float_field)
    assert result == {"price": 19.99}


# LLM-generated content at query #5
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.start == 0
    assert token.end == 9
    assert token.value == {"key": "value"}
    
    # Test nested structures
    content = """
    name: John
    age: 30
    hobbies:
      - reading
      - hiking
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert "name" in token.value
    assert "age" in token.value
    assert "hobbies" in token.value
    assert isinstance(token.value["hobbies"], ListToken)
    
    # Test different data types
    content = """
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    string: hello
    """
    token = tokenize_yaml(content)
    assert isinstance(token.value["integer"], ScalarToken)
    assert token.value["integer"].value == 42
    assert isinstance(token.value["float"], ScalarToken)
    assert token.value["float"].value == 3.14
    assert isinstance(token.value["boolean"], ScalarToken)
    assert token.value["boolean"].value is True
    assert isinstance(token.value["null_value"], ScalarToken)
    assert token.value["null_value"].value is None
    assert isinstance(token.value["string"], ScalarToken)
    assert token.value["string"].value == "hello"
    
    # Test empty content
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
    
    # Test bytes input
    content_bytes = b"key: value"
    token = tokenize_yaml(content_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed list")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test list at root level
    content = "- item1\n- item2"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0].value == "item1"
    assert token.value[1].value == "item2"
    
    # Test complex nested structure
    content = """
    users:
      - name: Alice
        age: 25
        active: true
      - name: Bob
        age: 30
        active: false
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    users = token.value["users"]
    assert isinstance(users, ListToken)
    assert len(users.value) == 2
    assert users.value[0].value["name"].value == "Alice"
    assert users.value[1].value["name"].value == "Bob"
    
    # Test scalar only content
    content = "just a string"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "just a string"
    
    # Test number only content
    content = "42"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    
    # Test boolean only content
    content = "true"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    # Test null only content
    content = "null"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar tokenization
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 4
    
    # Test integer tokenization
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert isinstance(result.value, int)
    
    # Test float tokenization
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert isinstance(result.value, float)
    
    # Test boolean tokenization
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_yaml("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test null tokenization
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test list tokenization
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert result.value[0].value == "item1"
    assert result.value[1].value == "item2"
    
    # Test dict tokenization
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert "key" in result.value
    assert result.value["key"].value == "value"
    
    # Test nested structure
    result = tokenize_yaml("items:\n  - name: test\n    value: 42")
    assert isinstance(result, DictToken)
    items = result.value["items"]
    assert isinstance(items, ListToken)
    assert len(items.value) == 1
    assert isinstance(items.value[0], DictToken)
    
    # Test bytes input
    result = tokenize_yaml(b"test: value")
    assert isinstance(result, DictToken)
    assert result.value["test"].value == "value"
    
    # Test empty string
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0
    
    # Test whitespace only
    try:
        tokenize_yaml("   \n  \t  \n")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "parse_error" in e.code
    
    # Test YAML with special characters
    result = tokenize_yaml("multiline: |\n  line1\n  line2")
    assert isinstance(result, DictToken)
    assert result.value["multiline"].value == "line1\nline2\n"
    
    # Test YAML with anchors and aliases
    result = tokenize_yaml("base: &base\n  key: value\ncopy: *base")
    assert isinstance(result, DictToken)
    assert result.value["base"]["key"].value == "value"
    assert result.value["copy"]["key"].value == "value"
    
    # Test position tracking in tokens
    content = "first:\n  second: value"
    result = tokenize_yaml(content)
    assert result.start == 0
    assert result.end == len(content) - 1
    assert result.value["first"].start == content.find("first")
    assert result.value["first"].end == len(content) - 1


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    yaml_content = "name: John\nage: 30"
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] == {"name": "John", "age": 30}
    assert result[1] == []
    
    # Test with invalid YAML (parse error)
    invalid_yaml = "name: John\n  age: 30"  # Invalid indentation
    result = validate_yaml(invalid_yaml, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "parse_error"
    
    # Test with validation error
    yaml_content = "name: John\nage: 'thirty'"  # age is string, should be int
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "type"
    
    # Test with empty content
    result = validate_yaml("", SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "no_content"
    
    # Test with bytes input
    yaml_bytes = b"name: Alice\nage: 25"
    result = validate_yaml(yaml_bytes, SimpleSchema)
    assert result[0] == {"name": "Alice", "age": 25}
    assert result[1] == []
    
    # Test with nested schema
    class NestedSchema(Schema):
        user = SimpleSchema
        active = Field(bool)
    
    yaml_content = "user:\n  name: Bob\n  age: 40\nactive: true"
    result = validate_yaml(yaml_content, NestedSchema)
    assert result[0] == {"user": {"name": "Bob", "age": 40}, "active": True}
    assert result[1] == []
    
    # Test with list validation
    class ListSchema(Schema):
        items = Field(list)
    
    yaml_content = "items:\n  - apple\n  - banana\n  - cherry"
    result = validate_yaml(yaml_content, ListSchema)
    assert result[0] == {"items": ["apple", "banana", "cherry"]}
    assert result[1] == []
    
    # Test with Field validator directly
    field_validator = Field(str)
    yaml_content = "Hello World"
    result = validate_yaml(yaml_content, field_validator)
    assert result[0] == "Hello World"
    assert result[1] == []
    
    # Test with invalid type for field validator
    yaml_content = "123"
    result = validate_yaml(yaml_content, field_validator)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "type"


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer
    
    # Test valid YAML with string field
    string_field = String(max_length=10)
    result = validate_yaml("hello: world", string_field)
    assert result == ("world", [])
    
    # Test invalid YAML with string field (too long)
    result = validate_yaml("hello: thisiswaytoolong", string_field)
    assert len(result[1]) == 1
    assert result[1][0].code == "max_length"
    
    # Test valid YAML with integer field
    int_field = Integer(minimum=0, maximum=100)
    result = validate_yaml("value: 42", int_field)
    assert result == (42, [])
    
    # Test invalid YAML with integer field (out of range)
    result = validate_yaml("value: 150", int_field)
    assert len(result[1]) == 1
    assert result[1][0].code == "max_value"
    
    # Test with Schema validation
    from typesystem.schemas import Schema
    
    class PersonSchema(Schema):
        name = String(max_length=50)
        age = Integer(minimum=0)
    
    # Test valid YAML with schema
    valid_yaml = """
    name: John Doe
    age: 30
    """
    result = validate_yaml(valid_yaml, PersonSchema)
    assert isinstance(result[0], dict)
    assert result[0]["name"] == "John Doe"
    assert result[0]["age"] == 30
    assert result[1] == []
    
    # Test invalid YAML with schema (missing required field)
    invalid_yaml = """
    name: John Doe
    """
    result = validate_yaml(invalid_yaml, PersonSchema)
    assert len(result[1]) == 1
    assert result[1][0].code == "required"
    
    # Test invalid YAML with schema (invalid age)
    invalid_yaml = """
    name: John Doe
    age: -5
    """
    result = validate_yaml(invalid_yaml, PersonSchema)
    assert len(result[1]) == 1
    assert result[1][0].code == "min_value"
    
    # Test with nested structures
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class UserSchema(Schema):
        name = String()
        address = AddressSchema
    
    valid_nested_yaml = """
    name: Alice
    address:
      street: 123 Main St
      city: Springfield
    """
    result = validate_yaml(valid_nested_yaml, UserSchema)
    assert result[1] == []
    assert result[0]["address"]["city"] == "Springfield"
    
    # Test with bytes input
    result = validate_yaml(b"value: test", string_field)
    assert result == ("test", [])
    
    # Test empty YAML content
    from typesystem.base import ParseError
    import pytest
    
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("", string_field)
    assert exc_info.value.code == "no_content"
    
    # Test invalid YAML syntax
    with pytest.raises(ParseError) as exc_info:
        validate_yaml("invalid: [unclosed list", string_field)
    assert exc_info.value.code == "parse_error"
    
    # Test YAML with multiple values (should validate first value)
    result = validate_yaml("""
    first: value1
    second: value2
    """, string_field)
    assert result == ("value1", [])
    
    # Test with boolean field
    from typesystem.fields import Boolean
    bool_field = Boolean()
    
    result = validate_yaml("flag: true", bool_field)
    assert result == (True, [])
    
    result = validate_yaml("flag: false", bool_field)
    assert result == (False, [])
    
    # Test with float field
    from typesystem.fields import Float
    float_field = Float()
    
    result = validate_yaml("value: 3.14", float_field)
    assert result == (3.14, [])
    
    # Test with null value
    result = validate_yaml("value: null", string_field)
    assert result == (None, [])
    
    # Test with list validation
    from typesystem.fields import Array
    array_field = Array(items=String())
    
    result = validate_yaml("items: [a, b, c]", array_field)
    assert result == (["a", "b", "c"], [])
    
    # Test with complex nested validation
    class ComplexSchema(Schema):
        id = Integer()
        tags = Array(items=String())
        metadata = Field(properties={"version": String()})
    
    complex_yaml = """
    id: 1
    tags: [tag1, tag2]
    metadata:
      version: "1.0"
    """
    result = validate_yaml(complex_yaml, ComplexSchema)
    assert result[1] == []
    assert result[0]["tags"] == ["tag1", "tag2"]


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer, Boolean
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with string field
    content = "name: John"
    field = String(max_length=10)
    value, errors = validate_yaml(content, field)
    assert value == "John"
    assert errors == []
    
    # Test 2: Invalid YAML with string field (too long)
    content = "name: Johnathan"
    field = String(max_length=5)
    value, errors = validate_yaml(content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test 3: Valid YAML with schema
    class PersonSchema(Schema):
        name = String(max_length=20)
        age = Integer(minimum=0)
    
    content = """
    name: Alice
    age: 25
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert value == {"name": "Alice", "age": 25}
    assert errors == []
    
    # Test 4: Invalid YAML with schema (multiple errors)
    content = """
    name: Bob
    age: -5
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "minimum"
    
    # Test 5: Invalid YAML parse error
    content = "name: [unclosed: list"
    field = String()
    try:
        validate_yaml(content, field)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "parse_error"
    
    # Test 6: Empty YAML content
    content = ""
    field = String()
    try:
        validate_yaml(content, field)
        assert False, "Should have raised ParseError"
    except Exception as e:
        assert isinstance(e, ParseError)
        assert e.code == "no_content"
    
    # Test 7: Valid YAML with nested structure
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class UserSchema(Schema):
        name = String()
        address = AddressSchema
    
    content = """
    name: Charlie
    address:
      street: Main St
      city: Metropolis
    """
    value, errors = validate_yaml(content, UserSchema)
    assert value == {
        "name": "Charlie",
        "address": {"street": "Main St", "city": "Metropolis"}
    }
    assert errors == []
    
    # Test 8: Invalid YAML with nested structure
    content = """
    name: David
    address:
      street: ""
      city: ""
    """
    value, errors = validate_yaml(content, UserSchema)
    assert value is None
    assert len(errors) == 2
    
    # Test 9: Valid YAML with boolean field
    content = "active: true"
    field = Boolean()
    value, errors = validate_yaml(content, field)
    assert value is True
    assert errors == []
    
    # Test 10: Valid YAML with integer field
    content = "count: 42"
    field = Integer()
    value, errors = validate_yaml(content, field)
    assert value == 42
    assert errors == []
    
    # Test 11: Bytes input
    content = b"name: Eve"
    field = String()
    value, errors = validate_yaml(content, field)
    assert value == "Eve"
    assert errors == []
    
    # Test 12: YAML with list validation
    from typesystem.fields import Array
    
    content = """
    - apple
    - banana
    - cherry
    """
    field = Array(items=String())
    value, errors = validate_yaml(content, field)
    assert value == ["apple", "banana", "cherry"]
    assert errors == []
    
    # Test 13: Complex YAML with mixed types
    content = """
    users:
      - name: Frank
        age: 30
        active: true
      - name: Grace
        age: 25
        active: false
    """
    
    class UserItem(Schema):
        name = String()
        age = Integer(minimum=0)
        active = Boolean()
    
    class UserList(Schema):
        users = Array(items=UserItem)
    
    value, errors = validate_yaml(content, UserList)
    assert len(value["users"]) == 2
    assert errors == []


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_yaml():
    # Mock yaml module to test ImportError handling
    import sys
    original_yaml = sys.modules.get('yaml')
    
    # Test 1: yaml not installed
    sys.modules['yaml'] = None
    try:
        from typesystem.tokenize.yaml_tokenizer import validate_yaml
        # Should raise AssertionError when yaml is None
        try:
            validate_yaml("test: value", Field(type="string"))
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert str(e) == "'pyyaml' must be installed."
    finally:
        if original_yaml:
            sys.modules['yaml'] = original_yaml
        else:
            del sys.modules['yaml']
    
    # Test 2: Valid YAML with simple field validation
    from typesystem.fields import String
    from typesystem.tokenize.yaml_tokenizer import validate_yaml
    
    content = "name: John"
    field = String(max_length=10)
    
    value, errors = validate_yaml(content, field)
    assert value == {"name": "John"}
    assert errors == []
    
    # Test 3: Invalid YAML (parse error)
    invalid_content = "name: John\n  age: 30"  # Bad indentation
    value, errors = validate_yaml(invalid_content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert errors[0].text.startswith("mapping values are not allowed here")
    
    # Test 4: Empty content
    empty_content = ""
    value, errors = validate_yaml(empty_content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    assert errors[0].text == "No content."
    
    # Test 5: Field validation failure
    content = "name: Johnathan"  # 10 characters max
    field = String(max_length=10)
    
    value, errors = validate_yaml(content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test 6: Schema validation
    from typesystem import Schema, fields
    
    class PersonSchema(Schema):
        name = fields.String(max_length=10)
        age = fields.Integer(minimum=0)
    
    content = "name: John\nage: 25"
    value, errors = validate_yaml(content, PersonSchema)
    assert value == {"name": "John", "age": 25}
    assert errors == []
    
    # Test 7: Schema validation failure
    content = "name: Johnathan\nage: -5"
    value, errors = validate_yaml(content, PersonSchema)
    assert value is None
    assert len(errors) == 2
    error_codes = [error.code for error in errors]
    assert "max_length" in error_codes
    assert "minimum" in error_codes
    
    # Test 8: Bytes input
    content_bytes = b"name: Alice"
    value, errors = validate_yaml(content_bytes, field)
    assert value == {"name": "Alice"}
    assert errors == []
    
    # Test 9: Complex nested structure
    class NestedSchema(Schema):
        users = fields.Array(fields.String())
    
    content = "users:\n  - Alice\n  - Bob"
    value, errors = validate_yaml(content, NestedSchema)
    assert value == {"users": ["Alice", "Bob"]}
    assert errors == []
    
    # Test 10: Invalid nested structure
    content = "users:\n  - Alice\n  - 123"  # 123 is not a string
    value, errors = validate_yaml(content, NestedSchema)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "type"


# LLM-generated content at query #11
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar tokenization
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 4
    
    # Test integer tokenization
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert isinstance(result.value, int)
    
    # Test float tokenization
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert isinstance(result.value, float)
    
    # Test boolean tokenization
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_yaml("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test null tokenization
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test list tokenization
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert result.value[0].value == "item1"
    assert result.value[1].value == "item2"
    
    # Test dict tokenization
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert "key" in result.value
    assert result.value["key"].value == "value"
    
    # Test nested structure
    result = tokenize_yaml("list:\n  - item\n  - 42")
    assert isinstance(result, DictToken)
    assert "list" in result.value
    list_token = result.value["list"]
    assert isinstance(list_token, ListToken)
    assert len(list_token.value) == 2
    assert list_token.value[0].value == "item"
    assert list_token.value[1].value == 42
    
    # Test bytes input
    result = tokenize_yaml(b"test")
    assert isinstance(result, ScalarToken)
    assert result.value == "test"
    
    # Test empty string
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0
    
    # Test whitespace only
    try:
        tokenize_yaml("   \n  ")
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "parse_error" in exc.code
    
    # Test malformed YAML
    try:
        tokenize_yaml("key: value\n  indented: wrong")
    except ParseError as exc:
        assert exc.code == "parse_error"
    
    # Test complex structure with all types
    yaml_content = """
string: hello
integer: 42
float: 3.14
boolean: true
null_value: null
list:
  - item1
  - 2
dict:
  nested: value
"""
    result = tokenize_yaml(yaml_content)
    assert isinstance(result, DictToken)
    assert result.value["string"].value == "hello"
    assert result.value["integer"].value == 42
    assert result.value["float"].value == 3.14
    assert result.value["boolean"].value is True
    assert result.value["null_value"].value is None
    assert isinstance(result.value["list"], ListToken)
    assert isinstance(result.value["dict"], DictToken)


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_yaml():
    # Mock yaml module to test without pyyaml dependency
    import sys
    from unittest.mock import Mock, patch, MagicMock
    
    # Test 1: yaml not installed
    with patch.dict(sys.modules, {'yaml': None}):
        from typesystem.tokenize.yaml_tokenizer import validate_yaml
        
        try:
            validate_yaml("test: value", Mock())
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "'pyyaml' must be installed." in str(e)
    
    # Test 2: Valid YAML with simple schema
    yaml_mock = Mock()
    yaml_mock.load = Mock(return_value=Mock(
        start=0,
        end=10,
        value={"name": "John", "age": 30},
        content="name: John\nage: 30"
    ))
    
    with patch.dict(sys.modules, {'yaml': yaml_mock}):
        from typesystem.tokenize.yaml_tokenizer import validate_yaml
        
        validator_mock = Mock()
        validator_mock.validate = Mock(return_value=({"name": "John", "age": 30}, []))
        
        result = validate_yaml("name: John\nage: 30", validator_mock)
        assert result == ({"name": "John", "age": 30}, [])
    
    # Test 3: Invalid YAML parsing
    yaml_mock = Mock()
    scanner_error = Mock()
    scanner_error.problem = "Invalid YAML"
    scanner_error.problem_mark = Mock(index=5)
    scanner_error.__class__.__name__ = "ScannerError"
    yaml_mock.load = Mock(side_effect=scanner_error)
    yaml_mock.ScannerError = type(scanner_error)
    yaml_mock.ParserError = Mock()
    
    with patch.dict(sys.modules, {'yaml': yaml_mock}):
        from typesystem.tokenize.yaml_tokenizer import validate_yaml, ParseError
        
        validator_mock = Mock()
        
        try:
            validate_yaml("invalid: [yaml", validator_mock)
            assert False, "Should have raised ParseError"
        except ParseError as e:
            assert e.text == "Invalid YAML."
            assert e.code == "parse_error"
    
    # Test 4: Empty content
    yaml_mock = Mock()
    
    with patch.dict(sys.modules, {'yaml': yaml_mock}):
        from typesystem.tokenize.yaml_tokenizer import validate_yaml, ParseError
        
        validator_mock = Mock()
        
        try:
            validate_yaml("", validator_mock)
            assert False, "Should have raised ParseError"
        except ParseError as e:
            assert e.text == "No content."
            assert e.code == "no_content"
    
    # Test 5: Bytes input
    yaml_mock = Mock()
    yaml_mock.load = Mock(return_value=Mock(
        start=0,
        end=8,
        value={"test": "value"},
        content="test: value"
    ))
    
    with patch.dict(sys.modules, {'yaml': yaml_mock}):
        from typesystem.tokenize.yaml_tokenizer import validate_yaml
        
        validator_mock = Mock()
        validator_mock.validate = Mock(return_value=({"test": "value"}, []))
        
        result = validate_yaml(b"test: value", validator_mock)
        assert result == ({"test": "value"}, [])
    
    # Test 6: Validation with errors
    yaml_mock = Mock()
    yaml_mock.load = Mock(return_value=Mock(
        start=0,
        end=10,
        value={"age": "not_a_number"},
        content="age: not_a_number"
    ))
    
    with patch.dict(sys.modules, {'yaml': yaml_mock}):
        from typesystem.tokenize.yaml_tokenizer import validate_yaml
        
        validator_mock = Mock()
        error_mock = Mock()
        error_mock.text = "Must be a number"
        error_mock.code = "invalid_type"
        error_mock.position = Mock(line_no=1, column_no=1, char_index=0)
        validator_mock.validate = Mock(return_value=({"age": "not_a_number"}, [error_mock]))
        
        result = validate_yaml("age: not_a_number", validator_mock)
        assert result == ({"age": "not_a_number"}, [error_mock])


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_yaml():
    # Test 1: Valid YAML with simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}
    
    # Test 2: Invalid YAML - type mismatch
    content = "name: John\nage: 'thirty'"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"
    assert errors[0].position is not None
    
    # Test 3: Invalid YAML - missing required field
    content = "name: John"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"
    
    # Test 4: Empty YAML content
    content = ""
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test 5: Bytes input
    content = b"name: Alice\nage: 25"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "Alice", "age": 25}
    
    # Test 6: Nested schema validation
    class AddressSchema(Schema):
        street = Field(str)
        city = Field(str)
    
    class PersonSchema(Schema):
        name = Field(str)
        address = Field(AddressSchema)
    
    content = """
    name: Bob
    address:
      street: Main St
      city: Springfield
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert errors == []
    assert value["name"] == "Bob"
    assert value["address"]["city"] == "Springfield"
    
    # Test 7: Invalid nested structure
    content = """
    name: Bob
    address:
      street: Main St
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"
    
    # Test 8: Malformed YAML syntax
    content = "name: John\nage: : 30"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert errors[0].position is not None
    
    # Test 9: Field validator instead of Schema
    from typesystem.fields import String, Integer
    
    class SimpleField(Field):
        def __init__(self):
            super().__init__(fields={"name": String(), "age": Integer()})
    
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, SimpleField())
    assert errors == []
    assert value == {"name": "John", "age": 30}
    
    # Test 10: List validation
    class ListSchema(Schema):
        items = Field(list)
    
    content = "items:\n  - apple\n  - banana\n  - cherry"
    value, errors = validate_yaml(content, ListSchema)
    assert errors == []
    assert value == {"items": ["apple", "banana", "cherry"]}
    
    # Test 11: Whitespace-only content
    content = "   \n  \t\n"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test 12: Complex YAML with multiple data types
    class ComplexSchema(Schema):
        name = Field(str)
        score = Field(float)
        active = Field(bool)
        tags = Field(list)
    
    content = """
    name: Test
    score: 95.5
    active: true
    tags:
      - python
      - testing
      - yaml
    """
    value, errors = validate_yaml(content, ComplexSchema)
    assert errors == []
    assert value["name"] == "Test"
    assert value["score"] == 95.5
    assert value["active"] is True
    assert "python" in value["tags"]


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with string field
    content = "name: John Doe"
    field = String(max_length=100)
    value, errors = validate_yaml(content, field)
    assert value == "John Doe"
    assert errors == []
    
    # Test 2: Invalid YAML with string field (too long)
    content = "name: " + "A" * 101
    field = String(max_length=100)
    value, errors = validate_yaml(content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test 3: Valid YAML with schema
    class PersonSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0)
    
    content = """
    name: Alice
    age: 30
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert value == {"name": "Alice", "age": 30}
    assert errors == []
    
    # Test 4: Invalid YAML with schema (multiple errors)
    content = """
    name: 
    age: -5
    """
    value, errors = validate_yaml(content, PersonSchema)
    assert value is None
    assert len(errors) == 2
    error_codes = {error.code for error in errors}
    assert "required" in error_codes
    assert "minimum" in error_codes
    
    # Test 5: Invalid YAML syntax
    content = "name: [unclosed: list"
    value, errors = validate_yaml(content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    
    # Test 6: Empty YAML content
    content = ""
    value, errors = validate_yaml(content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test 7: YAML with bytes input
    content = b"name: Bob"
    value, errors = validate_yaml(content, field)
    assert value == "Bob"
    assert errors == []
    
    # Test 8: Nested structure validation
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class CompanySchema(Schema):
        name = String()
        address = AddressSchema
    
    content = """
    name: Tech Corp
    address:
      street: 123 Main St
      city: Metropolis
    """
    value, errors = validate_yaml(content, CompanySchema)
    assert value == {
        "name": "Tech Corp",
        "address": {"street": "123 Main St", "city": "Metropolis"}
    }
    assert errors == []
    
    # Test 9: Invalid nested structure
    content = """
    name: Tech Corp
    address:
      street: 123 Main St
    """
    value, errors = validate_yaml(content, CompanySchema)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "required"
    
    # Test 10: YAML with list validation
    from typesystem.fields import Array
    
    field = Array(items=Integer())
    content = """
    - 1
    - 2
    - 3
    """
    value, errors = validate_yaml(content, field)
    assert value == [1, 2, 3]
    assert errors == []
    
    # Test 11: YAML with invalid list items
    content = """
    - 1
    - "two"
    - 3
    """
    value, errors = validate_yaml(content, field)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "type"


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_yaml():
    # Test 1: Valid YAML with simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}
    
    # Test 2: Invalid YAML - missing required field
    content = "name: John"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"
    
    # Test 3: Invalid YAML - wrong type
    content = "name: John\nage: 'thirty'"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type"
    
    # Test 4: Valid YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict, properties={"name": Field(str), "details": Field(dict)})
    
    content = "user:\n  name: Alice\n  details: {}\n"
    value, errors = validate_yaml(content, NestedSchema)
    assert errors == []
    assert value == {"user": {"name": "Alice", "details": {}}}
    
    # Test 5: Invalid YAML - parse error
    content = "name: John\n  age: 30"  # Invalid indentation
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    
    # Test 6: Empty YAML content
    content = ""
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test 7: Valid YAML with bytes input
    content = b"name: Bob\nage: 25"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "Bob", "age": 25}
    
    # Test 8: Valid YAML with Field validator directly
    validator = Field(dict, properties={"id": Field(int), "active": Field(bool)})
    content = "id: 123\nactive: true"
    value, errors = validate_yaml(content, validator)
    assert errors == []
    assert value == {"id": 123, "active": True}
    
    # Test 9: Invalid YAML with list validation
    class ListSchema(Schema):
        items = Field(list, items=Field(int))
    
    content = "items:\n  - 1\n  - 2\n  - 'three'"
    value, errors = validate_yaml(content, ListSchema)
    assert len(errors) == 1
    assert errors[0].code == "type"
    
    # Test 10: Valid YAML with complex nested structure
    class ComplexSchema(Schema):
        users = Field(list, items=Field(dict, properties={
            "id": Field(int),
            "name": Field(str),
            "tags": Field(list, items=Field(str))
        }))
    
    content = """
    users:
      - id: 1
        name: Alice
        tags: [admin, user]
      - id: 2
        name: Bob
        tags: [user]
    """
    value, errors = validate_yaml(content, ComplexSchema)
    assert errors == []
    assert len(value["users"]) == 2
    assert value["users"][0]["name"] == "Alice"


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_yaml():
    # Mock yaml module to test ImportError handling
    import sys
    from unittest.mock import patch, MagicMock
    
    # Test 1: yaml not installed
    with patch.dict(sys.modules, {'yaml': None}):
        from typesystem.tokenize.yaml_tokenizer import validate_yaml
        
        try:
            validate_yaml("test: value", MagicMock())
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert str(e) == "'pyyaml' must be installed."
    
    # Test 2: Valid YAML with simple schema validation
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    yaml_content = "name: John\nage: 25"
    value, errors = validate_yaml(yaml_content, TestSchema)
    
    assert errors == []
    assert value == {"name": "John", "age": 25}
    
    # Test 3: Invalid YAML - parse error
    invalid_yaml = "name: John\n  age: 25"  # Invalid indentation
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert errors[0].text.startswith("could not find expected")
    
    # Test 4: Empty YAML content
    value, errors = validate_yaml("", TestSchema)
    
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    assert errors[0].text == "No content."
    
    # Test 5: Validation error with positions
    yaml_content = "name: Johnathan\nage: -5"
    value, errors = validate_yaml(yaml_content, TestSchema)
    
    assert len(errors) == 2
    assert any(e.code == "max_length" for e in errors)
    assert any(e.code == "minimum" for e in errors)
    
    # Test 6: Valid YAML with nested structure
    class NestedSchema(Schema):
        user = TestSchema
        active = Boolean()
    
    from typesystem.fields import Boolean
    
    yaml_content = """
    user:
      name: Alice
      age: 30
    active: true
    """
    
    value, errors = validate_yaml(yaml_content, NestedSchema)
    
    assert errors == []
    assert value == {"user": {"name": "Alice", "age": 30}, "active": True}
    
    # Test 7: Bytes input
    bytes_content = b"name: Bob\nage: 40"
    value, errors = validate_yaml(bytes_content, TestSchema)
    
    assert errors == []
    assert value == {"name": "Bob", "age": 40}
    
    # Test 8: Complex YAML with lists
    class ListSchema(Schema):
        items = List(String())
        counts = List(Integer())
    
    yaml_content = """
    items:
      - apple
      - banana
      - cherry
    counts:
      - 1
      - 2
      - 3
    """
    
    value, errors = validate_yaml(yaml_content, ListSchema)
    
    assert errors == []
    assert value == {
        "items": ["apple", "banana", "cherry"],
        "counts": [1, 2, 3]
    }
    
    # Test 9: Field validator instead of Schema
    from typesystem.fields import Object
    
    field_validator = Object(properties={
        "title": String(),
        "count": Integer()
    })
    
    yaml_content = "title: Test\ncount: 42"
    value, errors = validate_yaml(yaml_content, field_validator)
    
    assert errors == []
    assert value == {"title": "Test", "count": 42}
    
    # Test 10: YAML with null values
    yaml_content = "name: null\nage: 30"
    value, errors = validate_yaml(yaml_content, TestSchema)
    
    assert len(errors) == 1
    assert errors[0].code == "required"


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml():
    # Test with simple field validation
    from typesystem.fields import String, Integer, Boolean
    from typesystem.schemas import Schema
    
    # Test 1: Valid YAML with string field
    content = "name: John"
    field = String(max_length=10)
    result, errors = validate_yaml(content, field)
    assert result == "John"
    assert errors == []
    
    # Test 2: Invalid YAML with string field (too long)
    content = "name: Johnathan"
    field = String(max_length=5)
    result, errors = validate_yaml(content, field)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test 3: Valid YAML with schema
    class PersonSchema(Schema):
        name = String(max_length=20)
        age = Integer(minimum=0)
    
    content = """
    name: Alice
    age: 25
    """
    result, errors = validate_yaml(content, PersonSchema)
    assert result == {"name": "Alice", "age": 25}
    assert errors == []
    
    # Test 4: Invalid YAML with schema (multiple errors)
    content = """
    name: Bob
    age: -5
    """
    result, errors = validate_yaml(content, PersonSchema)
    assert result is None
    assert len(errors) == 1
    assert errors[0].code == "minimum"
    
    # Test 5: Invalid YAML parse error
    content = "name: [unclosed list"
    field = String()
    try:
        validate_yaml(content, field)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None
    
    # Test 6: Empty content
    content = ""
    field = String()
    try:
        validate_yaml(content, field)
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test 7: Bytes input
    content = b"enabled: true"
    field = Boolean()
    result, errors = validate_yaml(content, field)
    assert result is True
    assert errors == []
    
    # Test 8: Nested structure validation
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class UserSchema(Schema):
        name = String()
        address = AddressSchema
    
    content = """
    name: Charlie
    address:
      street: Main St
      city: Metropolis
    """
    result, errors = validate_yaml(content, UserSchema)
    assert result == {
        "name": "Charlie",
        "address": {"street": "Main St", "city": "Metropolis"}
    }
    assert errors == []
    
    # Test 9: Invalid nested structure
    content = """
    name: Dana
    address:
      street: ""
      city: ""
    """
    result, errors = validate_yaml(content, UserSchema)
    assert result is None
    assert len(errors) == 2
    
    # Test 10: List validation
    from typesystem.fields import Array
    
    content = "items:\n  - apple\n  - banana\n  - cherry"
    field = Array(items=String())
    result, errors = validate_yaml(content, field)
    assert result == ["apple", "banana", "cherry"]
    assert errors == []


# LLM-generated content at query #18
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start == 0
    assert result.end == 9
    
    # Test nested structures
    result = tokenize_yaml("list:\n  - item1\n  - item2")
    assert isinstance(result, DictToken)
    assert isinstance(result.value["list"], ListToken)
    assert result.value["list"].value == ["item1", "item2"]
    
    # Test different data types
    yaml_content = """
string: hello
integer: 42
float: 3.14
boolean: true
null_value: null
"""
    result = tokenize_yaml(yaml_content)
    assert isinstance(result.value["string"], ScalarToken)
    assert result.value["string"].value == "hello"
    assert isinstance(result.value["integer"], ScalarToken)
    assert result.value["integer"].value == 42
    assert isinstance(result.value["float"], ScalarToken)
    assert result.value["float"].value == 3.14
    assert isinstance(result.value["boolean"], ScalarToken)
    assert result.value["boolean"].value is True
    assert isinstance(result.value["null_value"], ScalarToken)
    assert result.value["null_value"].value is None
    
    # Test bytes input
    result = tokenize_yaml(b"key: value")
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty content
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0
    
    # Test whitespace-only content
    try:
        tokenize_yaml("   \n  \t  \n")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed list")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "could not find expected" in e.text.lower()
    
    # Test list token
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert result.value == ["item1", "item2"]
    
    # Test scalar token directly
    result = tokenize_yaml("simple scalar")
    assert isinstance(result, ScalarToken)
    assert result.value == "simple scalar"
    
    # Test complex nested structure
    yaml_content = """
parent:
  child:
    grandchild: value
  list:
    - nested: item
"""
    result = tokenize_yaml(yaml_content)
    assert isinstance(result, DictToken)
    parent = result.value["parent"]
    assert isinstance(parent, DictToken)
    assert isinstance(parent.value["child"], DictToken)
    assert parent.value["child"].value["grandchild"] == "value"
    assert isinstance(parent.value["list"], ListToken)
    assert isinstance(parent.value["list"].value[0], DictToken)
    
    # Test position information
    result = tokenize_yaml("key: value")
    assert result.start == 0
    assert result.end == 9
    assert result.content == "key: value"


# LLM-generated content at query #19
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar types
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert result.start == 0
    assert result.end == 9
    assert "key" in result.value
    assert isinstance(result.value["key"], ScalarToken)
    assert result.value["key"].value == "value"
    
    # Test integer
    result = tokenize_yaml("number: 42")
    assert isinstance(result.value["number"], ScalarToken)
    assert result.value["number"].value == 42
    
    # Test float
    result = tokenize_yaml("pi: 3.14")
    assert isinstance(result.value["pi"], ScalarToken)
    assert result.value["pi"].value == 3.14
    
    # Test boolean
    result = tokenize_yaml("flag: true")
    assert isinstance(result.value["flag"], ScalarToken)
    assert result.value["flag"].value is True
    
    # Test null
    result = tokenize_yaml("empty: null")
    assert isinstance(result.value["empty"], ScalarToken)
    assert result.value["empty"].value is None
    
    # Test list
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(item, ScalarToken) for item in result.value)
    assert result.value[0].value == "item1"
    assert result.value[1].value == "item2"
    
    # Test nested structure
    result = tokenize_yaml("list:\n  - a\n  - b")
    assert isinstance(result, DictToken)
    assert isinstance(result.value["list"], ListToken)
    assert len(result.value["list"].value) == 2
    
    # Test bytes input
    result = tokenize_yaml(b"key: value")
    assert isinstance(result, DictToken)
    assert result.value["key"].value == "value"
    
    # Test empty string
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0
    
    # Test whitespace only
    try:
        tokenize_yaml("   \n  \t  \n")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test complex structure with positions
    content = "outer:\n  inner:\n    key: value"
    result = tokenize_yaml(content)
    assert result.start == 0
    assert result.end == len(content) - 1
    assert isinstance(result.value["outer"], DictToken)
    assert result.value["outer"].start > result.start
    assert result.value["outer"].end < result.end


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_yaml():
    # Test 1: Valid YAML with simple schema validation
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    yaml_content = "name: John\nage: 30"
    value, errors = validate_yaml(yaml_content, SimpleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}
    
    # Test 2: Invalid YAML - type mismatch
    yaml_content = "name: John\nage: 'thirty'"
    value, errors = validate_yaml(yaml_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"
    assert errors[0].position is not None
    
    # Test 3: Invalid YAML - missing required field
    yaml_content = "name: John"
    value, errors = validate_yaml(yaml_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"
    
    # Test 4: Malformed YAML - parse error
    yaml_content = "name: John\n  age: 30"  # Invalid indentation
    value, errors = validate_yaml(yaml_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert errors[0].position is not None
    
    # Test 5: Empty YAML content
    yaml_content = ""
    value, errors = validate_yaml(yaml_content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test 6: Valid YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict, properties={"name": Field(str), "address": Field(dict)})
    
    yaml_content = """
    user:
      name: Alice
      address:
        city: London
        country: UK
    """
    value, errors = validate_yaml(yaml_content, NestedSchema)
    assert errors == []
    assert value["user"]["name"] == "Alice"
    
    # Test 7: Valid YAML with list validation
    class ListSchema(Schema):
        items = Field(list, items=Field(int))
    
    yaml_content = "items: [1, 2, 3]"
    value, errors = validate_yaml(yaml_content, ListSchema)
    assert errors == []
    assert value["items"] == [1, 2, 3]
    
    # Test 8: Invalid YAML with wrong list item type
    yaml_content = "items: [1, 'two', 3]"
    value, errors = validate_yaml(yaml_content, ListSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"
    
    # Test 9: Valid YAML with bytes input
    yaml_bytes = b"name: Bob\nage: 25"
    value, errors = validate_yaml(yaml_bytes, SimpleSchema)
    assert errors == []
    assert value == {"name": "Bob", "age": 25}
    
    # Test 10: Field validator instead of Schema
    field_validator = Field(dict, properties={"title": Field(str, max_length=10)})
    yaml_content = "title: Hello World"
    value, errors = validate_yaml(yaml_content, field_validator)
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test 11: Valid YAML with all YAML types
    class AllTypesSchema(Schema):
        string = Field(str)
        integer = Field(int)
        float_num = Field(float)
        boolean = Field(bool)
        null_value = Field(type=None, allow_null=True)
    
    yaml_content = """
    string: hello
    integer: 42
    float_num: 3.14
    boolean: true
    null_value: null
    """
    value, errors = validate_yaml(yaml_content, AllTypesSchema)
    assert errors == []
    assert value["string"] == "hello"
    assert value["integer"] == 42
    assert value["float_num"] == 3.14
    assert value["boolean"] is True
    assert value["null_value"] is None
    
    # Test 12: Complex nested validation with multiple errors
    class ComplexSchema(Schema):
        users = Field(list, items=Field(dict, properties={
            "id": Field(int, minimum=1),
            "email": Field(str, format="email")
        }))
    
    yaml_content = """
    users:
      - id: 0
        email: not-an-email
      - id: 2
        email: valid@example.com
      - id: -1
        email: another@bad
    """
    value, errors = validate_yaml(yaml_content, ComplexSchema)
    assert len(errors) == 3  # id too small (2 users) + invalid email (2 users)


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_yaml():
    # Test 1: Valid YAML with simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}
    
    # Test 2: Invalid YAML - wrong data type
    content = "name: John\nage: thirty"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"
    assert errors[0].position is not None
    
    # Test 3: Invalid YAML - missing required field
    content = "name: John"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"
    
    # Test 4: Invalid YAML - parse error
    content = "name: John\nage: [30"
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert errors[0].position is not None
    
    # Test 5: Empty YAML content
    content = ""
    value, errors = validate_yaml(content, SimpleSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test 6: Valid YAML with nested structure
    class NestedSchema(Schema):
        user = Field(dict)
    
    class UserSchema(Schema):
        name = Field(str)
        address = Field(dict)
    
    class AddressSchema(Schema):
        street = Field(str)
        city = Field(str)
    
    content = """
    user:
      name: Alice
      address:
        street: Main St
        city: Boston
    """
    value, errors = validate_yaml(content, NestedSchema)
    assert errors == []
    
    # Test 7: Valid YAML with list field
    class ListSchema(Schema):
        numbers = Field(list)
    
    content = "numbers: [1, 2, 3]"
    value, errors = validate_yaml(content, ListSchema)
    assert errors == []
    assert value == {"numbers": [1, 2, 3]}
    
    # Test 8: Valid YAML with boolean field
    class BooleanSchema(Schema):
        active = Field(bool)
    
    content = "active: true"
    value, errors = validate_yaml(content, BooleanSchema)
    assert errors == []
    assert value == {"active": True}
    
    # Test 9: Valid YAML with null value
    class NullableSchema(Schema):
        optional = Field(str, allow_null=True)
    
    content = "optional: null"
    value, errors = validate_yaml(content, NullableSchema)
    assert errors == []
    assert value == {"optional": None}
    
    # Test 10: Valid YAML with bytes input
    content = b"name: Bob\nage: 25"
    value, errors = validate_yaml(content, SimpleSchema)
    assert errors == []
    assert value == {"name": "Bob", "age": 25}
    
    # Test 11: Complex validation with multiple errors
    class ComplexSchema(Schema):
        name = Field(str, max_length=10)
        age = Field(int, minimum=0, maximum=150)
        email = Field(str, format="email")
    
    content = """
    name: VeryLongNameThatExceedsLimit
    age: -5
    email: not-an-email
    """
    value, errors = validate_yaml(content, ComplexSchema)
    assert len(errors) == 3
    
    # Test 12: Valid YAML with Field validator directly
    field_validator = Field(dict, properties={
        "title": Field(str),
        "count": Field(int)
    })
    
    content = "title: Test\ncount: 42"
    value, errors = validate_yaml(content, field_validator)
    assert errors == []
    assert value == {"title": "Test", "count": 42}


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_yaml():
    # Mock yaml to simulate it being installed
    import sys
    from unittest.mock import Mock, patch, MagicMock
    
    # Test 1: yaml is not installed
    with patch.dict(sys.modules, {'yaml': None}):
        from typesystem.tokenize.yaml_tokenizer import validate_yaml
        try:
            validate_yaml("test: value", Mock())
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "'pyyaml' must be installed." in str(e)
    
    # Restore yaml for other tests
    import yaml
    
    # Test 2: Empty content
    with patch('typesystem.tokenize.yaml_tokenizer.tokenize_yaml') as mock_tokenize:
        mock_tokenize.side_effect = ParseError(
            text="No content.", 
            code="no_content", 
            position=Position(column_no=1, line_no=1, char_index=0)
        )
        
        validator = Mock()
        result = validate_yaml("", validator)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is None
        assert len(result[1]) == 1
        assert result[1][0].text == "No content."
        assert result[1][0].code == "no_content"
    
    # Test 3: Valid YAML with successful validation
    with patch('typesystem.tokenize.yaml_tokenizer.tokenize_yaml') as mock_tokenize:
        mock_token = Mock()
        mock_tokenize.return_value = mock_token
        
        with patch('typesystem.tokenize.yaml_tokenizer.validate_with_positions') as mock_validate:
            expected_value = {"test": "value"}
            mock_validate.return_value = (expected_value, [])
            
            validator = Mock()
            result = validate_yaml("test: value", validator)
            
            mock_tokenize.assert_called_once_with("test: value")
            mock_validate.assert_called_once_with(token=mock_token, validator=validator)
            assert result == (expected_value, [])
    
    # Test 4: Valid YAML with validation errors
    with patch('typesystem.tokenize.yaml_tokenizer.tokenize_yaml') as mock_tokenize:
        mock_token = Mock()
        mock_tokenize.return_value = mock_token
        
        with patch('typesystem.tokenize.yaml_tokenizer.validate_with_positions') as mock_validate:
            error_messages = [
                ParseError(text="Field required.", code="required", position=Position(1, 1, 0)),
                ParseError(text="Invalid type.", code="type", position=Position(2, 1, 10))
            ]
            mock_validate.return_value = (None, error_messages)
            
            validator = Mock()
            result = validate_yaml("invalid: yaml", validator)
            
            assert result == (None, error_messages)
    
    # Test 5: Invalid YAML syntax
    with patch('typesystem.tokenize.yaml_tokenizer.tokenize_yaml') as mock_tokenize:
        mock_tokenize.side_effect = ParseError(
            text="Scanner error.",
            code="parse_error",
            position=Position(column_no=1, line_no=1, char_index=0)
        )
        
        validator = Mock()
        result = validate_yaml("invalid: [yaml", validator)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is None
        assert len(result[1]) == 1
        assert result[1][0].text == "Scanner error."
        assert result[1][0].code == "parse_error"
    
    # Test 6: Bytes input
    with patch('typesystem.tokenize.yaml_tokenizer.tokenize_yaml') as mock_tokenize:
        mock_token = Mock()
        mock_tokenize.return_value = mock_token
        
        with patch('typesystem.tokenize.yaml_tokenizer.validate_with_positions') as mock_validate:
            mock_validate.return_value = ({"bytes": "test"}, [])
            
            validator = Mock()
            result = validate_yaml(b"bytes: test", validator)
            
            mock_tokenize.assert_called_once_with(b"bytes: test")
            assert result == ({"bytes": "test"}, [])
    
    # Test 7: Complex validator (Schema instance)
    with patch('typesystem.tokenize.yaml_tokenizer.tokenize_yaml') as mock_tokenize:
        mock_token = Mock()
        mock_tokenize.return_value = mock_token
        
        with patch('typesystem.tokenize.yaml_tokenizer.validate_with_positions') as mock_validate:
            class TestSchema(Schema):
                name = Field(str)
                age = Field(int)
            
            schema_instance = TestSchema()
            validated_data = {"name": "John", "age": 30}
            mock_validate.return_value = (validated_data, [])
            
            result = validate_yaml("name: John\nage: 30", schema_instance)
            
            mock_validate.assert_called_once_with(token=mock_token, validator=schema_instance)
            assert result == (validated_data, [])
    
    # Test 8: Complex validator (Field instance)
    with patch('typesystem.tokenize.yaml_tokenizer.tokenize_yaml') as mock_tokenize:
        mock_token = Mock()
        mock_tokenize.return_value = mock_token
        
        with patch('typesystem.tokenize.yaml_tokenizer.validate_with_positions') as mock_validate:
            field_validator = Field(str)
            validated_data = "string value"
            mock_validate.return_value = (validated_data, [])
            
            result = validate_yaml("string value", field_validator)
            
            mock_validate.assert_called_once_with(token=mock_token, validator=field_validator)
            assert result == (validated_data, [])


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validator
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    yaml_content = "name: John\nage: 30"
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] == {"name": "John", "age": 30}
    assert result[1] == []
    
    # Test with invalid YAML (parse error)
    invalid_yaml = "name: John\nage: :"
    result = validate_yaml(invalid_yaml, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "parse_error"
    
    # Test with validation error
    yaml_content = "name: John\nage: 'thirty'"
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "type"
    
    # Test with empty content
    result = validate_yaml("", SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "no_content"
    
    # Test with bytes input
    yaml_bytes = b"name: Alice\nage: 25"
    result = validate_yaml(yaml_bytes, SimpleSchema)
    assert result[0] == {"name": "Alice", "age": 25}
    assert result[1] == []
    
    # Test with nested schema
    class NestedSchema(Schema):
        user = SimpleSchema
        active = Field(bool)
    
    yaml_content = "user:\n  name: Bob\n  age: 40\nactive: true"
    result = validate_yaml(yaml_content, NestedSchema)
    assert result[0] == {"user": {"name": "Bob", "age": 40}, "active": True}
    assert result[1] == []
    
    # Test with Field validator directly
    field_validator = Field(str)
    yaml_content = "'test string'"
    result = validate_yaml(yaml_content, field_validator)
    assert result[0] == "test string"
    assert result[1] == []
    
    # Test with invalid type for Field validator
    yaml_content = "123"
    result = validate_yaml(yaml_content, field_validator)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "type"
    
    # Test with whitespace-only content
    result = validate_yaml("   \n  \n", SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "no_content"


# LLM-generated content at query #24
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing with different data types
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.start == 0
    assert token.end == 9
    assert token.value == {"key": "value"}
    
    # Test with integer
    content = "number: 42"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["number"] == 42
    
    # Test with float
    content = "pi: 3.14"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["pi"] == 3.14
    
    # Test with boolean
    content = "flag: true"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["flag"] is True
    
    # Test with null
    content = "empty: null"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["empty"] is None
    
    # Test with list
    content = "- item1\n- item2"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    
    # Test with nested structure
    content = "users:\n  - name: Alice\n    age: 30"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert "users" in token.value
    assert isinstance(token.value["users"], list)
    assert token.value["users"][0]["name"] == "Alice"
    assert token.value["users"][0]["age"] == 30
    
    # Test with bytes input
    content_bytes = b"key: value"
    token = tokenize_yaml(content_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test empty content raises ParseError
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0
    
    # Test whitespace-only content raises ParseError
    try:
        tokenize_yaml("   \n  \t  \n")
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
    
    # Test invalid YAML raises ParseError
    try:
        tokenize_yaml("key: [unclosed list")
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "parse_error" in exc.code
    
    # Test scalar token
    content = "simple string"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "simple string"
    
    # Test complex mapping
    content = """
    server:
      host: localhost
      port: 8080
      ssl: true
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert "server" in token.value
    assert token.value["server"]["host"] == "localhost"
    assert token.value["server"]["port"] == 8080
    assert token.value["server"]["ssl"] is True
    
    # Test list of mixed types
    content = "- string\n- 123\n- true\n- null"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["string", 123, True, None]


# LLM-generated content at query #25
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic YAML parsing
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.start == 0
    assert token.end == 9
    assert token.value == {"key": "value"}
    
    # Test nested structures
    content = """
    outer:
      inner:
        - item1
        - item2
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert "outer" in token.value
    assert isinstance(token.value["outer"], DictToken)
    assert isinstance(token.value["outer"].value["inner"], ListToken)
    
    # Test scalar types
    content = """
    string: hello
    integer: 42
    float: 3.14
    boolean: true
    null_value: null
    """
    token = tokenize_yaml(content)
    assert isinstance(token.value["string"], ScalarToken)
    assert token.value["string"].value == "hello"
    assert isinstance(token.value["integer"], ScalarToken)
    assert token.value["integer"].value == 42
    assert isinstance(token.value["float"], ScalarToken)
    assert token.value["float"].value == 3.14
    assert isinstance(token.value["boolean"], ScalarToken)
    assert token.value["boolean"].value is True
    assert isinstance(token.value["null_value"], ScalarToken)
    assert token.value["null_value"].value is None
    
    # Test empty content
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0
    
    # Test bytes input
    content_bytes = b"key: value"
    token = tokenize_yaml(content_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test YAML parse error
    invalid_content = "key: [unclosed list"
    try:
        tokenize_yaml(invalid_content)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None
    
    # Test complex nested structure with positions
    content = """
    users:
      - name: Alice
        age: 30
      - name: Bob
        age: 25
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    users = token.value["users"]
    assert isinstance(users, ListToken)
    assert len(users.value) == 2
    assert isinstance(users.value[0], DictToken)
    assert users.value[0].value["name"].value == "Alice"
    assert users.value[0].value["age"].value == 30


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_yaml():
    # Mock yaml module to test ImportError handling
    import sys
    import typesystem.tokenize.yaml_tokenizer as yaml_tokenizer_module
    
    # Test 1: Test with pyyaml not installed
    original_yaml = yaml_tokenizer_module.yaml
    yaml_tokenizer_module.yaml = None
    
    try:
        yaml_tokenizer_module.validate_yaml("test: value", Field())
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        yaml_tokenizer_module.yaml = original_yaml
    
    # Test 2: Test basic valid YAML validation
    class TestSchema(Schema):
        name = Field(type="string")
        age = Field(type="integer")
    
    content = "name: John\nage: 30"
    result, errors = yaml_tokenizer_module.validate_yaml(content, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}
    
    # Test 3: Test validation with errors
    invalid_content = "name: John\nage: not_a_number"
    result, errors = yaml_tokenizer_module.validate_yaml(invalid_content, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "type"
    assert errors[0].text == "Must be a number."
    
    # Test 4: Test with Field validator
    field_validator = Field(type="string", max_length=5)
    content = '"hello"'
    result, errors = yaml_tokenizer_module.validate_yaml(content, field_validator)
    assert errors == []
    assert result == "hello"
    
    # Test 5: Test with Field validator error
    content = '"toolong"'
    result, errors = yaml_tokenizer_module.validate_yaml(content, field_validator)
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test 6: Test with empty YAML content
    content = ""
    result, errors = yaml_tokenizer_module.validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test 7: Test with invalid YAML syntax
    invalid_yaml = "name: John\n  age: 30"  # Bad indentation
    result, errors = yaml_tokenizer_module.validate_yaml(invalid_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    
    # Test 8: Test with bytes input
    content_bytes = b"name: Alice\nage: 25"
    result, errors = yaml_tokenizer_module.validate_yaml(content_bytes, TestSchema)
    assert errors == []
    assert result == {"name": "Alice", "age": 25}
    
    # Test 9: Test with nested schema validation
    class NestedSchema(Schema):
        user = TestSchema
    
    content = "user:\n  name: Bob\n  age: 40"
    result, errors = yaml_tokenizer_module.validate_yaml(content, NestedSchema)
    assert errors == []
    assert result == {"user": {"name": "Bob", "age": 40}}
    
    # Test 10: Test with list validation
    class ListSchema(Schema):
        items = Field(type="array", items=Field(type="integer"))
    
    content = "items:\n  - 1\n  - 2\n  - 3"
    result, errors = yaml_tokenizer_module.validate_yaml(content, ListSchema)
    assert errors == []
    assert result == {"items": [1, 2, 3]}


# LLM-generated content at query #27
#--------------------------

```python
def test_tokenize_yaml():
    # Test basic scalar tokenization
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 4
    
    # Test integer tokenization
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert isinstance(result.value, int)
    
    # Test float tokenization
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert isinstance(result.value, float)
    
    # Test boolean tokenization
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_yaml("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test null tokenization
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test list tokenization
    result = tokenize_yaml("- item1\n- item2")
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(item, ScalarToken) for item in result.value)
    assert result.value[0].value == "item1"
    assert result.value[1].value == "item2"
    
    # Test dict tokenization
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert "key" in result.value
    assert isinstance(result.value["key"], ScalarToken)
    assert result.value["key"].value == "value"
    
    # Test nested structure
    result = tokenize_yaml("key:\n  nested: value")
    assert isinstance(result, DictToken)
    nested_dict = result.value["key"]
    assert isinstance(nested_dict, DictToken)
    assert nested_dict.value["nested"].value == "value"
    
    # Test bytes input
    result = tokenize_yaml(b"test: value")
    assert isinstance(result, DictToken)
    assert result.value["test"].value == "value"
    
    # Test empty string
    try:
        tokenize_yaml("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0
    
    # Test whitespace-only string
    try:
        tokenize_yaml("   \n  \t  \n")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid YAML
    try:
        tokenize_yaml("key: [unclosed")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "parse_error" in e.code
    
    # Test complex structure with mixed types
    yaml_content = """
    name: John
    age: 30
    active: true
    scores: [95.5, 87.0, 92.3]
    metadata:
      created: 2023-01-01
      tags: [python, testing]
    """
    result = tokenize_yaml(yaml_content)
    assert isinstance(result, DictToken)
    assert result.value["name"].value == "John"
    assert result.value["age"].value == 30
    assert result.value["active"].value is True
    assert isinstance(result.value["scores"], ListToken)
    assert len(result.value["scores"].value) == 3
    assert result.value["scores"].value[0].value == 95.5
    assert isinstance(result.value["metadata"], DictToken)


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_yaml():
    # Test with valid YAML and simple field validation
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    yaml_content = "name: John\nage: 30"
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] == {"name": "John", "age": 30}
    assert result[1] == []
    
    # Test with invalid YAML (parse error)
    invalid_yaml = "name: John\n  age: 30"  # Invalid indentation
    result = validate_yaml(invalid_yaml, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "parse_error"
    
    # Test with validation error
    yaml_content = "name: John\nage: 'thirty'"  # age is string instead of int
    result = validate_yaml(yaml_content, SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "type"
    
    # Test with empty content
    result = validate_yaml("", SimpleSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "no_content"
    
    # Test with bytes input
    yaml_bytes = b"name: Alice\nage: 25"
    result = validate_yaml(yaml_bytes, SimpleSchema)
    assert result[0] == {"name": "Alice", "age": 25}
    assert result[1] == []
    
    # Test with nested schema validation
    class NestedSchema(Schema):
        user = SimpleSchema
        active = Field(bool)
    
    yaml_content = "user:\n  name: Bob\n  age: 40\nactive: true"
    result = validate_yaml(yaml_content, NestedSchema)
    assert result[0] == {"user": {"name": "Bob", "age": 40}, "active": True}
    assert result[1] == []
    
    # Test with list validation
    class ListSchema(Schema):
        items = Field(list)
    
    yaml_content = "items:\n  - apple\n  - banana\n  - cherry"
    result = validate_yaml(yaml_content, ListSchema)
    assert result[0] == {"items": ["apple", "banana", "cherry"]}
    assert result[1] == []
    
    # Test with Field validator directly
    field_validator = Field(str)
    yaml_content = "'hello world'"
    result = validate_yaml(yaml_content, field_validator)
    assert result[0] == "hello world"
    assert result[1] == []
    
    # Test with invalid field validation
    yaml_content = "42"
    result = validate_yaml(yaml_content, field_validator)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].code == "type"


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_yaml():
    # Mock yaml module to test without pyyaml installed
    import sys
    original_yaml = sys.modules.get('yaml')
    
    # Test 1: yaml not installed
    sys.modules['yaml'] = None
    try:
        from typesystem.tokenize.yaml import validate_yaml
        import pytest
        
        with pytest.raises(AssertionError) as exc_info:
            validate_yaml("test: value", Field())
        assert "'pyyaml' must be installed." in str(exc_info.value)
    finally:
        if original_yaml:
            sys.modules['yaml'] = original_yaml
        else:
            del sys.modules['yaml']
    
    # Test 2: Valid YAML with simple field validation
    from typesystem.fields import String
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String(max_length=10)
        age = String()
    
    content = "name: John\nage: 25"
    result, errors = validate_yaml(content, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": "25"}
    
    # Test 3: Invalid YAML - parse error
    invalid_content = "name: John\n  age: 25"  # Invalid indentation
    result, errors = validate_yaml(invalid_content, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"
    assert "mapping values are not allowed here" in errors[0].text.lower()
    
    # Test 4: Empty content
    result, errors = validate_yaml("", TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"
    
    # Test 5: Validation error
    content = "name: Johnathan\nage: 25"  # name too long
    result, errors = validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    assert "name" in errors[0].text
    
    # Test 6: Valid YAML with nested structure
    class NestedSchema(Schema):
        user = TestSchema
        active = String()
    
    content = "user:\n  name: John\n  age: 25\nactive: true"
    result, errors = validate_yaml(content, NestedSchema)
    assert errors == []
    assert result == {"user": {"name": "John", "age": "25"}, "active": "true"}
    
    # Test 7: Bytes input
    content_bytes = b"name: John\nage: 25"
    result, errors = validate_yaml(content_bytes, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": "25"}
    
    # Test 8: Field validator instead of Schema
    field = String(max_length=5)
    result, errors = validate_yaml("test", field)
    assert errors == []
    assert result == "test"
    
    # Test 9: Field validation error
    result, errors = validate_yaml("toolong", field)
    assert len(errors) == 1
    assert errors[0].code == "max_length"
    
    # Test 10: Complex YAML with lists
    class ListSchema(Schema):
        items = String()
    
    content = "items:\n  - item1\n  - item2\n  - item3"
    result, errors = validate_yaml(content, ListSchema)
    assert errors == []
    assert result == {"items": ["item1", "item2", "item3"]}


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_yaml():
    # Mock yaml module to test without actual pyyaml dependency
    import sys
    from unittest.mock import Mock, patch, MagicMock
    
    # Test 1: yaml is None raises assertion
    with patch.dict(sys.modules, {'yaml': None}):
        try:
            from typesystem.tokenize.yaml import validate_yaml
            # This should raise AssertionError
            validate_yaml("test", Mock())
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert str(e) == "'pyyaml' must be installed."
    
    # Test 2: Successful validation with simple schema
    mock_yaml = Mock()
    mock_safe_loader = Mock()
    
    # Create a mock token
    mock_token = Mock()
    mock_token.start = 0
    mock_token.end = 10
    
    # Mock the validator
    mock_validator = Mock()
    mock_validator.validate.return_value = ("validated_value", [])
    
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml.SafeLoader', mock_safe_loader):
            with patch('typesystem.tokenize.yaml.tokenize_yaml') as mock_tokenize:
                with patch('typesystem.tokenize.yaml.validate_with_positions') as mock_validate:
                    mock_tokenize.return_value = mock_token
                    mock_validate.return_value = ("validated_value", [])
                    
                    from typesystem.tokenize.yaml import validate_yaml
                    result = validate_yaml("key: value", mock_validator)
                    
                    assert result == ("validated_value", [])
                    mock_tokenize.assert_called_once_with("key: value")
                    mock_validate.assert_called_once_with(token=mock_token, validator=mock_validator)
    
    # Test 3: Validation with errors
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml.SafeLoader', mock_safe_loader):
            with patch('typesystem.tokenize.yaml.tokenize_yaml') as mock_tokenize:
                with patch('typesystem.tokenize.yaml.validate_with_positions') as mock_validate:
                    mock_tokenize.return_value = mock_token
                    mock_validate.return_value = (None, ["error1", "error2"])
                    
                    from typesystem.tokenize.yaml import validate_yaml
                    result = validate_yaml("invalid: yaml", mock_validator)
                    
                    assert result == (None, ["error1", "error2"])
    
    # Test 4: Bytes input
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml.SafeLoader', mock_safe_loader):
            with patch('typesystem.tokenize.yaml.tokenize_yaml') as mock_tokenize:
                with patch('typesystem.tokenize.yaml.validate_with_positions') as mock_validate:
                    mock_tokenize.return_value = mock_token
                    mock_validate.return_value = ("bytes_value", [])
                    
                    from typesystem.tokenize.yaml import validate_yaml
                    result = validate_yaml(b"bytes: input", mock_validator)
                    
                    mock_tokenize.assert_called_once_with(b"bytes: input")
    
    # Test 5: Empty content handling (should be caught by tokenize_yaml)
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml.SafeLoader', mock_safe_loader):
            with patch('typesystem.tokenize.yaml.tokenize_yaml') as mock_tokenize:
                mock_tokenize.side_effect = ParseError(
                    text="No content.", 
                    code="no_content", 
                    position=Mock()
                )
                
                from typesystem.tokenize.yaml import validate_yaml
                from typesystem.base import ParseError
                
                try:
                    validate_yaml("", mock_validator)
                    assert False, "Should have raised ParseError"
                except ParseError as e:
                    assert e.text == "No content."
                    assert e.code == "no_content"
    
    # Test 6: YAML parse error handling
    with patch('typesystem.tokenize.yaml.yaml', mock_yaml):
        with patch('typesystem.tokenize.yaml.SafeLoader', mock_safe_loader):
            with patch('typesystem.tokenize.yaml.tokenize_yaml') as mock_tokenize:
                mock_tokenize.side_effect = ParseError(
                    text="parse error.", 
                    code="parse_error", 
                    position=Mock()
                )
                
                from typesystem.tokenize.yaml import validate_yaml
                
                try:
                    validate_yaml("invalid yaml content", mock_validator)
                    assert False, "Should have raised ParseError"
                except ParseError as e:
                    assert e.text == "parse error."
                    assert e.code == "parse_error"


